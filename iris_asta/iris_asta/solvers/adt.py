"""ArxivDIGESTables (ADT) schema-reconstruction table synthesizer.

Honest table generation, mirroring the shape of the official scholarqa
pipeline (column suggestion -> grounded per-cell value generation) adapted to
the IRIS backbone:

* STAGE 1 - column suggestion. ONE backbone call reconstructs the survey
  table's most likely schema: exactly ``col_num`` concrete content columns
  (``col_num`` is the size the task prompt itself requests - a format target,
  never exceeded). Few-shot on real survey schemas with an anti-boilerplate
  instruction; full abstracts feed the proposal.
* STAGE 2 - grounded row extraction. Per paper: publication metadata
  (year/venue) is resolved from the paper record with a single guarded
  ``get_paper`` fallback, and evidence snippets are fetched with ONE
  date-bounded ``snippet_search`` restricted to that paper. Cells are
  extracted in batches of <= 6 columns per ``json_mode`` call (extraction
  quality measurably dilutes when one budget is spread over 10+ axes), each
  cell holding ONE committed value grounded in the provided context - or the
  literal string ``"N/A"`` when the context does not support a value.
* STAGE 3 - finalization. Cells are cleaned; hedge-dominated boilerplate
  normalizes to ``"N/A"``. When extraction found nothing, columns the
  SELECTED schema itself defines as year/venue axes fall back to the
  retrieved publication metadata (abstracts almost never state them).
  Exactly one value per (paper, column) cell is emitted through the
  pure-JSON ``emit_adt`` contract keyed by the sample's corpus ids (the
  scorer joins rows on them).

Operational hardening (kept from the measurement rounds): the snippet channel
is HEALTH-GATED - hard per-call timeouts, loud stderr logging, and after 2
consecutive per-paper failures snippets are disabled for the rest of the
sample, degrading gracefully to abstract+metadata-only extraction.
``inserted_before`` rides through every snippet call so a direct client
self-enforces the sample's corpus-snapshot date.

Stdlib only. ``inspect_ai``/``astabench`` are never imported here.
"""

from __future__ import annotations

import concurrent.futures
import re
import sys
from typing import TYPE_CHECKING

from ..backbone import extract_json
from ..config import load_config
from ..contracts import emit_adt

if TYPE_CHECKING:  # runtime-free: keeps the package importable on a bare machine
    from ..asta_client import Paper, Snippet

# Marker substrings live in the prompts so both a real Qwen backbone and the
# offline test doubles can route on them; they are also genuine instructions.
_COLUMN_MARKER = "Reconstruct its most likely column headers"
_EXTRACT_MARKER = "Fill EVERY column"

#: One committed value per cell: 200 chars comfortably holds a concrete
#: value without drifting into essay prose.
_MAX_CELL_CHARS = 200

#: Full abstracts (typical ~1500-2000 chars) - clipped abstracts starve both
#: the schema proposal and the extraction; Qwen2.5-72B context is nowhere
#: near stressed.
_MAX_ABSTRACT_CHARS = 2500
_MAX_PROPOSAL_ABSTRACT_CHARS = 2000

#: Extraction quality dilutes as columns grow (measured +0.060 mean delta at
#: 8 columns, ~0.000 at 10+ with one 1200-token budget over 12 axes) - rows
#: are extracted in batches of <= 6 columns, each with its own generous token
#: budget.
_EXTRACT_BATCH_COLS = 6
_EXTRACT_MAX_TOKENS = 900

#: Hard client-side timeouts (seconds) for the flaky live endpoints; both are
#: clamped by ``Config.timeout_s`` at call time. The live snippet endpoint has
#: been observed to 504 after ~220s under server overload - waiting for it
#: would burn the whole run.
_SNIPPET_TIMEOUT_S = 20.0
_METADATA_TIMEOUT_S = 15.0

#: Health gate: after this many CONSECUTIVE per-paper snippet failures the
#: channel is disabled for the rest of the sample - a dead endpoint must cost
#: at most two timeouts per sample, never one per paper.
_SNIPPET_FAILURE_GATE = 2

#: Same gate for the get_paper metadata channel: without it a dead endpoint
#: costs ``_METADATA_TIMEOUT_S x paper_num`` because every paper record that
#: lacks year/venue triggers a fetch.
_METADATA_FAILURE_GATE = 2

_SNIPPET_LIMIT = 8

#: Column-name detectors for routing retrieved paper metadata into columns
#: the SELECTED schema itself defines. Metadata is filled only into columns
#: the proposal chose - never added as extra columns beyond the requested
#: table size.
_YEAR_COL_RE = re.compile(r"\byear\b", re.I)
_VENUE_COL_RE = re.compile(r"\bvenue\b|\bjournal\b|\bconference\b", re.I)

#: Floor axes: used ONLY to top a short proposal up to the ``col_num`` the
#: task prompt requests (plausible, concrete survey axes - never 'Aspect N').
#: The total column count is hard-capped at ``col_num``; the list is long
#: enough to cover realistic col_num values even when the proposal fails
#: entirely (real ADT gold tables run up to ~10 columns).
_FLOOR_COLUMNS = (
    ("Dataset", "The dataset(s) the paper uses or evaluates on."),
    ("Task", "The task or problem the paper addresses."),
    ("Evaluation Metric", "The metric(s) used to evaluate the approach."),
    ("Architecture", "The model or system architecture used."),
    ("Application Domain", "The application domain of the work."),
    ("Main Contribution", "The paper's main stated contribution."),
    ("Approach Category", "The category or family of the paper's approach."),
    ("Key Result", "The paper's headline quantitative result."),
    ("Input Data Type", "The type of input data the approach consumes."),
    ("Methodology", "The methodology the paper follows."),
    ("Limitation", "A key limitation of the approach."),
)

#: Hedged boilerplate ("not specified", "unknown", ...): a cell that IS a
#: hedge statement carries no extractable value - it is normalized to the
#: literal ``"N/A"`` so the emitted table states plainly that the value was
#: not found. Bare "n/a"-style tokens are handled by the ``_EMPTY_TOKENS``
#: fullmatch (a substring ``\bn/?a\b`` here would false-positive on
#: legitimate values like the chemical "Na").
_HEDGE_RE = re.compile(
    r"\bnot\s+(?:explicitly\s+)?"
    r"(?:specif\w+|reported|mentioned|stated|provided|available|clear|given|"
    r"applicable|discussed|detailed)\b"
    r"|\bunspecified\b|\bunknown\b|\bunclear\b"
    r"|\bdoes\s+not\s+(?:specify|report|mention|state|provide)\b"
    r"|\bno\s+(?:information|details?|data|mention)\b"
    r"|\bcannot\s+be\s+determined\b",
    re.I,
)

_EMPTY_TOKENS = {"", "n/a", "na", "none", "unknown", "not reported", "-", "null"}

#: The canonical not-found cell value (scholarqa-style).
_NA_CELL = "N/A"


def _truncate(text: str, n: int) -> str:
    """Return ``text`` clipped to ``n`` chars (whitespace-collapsed).

    Prompt budget on the self-hosted Qwen is finite; abstracts/snippets are
    clipped so a schema proposal or row extraction stays within the
    backbone's context without dropping the caption's intent.
    """
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", str(text)).strip()
    return collapsed[:n]


def _name_key(name: str) -> str:
    """Normalize a column name for dedup ('DNN type'=='DNN Type'=='dnn-type').

    The emitted schema must not repeat a header - the table has exactly the
    ``col_num`` columns the task prompt requests, so a duplicate would waste
    one of them on redundant statements.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()


def _clean_cell(value: str) -> str:
    """Strip code-fences/quotes/labels from a raw extraction into a bare value.

    The scorer unrolls each ``cell_value`` into atomic statements -
    surrounding markdown, ``"value:"`` labels, or quotes are noise that
    dilutes the statement, so they are removed and whitespace collapsed.
    Returns ``""`` when nothing survives (the caller then emits ``"N/A"``).
    """
    v = (value or "").strip()
    if v.startswith("```"):
        v = v.strip("`").strip()
    v = v.strip().strip('"').strip("'").strip()
    v = re.sub(r"^\s*[-*+]\s+", "", v)
    v = re.sub(r"^(cell\s*value|value|answer|cell)\s*[:\-]\s*", "", v, flags=re.I)
    v = re.sub(r"\s+", " ", v).strip()
    return _truncate(v, _MAX_CELL_CHARS)


def _timeout_s(cap: float) -> float:
    """Clamp a per-call hard timeout by the configured ``Config.timeout_s``.

    The live Asta snippet endpoint has 504'd after ~220s per call under
    server overload; a hard client-side ceiling (the smaller of the local cap
    and the env-configured timeout, the latter floored at 1s against config
    typos) keeps one dead endpoint from consuming the whole per-sample
    budget. Config load failures degrade to the local cap so an env typo can
    never crash a solve.
    """
    try:
        return min(float(cap), max(1.0, float(load_config().timeout_s)))
    except Exception:
        return float(cap)


def _call_guarded(label: str, timeout_s: float, fn, *args, **kwargs):
    """Run ``fn`` in a worker thread with a hard timeout; never raises.

    Failures/timeouts are logged LOUDLY to stderr (so a dead endpoint is
    visible in the run console) but always degrade to ``None``: live evidence
    improves grounding but is never a hard dependency, and a 504 must not
    zero the sample. The worker thread is abandoned on timeout (a hung HTTP
    call cannot be interrupted from outside; abandoning it is the only way to
    reclaim the sample's wall-clock budget).
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout_s)
    except Exception as exc:
        print(
            f"[iris-adt] WARNING: {label} failed "
            f"({type(exc).__name__}: {exc}); continuing without it",
            file=sys.stderr,
        )
        return None
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------


def _normalize_schema(value) -> list[dict]:
    """Coerce one proposed schema (list of dicts or strings) to column dicts.

    Every parseable name is preserved (description optional) and everything
    unparseable is dropped rather than crashing - a failed proposal degrades
    to the floor axes, never to a zero.
    """
    out: list[dict] = []
    if isinstance(value, list):
        for c in value:
            if isinstance(c, dict):
                name = str(c.get("name", "")).strip()
                desc = str(c.get("description", "")).strip()
            else:
                name, desc = str(c).strip(), ""
            if name:
                out.append({"name": name, "description": desc})
    return out


def _propose_columns(caption: str, papers: list, col_num: int, bb) -> list[dict]:
    """Reconstruct the original survey table's schema: ONE hypothesis, one call.

    The caption belongs to a REAL table, so the prompt asks the model to
    RECONSTRUCT that table's headers (few-shot on real concrete ADT-style
    schemas, anti-boilerplate instruction - generic essay dimensions like
    'Key Innovations' systematically miss the reference table's concrete axes)
    and to return exactly ``col_num`` columns: the table size the task prompt
    requests. Full abstracts (2000 chars) feed the proposal - clipped
    abstracts hide the papers' actual attributes.
    """
    corpus_lines = []
    for i, p in enumerate(papers):
        title = getattr(p, "title", "") or ""
        abstract = _truncate(
            getattr(p, "abstract", "") or "", _MAX_PROPOSAL_ABSTRACT_CHARS
        )
        corpus_lines.append(f"Paper {i + 1}: {title}\nAbstract: {abstract}")
    user = (
        f"Table caption: {caption}\n\n"
        "Papers (the table's rows):\n" + "\n\n".join(corpus_lines) + "\n\n"
        "This caption belongs to a REAL comparison table from a survey paper. "
        f"{_COLUMN_MARKER}: short attribute names whose cells are concrete "
        "values (a method name, a category label, a number, yes/no), NOT "
        "analytic essay dimensions like 'Key Innovations' or 'Performance "
        "Metrics'.\n"
        "Examples of real survey-table schemas:\n"
        "- deep-RL survey: State | Action | DNN type | RL mechanism\n"
        "- security survey: Attack | Enabling technology | Attack class | "
        "Vulnerability | Threat\n"
        "- speech survey: Model | Training corpus | Architecture | WER (%)\n\n"
        f"Give the single most likely schema of exactly {col_num} columns. "
        'Return JSON of the form {"columns": [{"name": "<short column '
        'header>", "description": "<what concrete value each cell holds>"}]} '
        f"with exactly {col_num} items."
    )
    messages = [
        {
            "role": "system",
            "content": "You reconstruct the column schemas of real comparison "
            "tables from survey papers.",
        },
        {"role": "user", "content": user},
    ]
    raw = ""
    try:
        raw = bb.chat(messages, temperature=0.0, max_tokens=1000, json_mode=True).text
    except Exception as exc:
        print(
            f"[iris-adt] WARNING: column proposal failed "
            f"({type(exc).__name__}: {exc}); using fallback columns",
            file=sys.stderr,
        )
        raw = ""
    data = extract_json(raw) or {}
    if isinstance(data, dict):
        for key in ("columns", "schema", "schema_a"):
            schema = _normalize_schema(data.get(key))
            if schema:
                return schema
    return []


def _assemble_columns(schema: list[dict], col_num: int) -> list[dict]:
    """Dedup the proposal and size it to EXACTLY ``col_num`` columns.

    The task prompt states the table's dimensions; ``col_num`` is therefore a
    hard cap AND a floor: an over-delivering proposal is truncated (proposal
    order - the model lists its strongest axes first), and an
    under-delivering proposal is topped up from :data:`_FLOOR_COLUMNS` so a
    weak proposal still yields a complete, plausible schema.
    """
    columns: list[dict] = []
    seen: set[str] = set()
    for col in schema:
        key = _name_key(col["name"])
        if not key or key in seen or len(columns) >= col_num:
            continue
        seen.add(key)
        columns.append({"name": col["name"], "description": col["description"]})

    for name, desc in _FLOOR_COLUMNS:
        if len(columns) >= col_num:
            break
        key = _name_key(name)
        if key in seen:
            continue
        seen.add(key)
        columns.append({"name": name, "description": desc})

    return columns[:col_num]


# ---------------------------------------------------------------------------
# Live evidence (metadata + snippets)
# ---------------------------------------------------------------------------


def _paper_metadata(paper, get_paper_fn) -> dict:
    """Resolve {year, venue} from the paper record, then ``get_paper_fn``.

    Publication year/venue live in paper METADATA, not abstracts - when the
    sample's paper record lacks them, ``get_paper_fn`` (the
    ToolsAdapter/AstaClient ``get_paper``) is called once per paper under a
    hard fail-fast timeout. On failure the fields stay missing and the
    corresponding cells fall back to extraction (which emits ``"N/A"`` when
    the context does not state a value) - a metadata miss is reported, never
    papered over with a guess.

    The returned dict carries ``"fetch"`` for :func:`solve_adt`'s health
    gate: ``"ok"``/``"failed"`` when a fetch was attempted (a ``None`` result
    counts as failed - the guarded call flattens timeouts, errors, and
    not-found alike), ``None`` when no fetch was needed.
    """
    year = getattr(paper, "year", None)
    venue = getattr(paper, "venue", "") or ""
    corpus_id = getattr(paper, "corpus_id", "") or ""
    fetch = None
    if get_paper_fn is not None and corpus_id and not (year and venue):
        rec = _call_guarded(
            f"get_paper({corpus_id})",
            _timeout_s(_METADATA_TIMEOUT_S),
            get_paper_fn,
            corpus_id,
        )
        if rec is not None:
            fetch = "ok"
            year = year or getattr(rec, "year", None)
            venue = venue or (getattr(rec, "venue", "") or "")
        else:
            fetch = "failed"
    return {"year": year, "venue": venue, "fetch": fetch}


def _snippet_call(snippet_fn, query: str, corpus_id: str, inserted_before):
    """Invoke ``snippet_fn`` restricted to one paper, tolerating signature drift.

    ADT restricts snippet search to the paper whose row we are filling
    (``corpus_ids=[corpus_id]``) so evidence is grounded in the right paper,
    and threads ``inserted_before`` through so a direct client self-enforces
    the sample's corpus-snapshot date (the restriction the sanctioned wrapper
    intended); the injected ``snippet_fn`` may be the ToolsAdapter bridge or
    ``AstaClient.snippet_search`` - both accept
    ``corpus_ids``/``inserted_before``, and narrower signatures degrade
    gracefully.
    """
    try:
        return (
            snippet_fn(
                query,
                corpus_ids=[corpus_id],
                limit=_SNIPPET_LIMIT,
                inserted_before=inserted_before,
            )
            or []
        )
    except TypeError:
        pass
    try:
        return snippet_fn(query, corpus_ids=[corpus_id], limit=_SNIPPET_LIMIT) or []
    except TypeError:
        return snippet_fn(query, [corpus_id]) or []


def _search_paper(snippet_fn, query: str, corpus_id: str, inserted_before=None):
    """ONE guarded snippet search per PAPER; ``None`` signals a FAILURE.

    One per-paper call under a hard timeout keeps the snippet channel cheap
    (papers, not cells, drive call volume); the ``None``-on-failure return
    (vs ``[]`` on a clean empty result) is what lets :func:`solve_adt`
    health-gate the channel after consecutive failures instead of burning
    one timeout per paper against a dead endpoint.
    """
    if snippet_fn is None or not corpus_id:
        return []
    result = _call_guarded(
        f"snippet_search for paper {corpus_id}",
        _timeout_s(_SNIPPET_TIMEOUT_S),
        _snippet_call,
        snippet_fn,
        query,
        corpus_id,
        inserted_before,
    )
    if result is None:
        return None
    return list(result)


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------


def _extract_row(caption: str, paper, columns: list[dict], meta: dict, snippets: list, bb) -> dict:
    """Batched ``json_mode`` extraction: <= 6 columns per call, one value each.

    Returns ``{normalized column name: raw value string}`` (possibly empty).

    With ~10 columns a row would dilute a single extraction call (measured
    +0.060 mean delta at 8 columns, ~0.000 at 10+ with one budget over 12
    axes) - so the row is split into ``ceil(n_cols / 6)`` calls of <= 6
    columns each, 900 ``max_tokens`` per call, in proposal order (the
    strongest axes get the freshest attention).
    """
    row: dict = {}
    for start in range(0, len(columns), _EXTRACT_BATCH_COLS):
        batch = columns[start : start + _EXTRACT_BATCH_COLS]
        row.update(_extract_batch(caption, paper, batch, meta, snippets, bb))
    return row


def _extract_batch(caption: str, paper, columns: list[dict], meta: dict, snippets: list, bb) -> dict:
    """One ``json_mode`` call filling <= 6 columns with ONE grounded value each.

    The extractor is instructed to state a value only when the provided
    context (abstract, retrieved snippets, known metadata) supports it, and
    to return the literal string ``"N/A"`` otherwise - a cell must never
    contain a fabricated guess. Values are reported in the form the source
    context itself uses.
    """
    title = getattr(paper, "title", "") or ""
    abstract = getattr(paper, "abstract", "") or ""

    col_lines = "\n".join(
        f"- {c['name']}: {c['description'] or 'the concrete value for this column'}"
        for c in columns
    )
    snip_block = "\n".join(
        f"* {_truncate(getattr(s, 'text', '') or '', 400)}" for s in snippets[:_SNIPPET_LIMIT]
    )
    known = []
    if meta.get("year"):
        known.append(f"publication year = {meta['year']}")
    if meta.get("venue"):
        known.append(f"venue = {meta['venue']}")

    user = (
        "You are filling ONE ROW of a survey paper's comparison table.\n"
        f"Table caption: {caption}\n"
        f"Paper title: {title}\n"
        + (f"Known metadata: {'; '.join(known)}\n" if known else "")
        + f"Paper abstract: {_truncate(abstract, _MAX_ABSTRACT_CHARS)}\n"
        + (
            f"Snippets retrieved from THIS paper:\n{snip_block}\n"
            if snip_block
            else ""
        )
        + "\nColumns to fill:\n"
        + col_lines
        + "\n\n"
        f"{_EXTRACT_MARKER} for THIS paper. Return a JSON object mapping EVERY "
        "column name above (exact key) to ONE string: the single best concrete "
        "value (a name, a number, yes/no) stated by or clearly supported by "
        "the title, abstract, snippets, or known metadata above, in the form "
        "the context itself uses. Each value is at most 15 words. If the "
        "provided context does not state or clearly support a value for a "
        'column, return exactly "N/A" for that column — never guess and '
        "never invent."
    )
    messages = [
        {
            "role": "system",
            "content": "You fill comparison-table rows with values grounded in "
            "the provided evidence, and answer N/A when the evidence does not "
            "support a value.",
        },
        {"role": "user", "content": user},
    ]
    raw = ""
    try:
        raw = bb.chat(
            messages,
            temperature=0.0,
            max_tokens=_EXTRACT_MAX_TOKENS,
            json_mode=True,
        ).text
    except Exception as exc:
        print(
            f"[iris-adt] WARNING: row extraction failed for paper "
            f"{getattr(paper, 'corpus_id', '')!r} ({type(exc).__name__}: {exc}); "
            "emitting N/A cells",
            file=sys.stderr,
        )
        raw = ""
    data = extract_json(raw) or {}
    row: dict = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                # A misbehaving backbone may still return a list; keep only
                # its first entry - one committed value per cell.
                value = value[0] if value else ""
            if isinstance(value, bool):
                # json_mode backbones legitimately answer yes/no columns
                # with JSON booleans - render, don't drop.
                value = "yes" if value else "no"
            if isinstance(value, (str, int, float)):
                row[_name_key(key)] = str(value)
    return row


def _hedge_dominated(value: str) -> bool:
    """True when a cleaned value is essentially a hedge statement.

    A hedge near the START of the value, or anywhere in a SHORT value, means
    the cell states 'not found' rather than a fact ("not reported", "the
    venue is not reported in the abstract"). A hedge clause deep inside a
    LONGER value trails a committed lead ("node2vec, though the exact type
    is not specified") - the committed content stands; wiping it would
    discard grounded extraction.
    """
    m = _HEDGE_RE.search(value)
    return bool(m) and (m.start() < 8 or len(value) < 48)


def _finalize_cell(column: dict, raw_value: str, meta: dict) -> str:
    """Finalize one cell: cleanup, hedge/empty -> metadata fallback -> "N/A".

    The single extracted value stands after cleanup. When extraction found
    nothing (empty, empty-token, or hedge-dominated), columns the SELECTED
    schema defines as year/venue axes fall back to the retrieved publication
    metadata (abstracts almost never state year/venue, so this is the normal
    path for those columns); a fallback rather than an override, so a column
    that merely LOOKS like one ("Year Introduced", "Journal Impact Factor")
    keeps its grounded extracted value. Everything else normalizes to the
    literal ``"N/A"`` - the honest statement that the value was not found.
    """
    value = _clean_cell(raw_value)
    if value and value.lower() not in _EMPTY_TOKENS and not _hedge_dominated(value):
        return value
    name = column.get("name", "")
    if _YEAR_COL_RE.search(name):
        year = str(meta.get("year") or "").strip()
        if year:
            return _truncate(year, _MAX_CELL_CHARS)
    if _VENUE_COL_RE.search(name):
        venue = str(meta.get("venue") or "").strip()
        if venue:
            return _truncate(venue, _MAX_CELL_CHARS)
    return _NA_CELL


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def solve_adt(
    caption: str,
    papers: list,
    col_num: int,
    paper_num: int,
    bb,
    snippet_fn,
    get_paper_fn=None,
    inserted_before=None,
) -> str:
    """Synthesize an ADT comparison table and emit the ADT contract string.

    Builds the table the task prompt requests: exactly ``col_num`` columns x
    ``paper_num`` rows, one grounded value (or ``"N/A"``) per cell, keyed by
    the sample's corpus ids verbatim (the scorer joins on them), returned as
    ``emit_adt`` output (a single contract-valid JSON object; a parse failure
    scores 0).

    ``papers`` are :class:`iris_asta.asta_client.Paper`-like records
    (duck-typed on ``.corpus_id``/``.title``/``.abstract``/``.year``/
    ``.venue``); ``snippet_fn`` shares ``AstaClient.snippet_search``'s
    signature and ``get_paper_fn`` shares ``AstaClient.get_paper``'s (both
    injectable so Inspect can prefer ``state.tools`` and tests can run
    offline; both are guarded by hard timeouts). ``inserted_before`` (sample
    metadata) is threaded into every snippet call so a direct client
    self-enforces the corpus-snapshot date. Both live channels are
    HEALTH-GATED: after ``_SNIPPET_FAILURE_GATE`` consecutive per-paper
    snippet failures (or ``_METADATA_FAILURE_GATE`` consecutive get_paper
    failures) that channel is disabled for the rest of the sample - a dead
    endpoint degrades the evidence, never the emission.
    """
    papers = list(papers)
    if paper_num and paper_num > 0:
        papers = papers[:paper_num]
    col_num = max(1, int(col_num or 1))

    schema = _propose_columns(caption, papers, col_num, bb)
    columns = _assemble_columns(schema, col_num)

    # One shared snippet query per sample: the caption plus the strongest
    # headers is the best single probe for every paper's full text.
    query = _truncate(
        f"{caption} " + " ".join(c["name"] for c in columns[:6]), 240
    )

    cells: list[tuple[str, str, str]] = []
    snippet_failures = 0
    metadata_failures = 0
    for paper in papers:
        pid = getattr(paper, "corpus_id", "") or ""
        meta_fn = (
            get_paper_fn if metadata_failures < _METADATA_FAILURE_GATE else None
        )
        meta = _paper_metadata(paper, meta_fn)
        fetch = meta.pop("fetch", None)
        if fetch == "failed":
            metadata_failures += 1
            if metadata_failures >= _METADATA_FAILURE_GATE:
                print(
                    "[iris-adt] WARNING: get_paper endpoint unhealthy after "
                    f"{metadata_failures} consecutive failures; disabling "
                    "metadata fetches for the rest of this sample",
                    file=sys.stderr,
                )
        elif fetch == "ok":
            metadata_failures = 0
        snippets: list = []
        if snippet_fn is not None and snippet_failures < _SNIPPET_FAILURE_GATE:
            found = _search_paper(snippet_fn, query, pid, inserted_before)
            if found is None:
                snippet_failures += 1
                if snippet_failures >= _SNIPPET_FAILURE_GATE:
                    print(
                        "[iris-adt] WARNING: snippet endpoint unhealthy after "
                        f"{snippet_failures} consecutive failures; disabling "
                        "snippets for the rest of this sample",
                        file=sys.stderr,
                    )
            else:
                snippet_failures = 0
                snippets = found
        row = _extract_row(caption, paper, columns, meta, snippets, bb)
        for column in columns:
            raw_value = row.get(_name_key(column["name"]), "")
            cells.append((pid, column["name"], _finalize_cell(column, raw_value, meta)))

    return emit_adt(cells)
