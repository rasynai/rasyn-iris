"""Output-contract emitters for ASTABench tasks (stdlib only).

Every emitter returns a **single pure-JSON object string** — nothing before the
first ``{`` and nothing after the last ``}`` — because the benchmark-side
parsers take the first-``{``-to-last-``}`` slice of ``state.output.completion``
and a parse/validation failure scores 0 on SQA / PFB / ADT / E2E /
DiscoveryBench (LitQA2 parse failure = wrong answer, no partial credit).
The E2E scorer additionally runs STRICT ``json.loads`` on the whole
completion, so emitted strings must be pure JSON with no prose contamination.

All serialization uses ``ensure_ascii=True`` so the emitted string survives any
transport encoding intact.
"""

from __future__ import annotations

import json
import string

#: PFB scorer caps submissions at 250 results; extras can only dilute nDCG.
PFB_MAX_RESULTS = 250

#: E2E scorer truncates each artifact at 10000 chars; we truncate at 9500
#: defensively so scorer-side truncation never lands mid-escape or mid-payload.
E2E_ARTIFACT_CAP = 9500

_CORPUSID_PREFIX = "corpusid:"


def _dumps(obj: dict) -> str:
    """Serialize ``obj`` to a pure, ASCII-safe JSON object string.

    Benchmark fact served: scorers parse the completion as JSON (E2E even with
    strict ``json.loads`` over the whole string) — the emitted string must
    round-trip ``json.loads`` with NOTHING else in it.
    """
    return json.dumps(obj, ensure_ascii=True)


def _s(value) -> str:
    """Coerce a field value to ``str`` (``None`` becomes ``""``).

    Benchmark fact served: contract fields must be JSON strings; emitting
    ``null``/numbers where the scorer expects strings risks a validation
    failure that scores 0.
    """
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


# ---------------------------------------------------------------------------
# LitQA2
# ---------------------------------------------------------------------------

def emit_litqa2(letter: str) -> str:
    """Emit the LitQA2 answer contract ``{"answer": "<letter>"}``.

    Benchmark fact served: the LitQA2 scorer accepts exactly
    ``{"answer": "<letter>"}``; a parse failure counts as wrong with no
    partial credit, so this asserts a single A-Z letter (whitespace stripped,
    case-normalized to uppercase) and raises ``ValueError`` otherwise.
    """
    if letter is None:
        raise ValueError("emit_litqa2 requires a single A-Z letter, got None")
    normalized = str(letter).strip().upper()
    if len(normalized) != 1 or normalized not in string.ascii_uppercase:
        raise ValueError(
            "emit_litqa2 requires a single A-Z letter, got %r" % (letter,)
        )
    return _dumps({"answer": normalized})


# ---------------------------------------------------------------------------
# PaperFindingBench
# ---------------------------------------------------------------------------

def normalize_corpus_id(raw) -> str | None:
    """Normalize a raw paper id to a digits-only Semantic Scholar corpus id.

    Benchmark fact served: the PFB scorer matches ``paper_id`` against gold
    Semantic Scholar corpus ids as DIGITS-ONLY strings — any prefix (e.g.
    ``"CorpusId:123"``) or non-digit id simply never matches gold and wastes a
    ranked slot. Normalization: lowercase first, strip a ``"corpusid:"``
    prefix, strip whitespace, canonicalize leading zeros via ``int``.
    Returns ``None`` when the id cannot be repaired to digits-only.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s.startswith(_CORPUSID_PREFIX):
        s = s[len(_CORPUSID_PREFIX):].strip()
    if not s or not s.isascii() or not s.isdigit():
        return None
    return str(int(s))


def emit_pfb(results: list[tuple[str, str]]) -> str:
    """Emit the PaperFindingBench contract from ``[(corpus_id, evidence)]``.

    Benchmark facts served (mirroring the official scorer — deviation = 0):
    - shape is ``{"output": {"results": [{"paper_id", "markdown_evidence"}]}}``;
    - ``paper_id`` must be the Semantic Scholar corpus_id as a digits-only
      string (``"corpusid:"`` prefixes stripped after lowercasing) — entries
      whose id cannot be repaired to digits-only are dropped;
    - submitted order IS the relevance ranking (nDCG uses it) — input order is
      preserved exactly;
    - duplicates are deduplicated keeping the FIRST occurrence, then the list
      is capped at 250 results (dedup before cap so duplicates never waste
      ranked slots).
    """
    seen: set[str] = set()
    out: list[dict] = []
    for corpus_id, evidence in results or []:
        cid = normalize_corpus_id(corpus_id)
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        out.append({"paper_id": cid, "markdown_evidence": _s(evidence)})
        if len(out) >= PFB_MAX_RESULTS:
            break
    return _dumps({"output": {"results": out}})


# ---------------------------------------------------------------------------
# ScholarQA (SQA)
# ---------------------------------------------------------------------------

def _insert_id_into_text(text: str, citation_id: str) -> str:
    """Insert ``citation_id`` so it appears verbatim in ``text``.

    Benchmark fact served: the SQA scorer matches citations to sentences by
    substring — an id absent from the section text contributes nothing. The
    id is inserted just before the final terminal punctuation so it attaches
    to the last real sentence rather than dangling as its own fragment.
    """
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".!?":
        return stripped[:-1].rstrip() + " " + citation_id + stripped[-1]
    return (stripped + " " + citation_id).strip()


def _normalize_sqa(sections, repair: bool) -> list[dict]:
    """Validate (raise) or repair SQA sections into contract shape.

    Enforces both scorer traps: (1) each citation id must appear verbatim in
    its section text; (2) snippets must NOT be verbatim substrings of the
    section text (the scorer demotes such citations to title-only half
    credit).
    """
    if not isinstance(sections, list):
        raise ValueError("emit_sqa: sections must be a list of dicts")
    norm_sections: list[dict] = []
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            if repair:
                continue
            raise ValueError("emit_sqa: section %d is not a dict" % i)

        title = sec.get("title", "")
        if not isinstance(title, str):
            if not repair:
                raise ValueError("emit_sqa: section %d 'title' must be a str" % i)
            title = _s(title)

        text = sec.get("text", "")
        if not isinstance(text, str):
            if not repair:
                raise ValueError("emit_sqa: section %d 'text' must be a str" % i)
            text = _s(text)

        citations = sec.get("citations", [])
        if not isinstance(citations, list):
            if not repair:
                raise ValueError("emit_sqa: section %d 'citations' must be a list" % i)
            citations = []

        # Pass 1: normalize ids and repair the text so every kept citation id
        # appears verbatim inside its section text.
        kept: list[tuple[str, dict]] = []
        for j, cit in enumerate(citations):
            if not isinstance(cit, dict):
                if repair:
                    continue
                raise ValueError(
                    "emit_sqa: section %d citation %d is not a dict" % (i, j)
                )
            cid = cit.get("id", "")
            if not isinstance(cid, str):
                if not repair:
                    raise ValueError(
                        "emit_sqa: section %d citation %d 'id' must be a str" % (i, j)
                    )
                cid = _s(cid)
            cid = cid.strip()
            if not cid:
                if repair:
                    continue
                raise ValueError(
                    "emit_sqa: section %d citation %d has an empty id" % (i, j)
                )
            if cid not in text:
                if not repair:
                    raise ValueError(
                        "emit_sqa: citation id %r does not appear verbatim in its "
                        "section text (scorer matches citations to sentences by "
                        "substring; unmatched citations score nothing)" % cid
                    )
                text = _insert_id_into_text(text, cid)
            kept.append((cid, cit))

        # Pass 2: snippets checked against the FINAL text (after id repairs).
        norm_citations: list[dict] = []
        for cid, cit in kept:
            snippets = cit.get("snippets", [])
            if isinstance(snippets, str):
                if not repair:
                    raise ValueError(
                        "emit_sqa: citation %r 'snippets' must be a list of str" % cid
                    )
                snippets = [snippets]
            if not isinstance(snippets, list):
                if not repair:
                    raise ValueError(
                        "emit_sqa: citation %r 'snippets' must be a list" % cid
                    )
                snippets = []
            norm_snippets: list[str] = []
            for sn in snippets:
                if not isinstance(sn, str):
                    if not repair:
                        raise ValueError(
                            "emit_sqa: citation %r has a non-str snippet" % cid
                        )
                    sn = _s(sn)
                if sn in text:  # includes the empty snippet ("" in text is True)
                    if repair:
                        continue
                    raise ValueError(
                        "emit_sqa: a snippet of citation %r is a verbatim substring "
                        "of its section text — the scorer demotes such citations to "
                        "title-only half credit" % cid
                    )
                norm_snippets.append(sn)

            ctitle = cit.get("title", "")
            if not isinstance(ctitle, str):
                if not repair:
                    raise ValueError(
                        "emit_sqa: citation %r 'title' must be a str" % cid
                    )
                ctitle = _s(ctitle)

            metadata = cit.get("metadata", {})
            if not isinstance(metadata, dict):
                if not repair:
                    raise ValueError(
                        "emit_sqa: citation %r 'metadata' must be a dict" % cid
                    )
                metadata = {}

            norm_citations.append(
                {
                    "id": cid,
                    "snippets": norm_snippets,
                    "title": ctitle,
                    "metadata": metadata,
                }
            )

        norm_sections.append(
            {"title": title, "text": text, "citations": norm_citations}
        )
    return norm_sections


def validate_sqa_sections(sections: list[dict]) -> list[dict]:
    """Strict SQA validator: raises ``ValueError`` on any scorer trap.

    Benchmark facts served: (1) each citation id must appear verbatim inside
    its section text (the scorer matches citations to sentences by substring);
    (2) snippets must NOT be verbatim substrings of the section text (the
    scorer demotes such citations to title-only half credit). Returns the
    normalized sections on success.
    """
    return _normalize_sqa(sections, repair=False)


def repair_sqa_sections(sections: list[dict]) -> list[dict]:
    """Repairing SQA normalizer: fixes scorer traps instead of raising.

    Benchmark facts served: missing citation ids are inserted into the section
    text just before its final punctuation (the scorer matches citations to
    sentences by substring); snippets that are verbatim substrings of the
    section text are dropped (the scorer demotes such citations to title-only
    half credit — dropping the offending snippet is never worse). Unusable
    citations/sections (non-dict, empty id) are dropped.
    """
    return _normalize_sqa(sections, repair=True)


def emit_sqa(sections: list[dict], repair: bool = True) -> str:
    """Emit the SQA contract ``{"sections": [...]}`` from section dicts.

    Benchmark facts served: shape is ``{"sections": [{"title", "text",
    "citations": [{"id", "snippets", "title", "metadata"}]}]}``; each citation
    id must appear verbatim inside its section text (scorer matches citations
    to sentences by substring) and snippets must NOT be verbatim substrings of
    the section text (scorer demotes such citations to title-only half
    credit). With ``repair=True`` (default — a raise in production scores 0)
    violations are repaired via :func:`repair_sqa_sections`; with
    ``repair=False`` they raise via :func:`validate_sqa_sections`.
    """
    return _dumps({"sections": _normalize_sqa(sections, repair=repair)})


# ---------------------------------------------------------------------------
# ArxivDIGESTables (ADT)
# ---------------------------------------------------------------------------

def emit_adt(cells: list[tuple[str, str, str]]) -> str:
    """Emit the ADT contract from ``[(paper_id, column_name, cell_value)]``.

    Benchmark fact served: the ArxivDIGESTables scorer expects
    ``{"cell_values": [{"paper_id", "column_name", "cell_value"}]}`` — a parse
    or shape failure scores 0, so all fields are coerced to strings and the
    result is a single pure-JSON object.
    """
    values = [
        {
            "paper_id": _s(paper_id),
            "column_name": _s(column_name),
            "cell_value": _s(cell_value),
        }
        for paper_id, column_name, cell_value in cells or []
    ]
    return _dumps({"cell_values": values})


# ---------------------------------------------------------------------------
# DiscoveryBench
# ---------------------------------------------------------------------------

def emit_discoverybench(hypothesis: str, workflow: str) -> str:
    """Emit the DiscoveryBench contract ``{"hypothesis", "workflow"}``.

    Benchmark fact served: the DiscoveryBench scorer expects a FLAT object
    with exactly ``hypothesis`` and ``workflow`` string fields (no wrapper
    key); a parse failure scores 0.
    """
    return _dumps({"hypothesis": _s(hypothesis), "workflow": _s(workflow)})


# ---------------------------------------------------------------------------
# E2E discovery
# ---------------------------------------------------------------------------

def emit_e2e(
    report: str,
    code: list[tuple[str, str]],
    artifacts: list[tuple[str, str, str]],
    trace: str = "",
) -> str:
    """Emit the E2E contract from report/code/artifacts/trace parts.

    Benchmark facts served: shape is ``{"results": {"report", "code":
    [{"filename", "code"}], "artifacts": [{"filename", "format", "artifact"}],
    "trace"}}``; the scorer truncates each artifact at 10000 chars, so each is
    truncated at 9500 defensively; the E2E scorer uses STRICT ``json.loads``
    on the whole completion, so the emitted string is pure JSON with nothing
    outside the object.
    """
    payload = {
        "results": {
            "report": _s(report),
            "code": [
                {"filename": _s(filename), "code": _s(source)}
                for filename, source in code or []
            ],
            "artifacts": [
                {
                    "filename": _s(filename),
                    "format": _s(fmt),
                    "artifact": _s(artifact)[:E2E_ARTIFACT_CAP],
                }
                for filename, fmt, artifact in artifacts or []
            ],
            "trace": _s(trace),
        }
    }
    return _dumps(payload)


# ---------------------------------------------------------------------------
# Cross-task output guard
# ---------------------------------------------------------------------------

def validate_single_json_object(s: str) -> bool:
    """Return True iff ``s`` is exactly one JSON object with nothing outside.

    Benchmark fact served: benchmark parsers slice the completion from the
    first ``{`` to the last ``}`` (and the E2E scorer runs strict
    ``json.loads`` on the whole completion) — this guard verifies that
    (a) strict ``json.loads(s)`` succeeds and yields a dict, (b) ``s.strip()``
    starts with ``{`` and ends with ``}`` (nothing but whitespace outside the
    object), and (c) the benchmark's first-``{``-to-last-``}`` slice parses to
    the SAME object, so tolerant and strict parsers agree.
    """
    if not isinstance(s, str):
        return False
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    stripped = s.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return False
    start = s.find("{")
    end = s.rfind("}")
    try:
        sliced = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return False
    return sliced == obj
