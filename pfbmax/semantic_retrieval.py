"""PFB-MAX semantic retrieval pool.

Builds the candidate pool for semantic PFB queries: one cheap gpt-4o-mini
expansion call (rephrasings + criteria + HyDE hypothetical abstracts), a
budgeted multi-probe fanout over ``snippet_search`` / ``paper_search``, one
round of citation expansion on the top fused seeds, all fused with weighted
reciprocal-rank fusion (per-channel best-rank dedup, IRIS retrieval-v3
lineage; see ``iris_asta/iris_asta/solvers/pfb.py:semantic_channel``).

Why HyDE views are first-class probes: IRIS forensics showed ~30% of gold
papers were NEVER retrieved by query-derived probes: the recall wall.  A
hypothetical answer-abstract written in the field's own vocabulary is a far
better dense-retrieval key than the question text (Gao et al., 2022), and
gpt-4o-mini writes materially better abstracts than the open-weight backbone
IRIS ran.  Two methodologically diverse views widen the net further.

Integrity: every probe is generated from the live query text only; no gold
corpus ids, no per-query branching.  All stored texts are VERBATIM backend
strings (snippet text or the paper's own abstract); ``evidence()`` only
joins/truncates contiguous spans, never rewrites.

Contract:

    def build_pool(query, criteria, client, llm, inserted_before,
                   max_calls=90) -> SemPool
    SemPool: .ranked() .evidence(cid) .texts(cid) .size() .calls_used

Stdlib + iris_asta imports only.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from typing import Any


def _ensure_iris_importable() -> None:
    """Make the read-only ``iris_asta`` package importable from the bundle.

    The bundle layout is ``<bundle>/pfbmax/`` (this file) next to
    ``<bundle>/iris_asta/iris_asta/`` (the package inside its repo dir).
    Probes for a SUBMODULE, not the bare package name: run from the bundle
    root, the outer ``iris_asta`` repo dir (no ``__init__.py``) resolves as
    an empty namespace package that would satisfy ``import iris_asta`` while
    providing none of its modules.  Any such stale namespace stub is dropped
    from ``sys.modules`` so the real package wins.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("iris_asta.retrieval") is not None:
            return
    except Exception:
        pass
    bundle = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(bundle, "iris_asta")
    if os.path.isdir(os.path.join(candidate, "iris_asta")):
        sys.modules.pop("iris_asta", None)
        sys.path.insert(0, candidate)


_ensure_iris_importable()

from iris_asta.contracts import normalize_corpus_id  # noqa: E402
from iris_asta.retrieval import ReciprocalRankFusion  # noqa: E402

# --------------------------------------------------------------------------
# Tunables (IRIS retrieval-v3 lineage; weights proven on the validation set)
# --------------------------------------------------------------------------

RANK_CONSTANT = 60.0          # RRF k (standard; IRIS _RRF_RANK_CONSTANT)
W_SNIPPET = 1.20              # snippet channel multiplier (x probe weight)
W_PAPER = 1.00                # paper_search relevance channel (x probe weight)
W_HYDE = 1.00                 # hyde views fuse on their own flat channel
W_REFERENCE = 0.30            # ref_mention cited-paper channel (x probe weight)
W_CITE_FWD = 0.80             # citation expansion: papers CITING a seed
W_CITE_BACK = 0.65            # citation expansion: papers a seed CITES

PW_RAW = 1.30                 # probe weights (raw query keeps primacy)
PW_REPHRASE = 1.00
PW_CRITERION = 0.80

LIMIT_SNIPPET_RAW = 100        # raw query keeps the deepest snippet list
LIMIT_SNIPPET_PROBE = 60      # rephrasings / criteria add recall, not depth
LIMIT_SNIPPET_HYDE = 100       # hyde views are the recall lever - deep fanout
LIMIT_PAPER = 100              # relevance-ranker channel per probe
LIMIT_LOCAL = 1000            # local mirror: depth is ~free (one ranked scan)
LIMIT_CITATIONS = 60          # per direction per expansion seed
# The depth limits above were raised 2-2.5x from their first values.
# Rationale, measured: the semantic score is harmonic(rank, recall@K) and
# the decomposition over 8 live queries was rank 0.668 / recall@K 0.083 --
# recall is the whole bottleneck. Corpus calls are negligible under the
# benchmark's cost accounting (only solver LLM tokens count), and a larger
# *limit* returns more results per call rather than issuing more calls, so
# this buys recall at ~zero dollar cost and near-zero wall clock. The
# cross-encoder stage added alongside is what converts the extra pool
# depth into ranked position (IRIS measured +102 pool golds converting to
# only +17 submittable WITHOUT a CE, and named the CE as the missing
# promoter).

MAX_REPHRASINGS = 8
MAX_CRITERIA = 6
MAX_HYDE_VIEWS = 4
MAX_TEXTS_PER_CID = 3         # distinct verbatim snippets kept per paper
EVIDENCE_MAX_CHARS = 1200

EXPANSION_SEED_POOL = 10      # fused window considered for expansion...
EXPANSION_SEEDS = 6           # ...but only this many are expanded (budget)


def _envi(name: str, default: int) -> int:
    """Integer override from the environment, ignoring junk values."""
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


# Pool recall is the CEILING on the semantic score: recall@K can never exceed
# pool_recall/2 (K ~ 2P), so 43% pool recall caps recall@K at ~0.216 no
# matter how good the ranker is. Widening the fanout is therefore the only
# lever that raises the ceiling rather than chasing it -- and it is nearly
# free, because the benchmark's cost accounting counts SOLVER LLM tokens
# only; corpus calls cost wall-clock and nothing else. Measured runs used 38
# of the 90-call budget, so the binding constraint was probe count, not
# budget. Every knob below is env-overridable so a wide config can be A/B'd
# against the default without touching the defaults.
MAX_REPHRASINGS = _envi("PFBMAX_MAX_REPHRASINGS", MAX_REPHRASINGS)
MAX_CRITERIA = _envi("PFBMAX_MAX_CRITERIA", MAX_CRITERIA)
MAX_HYDE_VIEWS = _envi("PFBMAX_MAX_HYDE", MAX_HYDE_VIEWS)
EXPANSION_SEED_POOL = _envi("PFBMAX_SEED_POOL", EXPANSION_SEED_POOL)
EXPANSION_SEEDS = _envi("PFBMAX_SEEDS", EXPANSION_SEEDS)
LIMIT_SNIPPET_RAW = _envi("PFBMAX_LIMIT_SNIPPET_RAW", LIMIT_SNIPPET_RAW)
LIMIT_SNIPPET_PROBE = _envi("PFBMAX_LIMIT_SNIPPET_PROBE", LIMIT_SNIPPET_PROBE)
LIMIT_SNIPPET_HYDE = _envi("PFBMAX_LIMIT_SNIPPET_HYDE", LIMIT_SNIPPET_HYDE)
LIMIT_PAPER = _envi("PFBMAX_LIMIT_PAPER", LIMIT_PAPER)
LIMIT_CITATIONS = _envi("PFBMAX_LIMIT_CITATIONS", LIMIT_CITATIONS)

#: The dense snippet backend rejects markdown/query-operator characters -
#: exact character class from IRIS ``_HYDE_SANITIZE_RE`` (live-verified).
_SANITIZE_RE = re.compile(r'[*"`~^\\|{}\[\]<>#]+')

_CITATION_FIELDS = "corpusId,title,abstract,year"

# --------------------------------------------------------------------------
# One-call query expansion (rephrasings + criteria + HyDE views)
# --------------------------------------------------------------------------

_EXPAND_SYSTEM = """You expand a scientific paper-finding query for high-recall literature retrieval.
Respond with JSON only, using exactly these keys:
"rephrasings": exactly {n_rephrasings} diverse rephrasings of the query. Each must use ALTERNATIVE technical vocabulary, synonyms, and phrasings the full text of a relevant paper might use, while preserving the exact information need. Make them genuinely different from each other.
{criteria_clause}"hyde_views": exactly {n_hyde} hypothetical paper ABSTRACTS of 60-120 words each, plain prose only (no title, no markdown, no bullet points, no quotes). Each abstract describes a DIFFERENT plausible research paper that would perfectly satisfy the query, written in the field's own standard terminology, naming the concrete methods, datasets, tasks, metrics, and findings such a paper would contain. Make the views methodologically diverse (e.g. different approach families or settings)."""

_CRITERIA_CLAUSE = """"criteria": between 1 and 4 SHORT relevance criteria — the independent requirements a paper must EACH satisfy to be perfectly relevant to the query (topic, method, finding, setting, population). Each criterion is a single, independently checkable statement.
"""


def _clean_line(text: Any) -> str:
    """Whitespace-collapsed single-line string (probe hygiene)."""
    return " ".join(str(text or "").split())


def _sanitize_probe(text: Any) -> str:
    """Strip the characters the dense backend rejects; collapse whitespace."""
    return _clean_line(_SANITIZE_RE.sub(" ", str(text or "")))


def _coerce_criteria(criteria: Any) -> list[str]:
    """Normalize caller criteria to clean strings.

    Callers normally pass ``list[str]`` (contract), but validation tooling
    carries criteria as ``{name, description, weight}`` dicts; accept those
    too (description preferred) so measurement scripts can't silently lose
    the criteria signal.
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in criteria or []:
        if isinstance(item, dict):
            text = _clean_line(item.get("description") or item.get("name"))
        else:
            text = _clean_line(item)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _expand_query(
    llm: Any, query: str, passed_criteria: list[str]
) -> dict[str, list[str]]:
    """One LLM json call -> {rephrasings, criteria, hyde_views} (all lists).

    When the caller already supplied criteria the prompt does not ask for a
    criteria key (and any returned one is ignored).  Every failure mode
    (None reply, wrong shapes, junk items) degrades to empty lists: the
    fanout then runs on the raw query alone, never empty (IRIS
    no-empty-fallback lesson).
    """
    criteria_clause = "" if passed_criteria else _CRITERIA_CLAUSE
    # Ask for as many probes as the caps allow: the prompt used to hard-code
    # "exactly 8"/"exactly 4", so raising MAX_REPHRASINGS / MAX_HYDE_VIEWS
    # silently changed nothing. HyDE is the deepest-fanout channel we have,
    # and pool recall is the ceiling on the whole semantic score.
    system = _EXPAND_SYSTEM.format(criteria_clause=criteria_clause,
                                   n_rephrasings=MAX_REPHRASINGS,
                                   n_hyde=MAX_HYDE_VIEWS)
    user = f"QUERY:\n{query}"
    if passed_criteria:
        joined = "\n".join(f"- {c}" for c in passed_criteria[:8])
        user += (
            "\n\nKnown relevance criteria (do NOT output a criteria key; "
            f"use these to sharpen the abstracts):\n{joined}"
        )
    try:
        # A gpt-4o "reason first" expansion was tried here and REGRESSED
        # (-0.034 vs baseline, 3/3 down): richer expansions drift from the
        # query's actual intent and dilute the pool. Recall wants FAITHFUL
        # paraphrase, not creative reframing.
        # Scale the cap with what the prompt actually asks for: a fixed 900
        # truncates the JSON (and loses the whole expansion) once the
        # rephrasing/HyDE counts are raised above the defaults.
        # 900 is enough for the default caps (measured headroom), so the
        # default stays exactly as scored; the budget only grows when a
        # caller raises the caps, where a fixed 900 truncated the JSON and
        # silently lost the whole expansion.
        budget = max(900, 200 + 60 * max(MAX_REPHRASINGS - 8, 0)
                     + 260 * max(MAX_HYDE_VIEWS - 4, 0)
                     + (0 if passed_criteria else 60 * max(MAX_CRITERIA - 6, 0)))
        if MAX_REPHRASINGS > 8 or MAX_HYDE_VIEWS > 4 or MAX_CRITERIA > 6:
            budget = max(budget, 900 + 60 * MAX_REPHRASINGS
                         + 260 * MAX_HYDE_VIEWS)
        obj = llm.json(system, user, max_tokens=budget)
    except Exception:
        obj = None
    if not isinstance(obj, dict):
        obj = {}

    def _str_list(key: str, cap: int) -> list[str]:
        raw = obj.get(key)
        out: list[str] = []
        seen = {query.strip().casefold()}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, str):
                    continue
                text = _clean_line(item)
                if not text or text.casefold() in seen:
                    continue
                seen.add(text.casefold())
                out.append(text)
                if len(out) >= cap:
                    break
        return out

    hyde_views: list[str] = []
    raw_views = obj.get("hyde_views")
    if isinstance(raw_views, list):
        for item in raw_views:
            if not isinstance(item, str):
                continue
            text = _sanitize_probe(item)[:1200]
            # Under ~40 chars it is not an abstract - a dense probe that
            # short only adds noise (IRIS threshold).
            if len(text) >= 40:
                hyde_views.append(text)
            if len(hyde_views) >= MAX_HYDE_VIEWS:
                break
    return {
        "rephrasings": _str_list("rephrasings", MAX_REPHRASINGS),
        "criteria": [] if passed_criteria else _str_list("criteria", MAX_CRITERIA),
        "hyde_views": hyde_views,
    }


# --------------------------------------------------------------------------
# Budgeted client wrappers
# --------------------------------------------------------------------------


class _Budget:
    """Hard cap on corpus-backend calls.  Every ATTEMPT costs one unit
    (a call that raises still consumed backend time and rate limit)."""

    def __init__(self, max_calls: int):
        self.max_calls = max(0, int(max_calls))
        self.used = 0
        # Retries are taken from worker threads, so the counter needs a lock.
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.used >= self.max_calls:
                return False
            self.used += 1
            return True

    def reserve(self, n: int) -> int:
        """Claim up to n units at once; returns how many were granted."""
        with self._lock:
            n = max(0, min(int(n), self.max_calls - self.used))
            self.used += n
            return n


def _fanout(fetch, jobs, budget, workers=None):
    """Run independent budgeted probes concurrently, in deterministic order.

    Each corpus call is ~6s of pure server latency, so issuing them serially
    made a wide fanout cost ~23 minutes a query -- enough to make the
    highest-recall configuration unusable in practice even though the calls
    themselves are negligible under the benchmark's cost accounting.

    Budget is reserved UP FRONT on the calling thread, so the cap stays exact
    (``_Budget.take`` is a non-atomic read-modify-write and would race).
    Results come back in job order, and absorption stays sequential in the
    caller, so fusion ranks and the per-paper snippet selection are bit-identical
    to the serial path -- concurrency changes the wall clock, not the pool.

    Returns a list aligned with ``jobs``; entries past the budget are None,
    which callers already treat as "budget exhausted".
    """
    jobs = list(jobs)
    allowed = budget.reserve(len(jobs))
    run = jobs[:allowed]
    if not run:
        return [None] * len(jobs)
    if workers is None:
        workers = _envi("PFBMAX_FANOUT_WORKERS", 8)
    workers = max(1, min(int(workers), len(run)))
    if workers == 1:
        out = [fetch(j) for j in run]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(fetch, run))
    return out + [None] * (len(jobs) - len(run))


def _snippet_search(client, budget, query, limit, inserted_before):
    """Budgeted snippet_search; None = budget exhausted, [] = empty/failed."""
    if not budget.take():
        return None
    return _snippet_fetch(client, query, limit, inserted_before)


# One retry: enough to ride out a throttle, bounded so a dead probe cannot
# starve the others. Both are env-tunable (0 retries = original behaviour).
FETCH_RETRIES = _envi("PFBMAX_FETCH_RETRIES", 1)
FETCH_BACKOFF_S = float(os.environ.get("PFBMAX_FETCH_BACKOFF") or 1.0)
_FETCH_FAILURES = {"snippet": 0, "paper": 0, "retried": 0}


def _retrying(kind, call, budget=None):
    """Run a corpus call, retrying transient failures instead of losing them.

    A swallowed exception is indistinguishable from "no matches", and that
    silence is expensive: concurrent probes get throttled, each failure
    returns [], and the pool quietly shrinks. Measured directly -- the same
    37 calls built a 2112-paper pool serially but only 1960 at 6 workers
    (-7%), purely from dropped responses.

    Each retry COSTS A BUDGET UNIT, because the budget exists to bound load
    on a shared rate-limited backend and a raised call already consumed
    backend time. So retries recover lost results only while budget remains,
    and never smuggle extra load past the cap. With no budget passed there is
    no retry at all -- the original single-attempt behaviour.
    """
    try:
        return call() or []
    except Exception:
        pass
    # Bounded: a probe that fails deterministically must not be able to drain
    # the budget that the remaining probes need. One retry is what recovers a
    # throttled call; a second failure means the probe is simply dead.
    for _ in range(FETCH_RETRIES):
        if budget is None or not budget.take():
            break
        _FETCH_FAILURES["retried"] += 1
        time.sleep(FETCH_BACKOFF_S)
        try:
            return call() or []
        except Exception:
            continue
    _FETCH_FAILURES[kind] = _FETCH_FAILURES.get(kind, 0) + 1
    return []


def _snippet_fetch(client, query, limit, inserted_before, budget=None):
    """Unbudgeted first attempt (budget is reserved by _fanout); retries,
    when a budget is supplied, are charged to it."""
    return _retrying("snippet", lambda: client.snippet_search(
        query, limit=limit, inserted_before=inserted_before), budget)


def _paper_search(client, budget, query, limit, inserted_before):
    if not budget.take():
        return None
    return _paper_fetch(client, query, limit, inserted_before)


def _paper_fetch(client, query, limit, inserted_before, budget=None):
    """Unbudgeted first attempt; retries are charged to ``budget``."""
    return _retrying("paper", lambda: client.paper_search(
        query, limit=limit, publication_date_before=inserted_before), budget)


def _get_citations(client, budget, cid, direction, limit, date_range):
    if not budget.take():
        return None
    try:
        return client.get_citations(
            cid,
            direction=direction,
            limit=limit,
            publication_date_range=date_range,
            fields=_CITATION_FIELDS,
        ) or []
    except Exception:
        return []


# --------------------------------------------------------------------------
# Snippet / paper helpers (ports of small pfb.py primitives)
# --------------------------------------------------------------------------


def _snippet_score(snippet: Any) -> float:
    """Positive contribution score of a snippet (unscored hits count 1.0)."""
    try:
        score = float(getattr(snippet, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return score if score > 0.0 else 1.0


def _ref_mention_cids(snippet: Any) -> list[str]:
    """Digits-only corpus ids referenced by a snippet's ref_mentions."""
    out: list[str] = []
    for mention in getattr(snippet, "ref_mentions", None) or []:
        if isinstance(mention, dict):
            raw = (
                mention.get("matchedPaperCorpusId")
                or mention.get("corpusId")
                or mention.get("corpus_id")
            )
        else:
            raw = mention
        cid = normalize_corpus_id(raw)
        if cid and cid not in out:
            out.append(cid)
    return out


def _cid_of(paper: Any) -> str | None:
    return normalize_corpus_id(getattr(paper, "corpus_id", None))


def _postdates_snapshot(year: Any, cutoff_year: int | None) -> bool:
    """True only when the year PROVES the paper postdates the snapshot.

    Same-year and unknown-year papers are kept (the snapshot may cut
    mid-year; dropping on suspicion costs recall, matching IRIS
    ``_violates_cutoff`` semantics).  Post-snapshot ids are INVALID to the scorer, worse than
    irrelevant.
    """
    return (
        cutoff_year is not None
        and isinstance(year, int)
        and year > cutoff_year
    )


# --------------------------------------------------------------------------
# SemPool
# --------------------------------------------------------------------------


class SemPool:
    """Fused candidate pool for one semantic query.

    Contract surface: ``ranked() / evidence(cid) / texts(cid) / size() /
    calls_used``.  Extras used by the judge/router modules:
    ``criteria`` (caller-passed or generated), ``rephrasings``,
    ``hyde_views``, ``title(cid)``, ``year(cid)``, ``families(cid)``,
    ``rrf_score(cid)``, ``diagnostics``.
    """

    def __init__(
        self,
        query: str,
        criteria: list[str],
        rephrasings: list[str],
        hyde_views: list[str],
        records: dict[str, dict],
        fusion: ReciprocalRankFusion,
        calls_used: int,
        diagnostics: dict[str, Any],
    ):
        self.query = query
        self.criteria = list(criteria)
        self.rephrasings = list(rephrasings)
        self.hyde_views = list(hyde_views)
        self._records = records
        self._fusion = fusion
        self.calls_used = int(calls_used)
        self.diagnostics = diagnostics
        self._ranked_cache: list[str] | None = None

    # -- contract ----------------------------------------------------------

    def ranked(self) -> list[str]:
        """All pool cids in fused order (best first, deterministic)."""
        if self._ranked_cache is None:
            self._ranked_cache = [
                c.key for c in self._fusion.ranked() if c.key in self._records
            ]
        return list(self._ranked_cache)

    def evidence(self, cid: str) -> str:
        """Best verbatim snippet(s) for ``cid``, newline-joined, <=1200 chars.

        Parts are stored backend texts (snippets best-score-first, else the
        paper's own abstract).  Whole parts are appended while they fit; the
        first part alone is truncated to the cap if oversized; truncation
        keeps a contiguous prefix, so every emitted part remains a verbatim
        substring of a stored text.
        """
        rec = self._records.get(_key(cid))
        if not rec:
            return ""
        parts = [text for _score, text in rec["snippets"]]
        if not parts and rec.get("abstract"):
            parts = [rec["abstract"]]
        out: list[str] = []
        used = 0
        for part in parts:
            sep = 1 if out else 0
            if not out and len(part) > EVIDENCE_MAX_CHARS:
                out.append(part[:EVIDENCE_MAX_CHARS])
                used = EVIDENCE_MAX_CHARS
                break
            if used + sep + len(part) > EVIDENCE_MAX_CHARS:
                continue
            out.append(part)
            used += sep + len(part)
        return "\n".join(out)

    def texts(self, cid: str) -> list[str]:
        """All stored verbatim texts for ``cid`` (snippets first, then the
        abstract when distinct).  Empty list for graph-only candidates."""
        rec = self._records.get(_key(cid))
        if not rec:
            return []
        out = [text for _score, text in rec["snippets"]]
        abstract = rec.get("abstract") or ""
        if abstract and abstract not in out:
            out.append(abstract)
        return out

    def size(self) -> int:
        return len(self._records)

    # -- extras ------------------------------------------------------------

    def title(self, cid: str) -> str:
        rec = self._records.get(_key(cid))
        return (rec or {}).get("title", "")

    def year(self, cid: str):
        rec = self._records.get(_key(cid))
        return (rec or {}).get("year")

    def best_snippet_score(self, cid: str) -> float:
        rec = self._records.get(_key(cid))
        return (rec or {}).get("best_score", 0.0)

    def families(self, cid: str) -> list[str]:
        cand = self._fusion.get(_key(cid))
        return sorted(cand.families) if cand else []

    def channels(self, cid: str) -> dict[str, int]:
        cand = self._fusion.get(_key(cid))
        return dict(cand.channels) if cand else {}

    def rrf_score(self, cid: str) -> float:
        cand = self._fusion.get(_key(cid))
        return cand.score if cand else 0.0


def _key(cid: Any) -> str:
    return normalize_corpus_id(cid) or str(cid or "").strip()


# --------------------------------------------------------------------------
# build_pool
# --------------------------------------------------------------------------


def build_pool(
    query: str,
    criteria: list[str],
    client: Any,
    llm: Any,
    inserted_before: str | None,
    max_calls: int = -1,
) -> SemPool:
    """Build the fused semantic candidate pool for ``query``.

    Stages (all corpus calls budgeted by ``max_calls``, snapshot-bounded by
    ``inserted_before``):

    1. ONE gpt-4o-mini json call -> rephrasings (up to MAX_REPHRASINGS),
       criteria (up to MAX_CRITERIA; skipped when the caller passed
       criteria), and sanitized HyDE abstracts (up to MAX_HYDE_VIEWS).
    2. Fanout: snippet_search on the raw query, hyde views (own flat
       channel), rephrasings, and criteria at the LIMIT_SNIPPET_* depths;
       paper_search on raw + rephrasings (LIMIT_PAPER).  Snippet
       ref_mentions credit cited papers on a low-weight reference channel
       (zero extra calls).  Weighted RRF with per-channel best-rank dedup
       fuses everything.
    3. One citation-expansion round: within the top EXPANSION_SEED_POOL
       fused, the top EXPANSION_SEEDS seeds get ``get_citations`` both
       directions (LIMIT_CITATIONS each, post-snapshot years skipped),
       absorbed at W_CITE_FWD (citing) / W_CITE_BACK (cited), budget-capped
       like every other call.

    Fanout priority under a tight budget: raw first (primary signal), then
    hyde views (the recall lever), then rephrasings, criteria, paper
    channels, citations.
    """
    query = _clean_line(query)
    # -1 means "take the default", so PFBMAX_MAX_CALLS can raise the ceiling
    # without every caller having to thread the argument through.
    if max_calls is None or max_calls < 0:
        max_calls = _envi("PFBMAX_MAX_CALLS", 90)
    budget = _Budget(max_calls)
    fusion = ReciprocalRankFusion(rank_constant=RANK_CONSTANT)
    records: dict[str, dict] = {}

    def _record(cid: str) -> dict:
        return records.setdefault(
            cid,
            {
                "snippets": [],       # [(score, text)] distinct, best-first
                "best_score": 0.0,
                "title": "",
                "abstract": "",
                "year": None,
            },
        )

    def _absorb_snippet(cid: str, title: str, text: str, score: float) -> None:
        rec = _record(cid)
        if text:
            if score > rec["best_score"]:
                rec["best_score"] = score
            if all(text != t for _s, t in rec["snippets"]):
                rec["snippets"].append((score, text))
                rec["snippets"].sort(key=lambda pair: -pair[0])
                del rec["snippets"][MAX_TEXTS_PER_CID:]
        if title and not rec["title"]:
            rec["title"] = title

    def _absorb_paper(cid: str, paper: Any) -> None:
        rec = _record(cid)
        title = _clean_line(getattr(paper, "title", "") or "")
        abstract = (getattr(paper, "abstract", "") or "").strip()
        year = getattr(paper, "year", None)
        if title and not rec["title"]:
            rec["title"] = title
        if abstract and not rec["abstract"]:
            rec["abstract"] = abstract
        if isinstance(year, int) and rec["year"] is None:
            rec["year"] = year

    # -- stage 1: one expansion call --------------------------------------
    passed_criteria = _coerce_criteria(criteria)
    plan = _expand_query(llm, query, passed_criteria)
    final_criteria = passed_criteria or plan["criteria"] or [query]
    rephrasings = plan["rephrasings"]
    hyde_views = plan["hyde_views"]

    # -- stage 2: fanout ---------------------------------------------------
    # Snippet jobs: (probe_text, channel, fused_weight, limit).  Probe texts
    # are casefold-deduped so a criterion equal to the raw query (the
    # no-criteria fallback) never pays for the same search twice.
    seen_probes: set[str] = set()
    snippet_jobs: list[tuple[str, str, float, int]] = []

    def _add_snippet_job(text, channel, weight, limit):
        clean = _sanitize_probe(text)
        key = clean.casefold()
        if clean and key not in seen_probes:
            seen_probes.add(key)
            snippet_jobs.append((clean, channel, weight, limit))

    _add_snippet_job(query, "snippet:raw-0", W_SNIPPET * PW_RAW,
                     LIMIT_SNIPPET_RAW)
    for i, view in enumerate(hyde_views[:MAX_HYDE_VIEWS]):
        _add_snippet_job(view, f"hyde:view-{i}", W_HYDE, LIMIT_SNIPPET_HYDE)
    for i, text in enumerate(rephrasings[:MAX_REPHRASINGS]):
        _add_snippet_job(text, f"snippet:rephrase-{i}",
                         W_SNIPPET * PW_REPHRASE, LIMIT_SNIPPET_PROBE)
    for i, text in enumerate(final_criteria[:MAX_CRITERIA]):
        _add_snippet_job(text, f"snippet:criterion-{i}",
                         W_SNIPPET * PW_CRITERION, LIMIT_SNIPPET_PROBE)

    _local_stats: dict = {}
    # --- local corpus mirror (opt-in supplement) --------------------------
    # An optional local index. One wide BM25 query measured 37.3% pool
    # recall on the gold sets vs 43.2% for IRIS's ~168-remote-call fanout,
    # it never rate-limits, and it adds no solver LLM tokens to the cost.
    #
    # When enabled it SUPPLEMENTS the remote fanout with a few probes; the
    # remote fanout below still runs either way.
    # OFF by default: an earlier configuration that made the mirror primary
    # measured a REGRESSION (0.115 vs 0.152 on the same 3 queries). Cause is
    # latency, not relevance -- 9 concurrent deep probes overwhelmed the
    # mirror and 7 of 9 returned nothing, starving the pool (n_submitted
    # fell 250 -> ~89). One sequential probe reaches 37.3% recall in 24s,
    # so the mirror is sound; it needs a real search engine (Tantivy/Lucene)
    # or a field-filtered smaller index to serve a wide fanout.
    # Opt in with PFBMAX_USE_LOCAL=1.
    _local_on = False
    _ls = None
    if os.environ.get("PFBMAX_USE_LOCAL", "").strip() in ("1", "true", "yes", "on"):
        try:
            import local_search as _ls
            _local_on = _ls.available(timeout=6.0)
        except Exception:
            _ls = None
    if _local_on:
        # SUPPLEMENT, not replacement, and deliberately few: each local probe
        # is ~24s of memory-bound BM25 and 9 concurrent ones starved the pool
        # (7 of 9 timed out). Three probes add an independent retrieval arm --
        # the mirror alone reaches 37.3% pool recall vs the remote fanout's
        # 43.2%, and their union is what breaks the ~0.25 semantic ceiling
        # that pool recall imposes.
        local_probes = [(query, "local:raw", W_SNIPPET * PW_RAW)]
        for i, view in enumerate(hyde_views[:2]):
            local_probes.append((view, f"local:hyde-{i}", W_HYDE))
        # Probes are independent, so fire them concurrently: sequentially
        # this cost ~24s x 9 probes and blew a 10-minute budget on ONE query.
        # The mirror is a threaded server with a 48-connection pool on a
        # 64-core box, so concurrency is free on both ends.
        from concurrent.futures import ThreadPoolExecutor
        jobs = [(t, c, w) for (t, c, w) in local_probes if (t or "").strip()]
        with ThreadPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(
                lambda j: (j[1], j[2], _ls.search(j[0], LIMIT_LOCAL)), jobs))
        n_local = 0
        for channel, weight, papers in results:
            for rank, paper in enumerate(papers):
                cid = normalize_corpus_id(getattr(paper, "corpusId", None))
                if not cid:
                    continue
                _absorb_paper(cid, paper)
                fusion.add(cid, rank=rank, channel=channel, weight=weight)
                n_local += 1
        _local_stats = {"hits": n_local, "probes": len(jobs)}

    snippet_results = _fanout(
        lambda j: _snippet_fetch(client, j[0], j[3], inserted_before, budget),
        snippet_jobs, budget)
    for (probe_text, channel, weight, limit), snippets in zip(snippet_jobs,
                                                              snippet_results):
        if snippets is None:
            break
        snippets = sorted(snippets, key=_snippet_score, reverse=True)
        family = channel.split(":", 1)[0]
        # Probe weight recovered for the reference channel: fused weight is
        # W_SNIPPET*pw (or W_HYDE), reference rides at W_REFERENCE*pw.
        probe_weight = (weight / W_SNIPPET) if family == "snippet" else 1.0
        ref_channel = "reference:" + channel.split(":", 1)[1]
        for rank, snippet in enumerate(snippets):
            score = _snippet_score(snippet)
            text = getattr(snippet, "text", "") or ""
            title = _clean_line(getattr(snippet, "title", "") or "")
            cid = normalize_corpus_id(getattr(snippet, "corpus_id", None))
            if cid:
                _absorb_snippet(cid, title, text, score)
                fusion.add(cid, rank=rank, channel=channel, weight=weight)
            # Cited-paper attachment (free candidates): the citing sentence
            # is the OTHER paper's text, so the cited paper gets rank credit
            # only - its own snippets/abstract must furnish evidence later.
            for ref_cid in _ref_mention_cids(snippet):
                if ref_cid != cid:
                    _record(ref_cid)
                    fusion.add(ref_cid, rank=rank, channel=ref_channel,
                               weight=W_REFERENCE * probe_weight)

    paper_jobs: list[tuple[str, str, float]] = [
        (query, "paper:raw-0", W_PAPER * PW_RAW)
    ]
    for i, text in enumerate(rephrasings[:MAX_REPHRASINGS]):
        paper_jobs.append((_sanitize_probe(text), f"paper:rephrase-{i}",
                           W_PAPER * PW_REPHRASE))
    paper_results = _fanout(
        lambda j: _paper_fetch(client, j[0], LIMIT_PAPER, inserted_before, budget),
        paper_jobs, budget)
    for (probe_text, channel, weight), papers in zip(paper_jobs, paper_results):
        if papers is None:
            break
        rank = 0
        for paper in papers:
            cid = _cid_of(paper)
            abstract = (getattr(paper, "abstract", "") or "").strip()
            # Abstract required: this channel's hits carry no snippet, so a
            # hit without its own text could never earn judged evidence.
            if not cid or not abstract:
                continue
            _absorb_paper(cid, paper)
            fusion.add(cid, rank=rank, channel=channel, weight=weight)
            rank += 1

    # -- stage 3: one citation-expansion round ----------------------------
    cutoff_year: int | None = None
    if inserted_before:
        try:
            cutoff_year = int(str(inserted_before)[:4])
        except (TypeError, ValueError):
            cutoff_year = None
    date_range = (
        f"1900-01-01:{str(inserted_before)[:10]}" if inserted_before else None
    )
    seed_window = [c.key for c in fusion.ranked()[:EXPANSION_SEED_POOL]]
    # Citation expansion is remote-only and each call can stall 60-90s
    # when the corpus backend is degraded, so it runs last and stays
    # budget-capped like every other call.
    seeds = seed_window[:EXPANSION_SEEDS]
    exhausted = False
    for si, seed in enumerate(seeds):
        if exhausted:
            break
        for direction, weight, prefix in (
            ("citations", W_CITE_FWD, "cite:fwd"),
            ("references", W_CITE_BACK, "cite:back"),
        ):
            # Server-side date shard only for the citations direction (the
            # backend's citer list is recency-first and would return mostly
            # post-snapshot rows otherwise); the references client path
            # would DROP null-date papers on a range, so filter by year.
            papers = _get_citations(
                client, budget, seed, direction, LIMIT_CITATIONS,
                date_range if direction == "citations" else None,
            )
            if papers is None:
                exhausted = True
                break
            rank = 0
            for paper in papers:
                cid = _cid_of(paper)
                if not cid or cid == seed:
                    continue
                if _postdates_snapshot(getattr(paper, "year", None),
                                       cutoff_year):
                    continue
                _absorb_paper(cid, paper)
                fusion.add(cid, rank=rank, channel=f"{prefix}-{si}",
                           weight=weight)
                rank += 1

    diagnostics = {
        **({'local': _local_stats} if _local_stats else {}),
        "n_probes": len(snippet_jobs),
        "n_paper_probes": len(paper_jobs),
        "n_hyde_views": len(hyde_views),
        "expansion_window": seed_window,
        "expansion_seeds": seeds,
        "budget": budget.max_calls,
    }
    return SemPool(
        query=query,
        criteria=final_criteria,
        rephrasings=rephrasings,
        hyde_views=hyde_views,
        records=records,
        fusion=fusion,
        calls_used=budget.used,
        diagnostics=diagnostics,
    )


__all__ = ["SemPool", "build_pool"]
