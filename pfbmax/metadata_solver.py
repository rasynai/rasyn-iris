"""Deterministic metadata-constraint solver.

``solve_metadata(query, client, llm, inserted_before) -> Submission``. The
metadata slice of PFB ("NAACL 2010 or 2012 papers co-authored by one of the
authors of the BERT paper", "Papers citing the DistilBERT paper after 2022
with more than 50 citations") is scored by exact-set F1 over corpus ids, so
the winning shape is: ONE gpt-4o-mini call that TRANSCRIBES the query's
stated constraints into a typed plan, then pure deterministic set algebra
over the corpus APIs. No LLM ever touches the result set.

Design (iris_asta is read-only and imported):

* Planner: gpt-4o-mini via the shared ``pfbmax.llm.LLM`` backbone (an inline
  stdlib OpenAI fallback keeps this module runnable if that import is
  unavailable). The plan schema extends IRIS's 16-field vocabulary with
  ``authors_any``, ``max_citations`` and ``journal_only``. Every field is
  deterministically cleaned (types coerced, years clamped to [1000, 2100],
  inverted bounds swapped, IRIS's hedged-endpoints repair, publication-type
  whitelist).
* Query-regex repairs: the constraint COUNTS and YEAR bounds are re-derived
  from the query text itself wherever a recognizable phrase exists and
  OVERRIDE the LLM's numbers: "more than 50 citations" -> 51, "at least one
  additional author" -> 2, "after/since 2022" -> year_min 2022 (INCLUSIVE:
  IRIS forensics; gold programs treat "after Y"/"Y and beyond" as >= Y),
  "before 2018" -> year_max 2017, "2022-2023" -> bounds, "2014 or 2017" ->
  exact membership. Deterministic beats stochastic on numbers the executor
  will apply exactly.
* Relation-slot repair: ``authors_of_paper`` survives only when the query
  literally says "authors of X"; otherwise the anchor title moves to
  ``must_cite`` under citing-language, else is dropped. Measured behavior
  (2026-08-10): gpt-4o-mini filed the anchor of "Papers citing the
  DistilBERT paper ..." under ``authors_of_paper``, silently turning a
  citer query into an author-pool query. That is the single most expensive
  plan error observed, and it is query-text checkable.
* Executor: IRIS's proven ``execute_plan`` (iris_asta/solvers/pfb.py),
  imported; it carries the machinery this slice lives on: anchor
  title->relevance resolution, author disambiguation (exact-name +
  paperCount, never candidates[0]), authors_of_paper anchoring (resolve the
  paper -> union its authors' papers), citer sets with date-window bisection
  past the 1000-row cap, venue alias table (S2 long forms), no-empty-fallback
  (submit the last nonempty stage, never []), snapshot-safe citation-count
  ranking, 250 cap.
* Author DE-FRAGMENTATION (measured on development queries): the corpus
  author index shards one person across many records (a full-name form
  can hold a dozen fragments of a few papers each while the initialism
  form holds hundreds), so any single-record resolution loses most of
  the author's papers, which zeroes author-constrained queries outright.
  ``_AuthorUnionClient``
  wraps the corpus client and serves a synthetic merged record (union of
  the top name-compatible records' pools) that IRIS's own disambiguation
  rule then deterministically selects. General, name-driven, cannot merge
  a different full first name or surname.
* Post-filters owned here: ``max_citations`` (IRIS's plan cannot express
  it) via one batched citation-count fetch, inclusive-on-error; and an
  ``exclude_author`` membership pass (drop results that appear in the
  excluded author's de-fragmented paper set, catching self-citations whose
  author list renders the name as an initialism that name-equality
  misses).
* Evidence: one ``get_paper_batch`` for the top ``EVIDENCE_TOP`` ids;
  evidence line ``«Title» (year): abstract[:300]`` (verbatim corpus text;
  unscored on this slice, so cheap is correct). Ids beyond the top batch
  carry empty evidence.
* Precision rule: emit ONLY the believed set; the executor never pads, and
  an unexecutable plan returns [] so the router's semantic channel can take
  over.

Approximation, documented: ``authors_any`` maps onto IRIS's ``authors``
field, whose executor intersects author pools but falls back to their UNION
when the intersection is empty: exact OR semantics whenever the listed
authors are distinct people (the realistic case), a precision-safe subset
otherwise.

Integrity: solver logic keys on query text + general rules only. No gold
corpus ids, no per-query branching. Stdlib + iris_asta imports only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from typing import Any

# -- import bootstrap (repo root for pfbmax.*, iris_asta/ for the package) --
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IRIS_DIR = os.path.join(_REPO_ROOT, "iris_asta")
for _p in (_REPO_ROOT, _IRIS_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from iris_asta.contracts import normalize_corpus_id  # noqa: E402
from iris_asta.solvers.pfb import MetadataPlan, execute_plan  # noqa: E402

#: (corpus_id, markdown_evidence) rows, most-relevant first.
Submission = list

#: How many top ids get hydrated (title/year/abstract) for evidence lines.
EVIDENCE_TOP = 25
_EVIDENCE_ABSTRACT_CHARS = 300
#: get_paper_batch chunk for the max_citations post-filter.
_BATCH_CHUNK = 100
_PLAN_MAX_TOKENS = 700
_YEAR_LO, _YEAR_HI = 1000, 2100

# ---------------------------------------------------------------------------
# Planner prompt (gpt-4o-mini). Transcription only: the LLM copies the
# query's EXPLICIT constraints into typed fields; all execution is code.
# Worked examples are invented entities exercising the schema, not benchmark
# queries.
# ---------------------------------------------------------------------------

PLAN_SYSTEM = """You convert ONE scientific paper-finding query into a strict JSON metadata plan.
Return JSON only, with EXACTLY these keys (use null / [] / false when a field is absent):
{"known_title": str|null,
 "venues": [str],
 "years": [int],
 "year_min": int|null,
 "year_max": int|null,
 "authors_all": [str],
 "authors_any": [str],
 "authors_of_paper": str|null,
 "min_authors": int|null,
 "must_cite_titles": [str],
 "must_not_cite_titles": [str],
 "cites_author": str|null,
 "exclude_author": str|null,
 "cites_venue": str|null,
 "publication_types": [str],
 "journal_only": bool,
 "min_citations": int|null,
 "max_citations": int|null,
 "topic_terms": str|null}
Field meanings:
- known_title: ONLY when the query asks for one specific known paper as the answer itself.
- venues: venues the RESULT papers were published in (a list means OR). Use the short venue name exactly as the query gives it ("NAACL", "ACL", "NeurIPS", "SPLASH", "Nature portfolio") with no years attached.
- years: EXPLICIT year disjunction ("published at 2014 or 2017" -> [2014, 2017]); leave year_min/year_max null when years is used.
- year_min/year_max: INCLUSIVE bounds. "in 2020" -> both 2020. "since 2019" / "after 2019" / "2019 and beyond" -> year_min 2019 (INCLUSIVE, never 2020). "before 2018" -> year_max 2017. "2022-2023" / "from 2015 to 2020" / "between 2015 and 2020" -> year_min/year_max (years stays []).
- authors_all: full names that must ALL be authors of each result paper.
- authors_any: full names where ANY ONE of them qualifies a paper.
- authors_of_paper: title of a paper whose AUTHORS define the wanted author pool — ONLY when the query literally asks for papers written/co-authored BY THE AUTHORS OF that paper ("co-authored by one of the authors of the BERT paper" -> the BERT paper's canonical title). NEVER use it for "papers citing X" — that is must_cite_titles. Do not also copy the title into must_cite_titles.
- min_authors: minimum author count ("more than 3 authors" -> 4; "and at least one additional author" -> 2).
- must_cite_titles: titles of anchor papers each RESULT must cite ("papers citing X and Y" -> both; AND across the list).
- must_not_cite_titles: anchor papers the results must NOT cite.
- cites_author: results must cite at least one paper BY this author ("citing papers by Y" -> Y).
- exclude_author: results must NOT be co-authored by this author ("citing papers by Y, but not self-citations of Y" -> cites_author Y AND exclude_author Y).
- cites_venue: results must cite at least one paper from this venue ("cites any NeurIPS paper" -> "NeurIPS").
- publication_types: only from ["JournalArticle", "Conference", "Review", "Book", "Dataset"].
- journal_only: true when the query restricts results to journal articles / journal papers.
- min_citations: minimum citation count of each result. "more than 50 citations" -> 51; "at least 30 citations" -> 30; "over 100 citations" -> 101; "cited by at least 30 other papers" -> 30.
- max_citations: maximum citation count ("fewer than 20 citations" -> 19; "at most 40 citations" -> 40).
- topic_terms: a short keyword phrase ONLY when the query has a topical/subject component; never restate structural constraints here.
Nicknames: when the query names a paper by NICKNAME or acronym ("the BERT paper", "the T5 paper", "DistilBERT"), write that paper's FULL canonical published title instead (e.g. T5 -> "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"; Spider -> "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task"; DistilBERT -> "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"). Copy explicitly quoted titles and author names verbatim. Do NOT invent constraints the query does not state.
Example query: "ICLR 2019 or 2021 papers co-authored by one of the authors of the 'GPT-2' paper and at least one additional author"
Example output: {"known_title": null, "venues": ["ICLR"], "years": [2019, 2021], "year_min": null, "year_max": null, "authors_all": [], "authors_any": [], "authors_of_paper": "Language Models are Unsupervised Multitask Learners", "min_authors": 2, "must_cite_titles": [], "must_not_cite_titles": [], "cites_author": null, "exclude_author": null, "cites_venue": null, "publication_types": [], "journal_only": false, "min_citations": null, "max_citations": null, "topic_terms": null}
Example query: "Journal articles by Barbara Prover with at least 20 citations, citing papers by Grace Checker, but not self-citations of Grace Checker"
Example output: {"known_title": null, "venues": [], "years": [], "year_min": null, "year_max": null, "authors_all": ["Barbara Prover"], "authors_any": [], "authors_of_paper": null, "min_authors": null, "must_cite_titles": [], "must_not_cite_titles": [], "cites_author": "Grace Checker", "exclude_author": "Grace Checker", "cites_venue": null, "publication_types": ["JournalArticle"], "journal_only": true, "min_citations": 20, "max_citations": null, "topic_terms": null}
Example query: "Papers citing the 'WidgetNet' paper and the 'GadgetBench' paper after 2020 with more than 30 citations"
Example output: {"known_title": null, "venues": [], "years": [], "year_min": 2020, "year_max": null, "authors_all": [], "authors_any": [], "authors_of_paper": null, "min_authors": null, "must_cite_titles": ["WidgetNet: A Widget Recognition Network", "GadgetBench: A Benchmark for Gadget Understanding"], "must_not_cite_titles": [], "cites_author": null, "exclude_author": null, "cites_venue": null, "publication_types": [], "journal_only": false, "min_citations": 31, "max_citations": null, "topic_terms": null}"""


# ---------------------------------------------------------------------------
# Deterministic field cleaning
# ---------------------------------------------------------------------------

def _clean_str(value: Any) -> str | None:
    """Coerce to a stripped non-empty str, else None."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _clean_str_list(value: Any) -> list[str]:
    """Coerce to a list of stripped non-empty strings (drop everything else)."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):  # LLMs sometimes wrap names/titles in dicts
            item = item.get("name") or item.get("title")
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _clean_year(value: Any) -> int | None:
    """Coerce to an int year clamped into [1000, 2100], else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # JSON "2012.0" is a year, not garbage
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return min(max(year, _YEAR_LO), _YEAR_HI)


def _clean_years(value: Any) -> list[int]:
    """Sorted, deduped list of valid years (drop everything else)."""
    if not isinstance(value, list):
        return []
    return sorted({y for y in (_clean_year(v) for v in value) if y is not None})


def _clean_count(value: Any) -> int | None:
    """Coerce to a non-negative int, else None."""
    if isinstance(value, bool):
        return None
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _norm_key(s: str) -> str:
    """Lowercase alnum-only key for whitelist lookups."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


#: S2 publicationTypes whitelist - planner output outside this set is dropped
#: rather than fed into exact set algebra as an unmatchable filter.
_PUB_TYPES = {
    "journalarticle": "JournalArticle",
    "journalarticles": "JournalArticle",
    "journal": "JournalArticle",
    "conference": "Conference",
    "conferencepaper": "Conference",
    "review": "Review",
    "book": "Book",
    "dataset": "Dataset",
}


def _clean_publication_types(value: Any) -> list[str]:
    out: list[str] = []
    for item in _clean_str_list(value):
        canon = _PUB_TYPES.get(_norm_key(item))
        if canon and canon not in out:
            out.append(canon)
    return out


def _clean_venues(value: Any) -> list[str]:
    """Venue names with any stray years stripped ("NAACL 2010" -> "NAACL")."""
    out: list[str] = []
    for item in _clean_str_list(value):
        item = re.sub(r"\b(?:19|20)\d{2}\b", "", item).strip(" ,;-")
        if item and item not in out:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Query-regex repairs (deterministic overrides of the LLM's numbers)
# ---------------------------------------------------------------------------

_Y = r"((?:19|20)\d{2})"

#: "by the authors of X" phrasing - the ONLY license for authors_of_paper.
_AUTHORS_OF_RE = re.compile(r"\bauthors?\s+of\b|'s\s+authors?\b", re.I)
#: citing-language: an anchor mentioned under these belongs in must_cite.
_CITING_RE = re.compile(r"\bcit(?:e|es|ing|ed)\b", re.I)

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUM = r"(\d+|" + "|".join(_WORD_NUM) + r")"


def _as_int(token: str) -> int | None:
    token = (token or "").strip().lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUM.get(token)


def _min_citations_from_query(query: str) -> int | None:
    """Citation-count floor stated by the query, exact-threshold mapped.

    "more than N citations" -> N+1 (strict), "at least N citations" -> N,
    "N or more citations" -> N, "cited by at least N other papers" -> N,
    "cited by more than N other papers" -> N+1. Returns None when no such
    phrase exists (the LLM's transcription then stands).
    """
    q = query or ""
    m = re.search(rf"\b(?:more\s+than|over|above|exceeding)\s+{_NUM}\s+citations?\b", q, re.I)
    if m:
        n = _as_int(m.group(1))
        return n + 1 if n is not None else None
    m = re.search(rf"\bat\s+least\s+{_NUM}\s+citations?\b", q, re.I)
    if m:
        return _as_int(m.group(1))
    m = re.search(rf"\b{_NUM}\s+or\s+more\s+citations?\b", q, re.I)
    if m:
        return _as_int(m.group(1))
    m = re.search(rf"\bcited\s+by\s+(?:at\s+least\s+)?{_NUM}\s+(?:or\s+more\s+)?other\s+papers?\b", q, re.I)
    if m:
        return _as_int(m.group(1))
    m = re.search(rf"\bcited\s+by\s+more\s+than\s+{_NUM}\s+other\s+papers?\b", q, re.I)
    if m:
        n = _as_int(m.group(1))
        return n + 1 if n is not None else None
    return None


def _max_citations_from_query(query: str) -> int | None:
    q = query or ""
    m = re.search(rf"\b(?:fewer|less)\s+than\s+{_NUM}\s+citations?\b", q, re.I)
    if m:
        n = _as_int(m.group(1))
        return max(n - 1, 0) if n is not None else None
    m = re.search(rf"\b(?:at\s+most|no\s+more\s+than)\s+{_NUM}\s+citations?\b", q, re.I)
    if m:
        return _as_int(m.group(1))
    return None


def _min_authors_from_query(query: str) -> int | None:
    """Author-count floor: "more than 3 authors" -> 4, "at least one
    additional author" -> 2 (the anchor author + N more)."""
    q = query or ""
    m = re.search(rf"\b(?:more\s+than|over)\s+{_NUM}\s+authors?\b", q, re.I)
    if m:
        n = _as_int(m.group(1))
        return n + 1 if n is not None else None
    m = re.search(rf"\bat\s+least\s+{_NUM}\s+additional\s+(?:co)?-?authors?\b", q, re.I)
    if m:
        n = _as_int(m.group(1))
        return n + 1 if n is not None else None
    m = re.search(rf"\bat\s+least\s+{_NUM}\s+(?:co)?-?authors?\b", q, re.I)
    if m:
        return _as_int(m.group(1))
    return None


def _year_repairs_from_query(query: str, plan: MetadataPlan) -> None:
    """Re-derive year constraints from the query text, overriding the LLM.

    Recognized: "YYYY or YYYY" (exact membership), "between/from YYYY and/to
    YYYY", "YYYY-YYYY" (bounds), "after/since YYYY" and "YYYY and beyond" ->
    year_min YYYY (INCLUSIVE, benchmark gold-program semantics per IRIS
    forensics), "before YYYY" -> year_max YYYY-1, "until/up to/through YYYY"
    -> year_max YYYY. Unrecognized phrasing leaves the LLM's fields alone.
    """
    q = query or ""
    pairs = re.findall(rf"\b{_Y}\s+or\s+{_Y}\b", q, re.I)
    if pairs:
        years = sorted({int(y) for pair in pairs for y in pair})
        plan.years = [min(max(y, _YEAR_LO), _YEAR_HI) for y in years]
        plan.year_min = plan.year_max = None
        return
    lo = hi = None
    m = (re.search(rf"\bbetween\s+{_Y}\s+and\s+{_Y}\b", q, re.I)
         or re.search(rf"\bfrom\s+{_Y}\s+to\s+{_Y}\b", q, re.I)
         or re.search(rf"\b{_Y}\s*[-–—]\s*{_Y}\b", q))
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
    m = re.search(rf"\b(?:after|since)\s+{_Y}\b", q, re.I)
    if m:
        lo = int(m.group(1))  # inclusive, never +1
    m = re.search(rf"\b{_Y}\s+and\s+(?:beyond|later|onwards?)\b", q, re.I)
    if m:
        lo = int(m.group(1))
    m = re.search(rf"\bbefore\s+{_Y}\b", q, re.I)
    if m:
        hi = int(m.group(1)) - 1
    m = re.search(rf"\b(?:until|up\s+to|through)\s+{_Y}\b", q, re.I)
    if m:
        hi = int(m.group(1))
    if lo is not None:
        plan.year_min = lo
        plan.years = []
    if hi is not None:
        plan.year_max = hi
        plan.years = []


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------

def build_plan(query: str, llm: Any = None) -> tuple[MetadataPlan, int | None]:
    """One planning call + deterministic cleaning -> (plan, max_citations).

    Deterministic given the LLM reply: all typing/clamping/repairs are code.
    On total planning failure the plan carries the raw query as topic_terms
    (a searchable seed); the executor then returns its best-effort pool or
    [] and the router's semantic channel can take over.
    """
    if llm is None:
        llm = _default_llm()
    obj = None
    if llm is not None:
        try:
            obj = llm.json(PLAN_SYSTEM, query, max_tokens=_PLAN_MAX_TOKENS)
        except Exception as exc:  # planning must degrade softly
            _log(f"planner call failed: {exc}")
            obj = None
    if not isinstance(obj, dict):
        return MetadataPlan(topic_terms=_clean_str(query)), None

    authors_all = _clean_str_list(obj.get("authors_all") or obj.get("authors"))
    authors_any = _clean_str_list(obj.get("authors_any"))
    # authors_any -> IRIS `authors`: the executor intersects author pools and
    # falls back to their union when the intersection is empty, which IS the
    # OR semantics for distinct people (documented approximation above).
    authors = authors_all + [a for a in authors_any if a not in authors_all]

    min_authors = _clean_count(obj.get("min_authors"))
    plan = MetadataPlan(
        known_title=_clean_str(obj.get("known_title")),
        venues=_clean_venues(obj.get("venues")),
        year_min=_clean_year(obj.get("year_min")),
        year_max=_clean_year(obj.get("year_max")),
        years=_clean_years(obj.get("years")),
        authors=authors,
        authors_of_paper=_clean_str(obj.get("authors_of_paper")),
        min_authors=min_authors if min_authors else None,  # >= 1 or absent
        must_cite=_clean_str_list(
            obj.get("must_cite_titles") if obj.get("must_cite_titles") is not None
            else obj.get("must_cite")
        ),
        must_not_cite=_clean_str_list(
            obj.get("must_not_cite_titles") if obj.get("must_not_cite_titles") is not None
            else obj.get("must_not_cite")
        ),
        cites_author=_clean_str(obj.get("cites_author")),
        exclude_author=_clean_str(obj.get("exclude_author")),
        cites_venue=_clean_str(obj.get("cites_venue")),
        publication_types=_clean_publication_types(obj.get("publication_types")),
        topic_terms=_clean_str(obj.get("topic_terms")),
        min_citations=_clean_count(obj.get("min_citations")),
    )

    # journal_only folds into the publication-type filter (null types PASS in
    # the executor - live-verified gold papers carry publicationTypes None).
    if obj.get("journal_only") is True and "JournalArticle" not in plan.publication_types:
        plan.publication_types.append("JournalArticle")

    # IRIS hedged-endpoints repair: the planner sometimes emits a wide
    # range's two ENDPOINTS as `years` alongside matching bounds ("2015 to
    # 2020" -> years [2015, 2020] + bounds). Exact membership would delete
    # every interior year, so keep the bounds and clear the list.
    if plan.years:
        if (
            plan.year_min is not None
            and plan.year_max is not None
            and set(plan.years) == {plan.year_min, plan.year_max}
            and plan.year_max - plan.year_min > 1
        ):
            plan.years = []
        else:
            # Explicit year disjunction supersedes range bounds.
            plan.year_min = plan.year_max = None

    # Query-text repairs override the LLM's numbers wherever the query
    # states them in a recognizable phrase.
    _year_repairs_from_query(query, plan)
    if plan.year_min is not None and plan.year_max is not None and plan.year_min > plan.year_max:
        plan.year_min, plan.year_max = plan.year_max, plan.year_min
    mc = _min_citations_from_query(query)
    if mc is not None:
        plan.min_citations = mc
    ma = _min_authors_from_query(query)
    if ma is not None:
        plan.min_authors = ma

    max_citations = _clean_count(obj.get("max_citations"))
    mxc = _max_citations_from_query(query)
    if mxc is not None:
        max_citations = mxc

    # authors_of_paper is only meaningful when the query LITERALLY asks for
    # papers by the authors of X ("...authors of the BERT paper"). Live gate
    # forensics (2026-08-10): on "Papers citing the DistilBERT paper ..."
    # gpt-4o-mini put the anchor title into authors_of_paper - turning
    # "papers citing X" into "papers by X's authors" and zeroing the query.
    # Deterministic repair keyed on the query text: no "authors of"-style
    # phrase -> the field is misplaced; with citing-language present the
    # title belongs in must_cite, otherwise it is dropped.
    if plan.authors_of_paper and not _AUTHORS_OF_RE.search(query or ""):
        misplaced = plan.authors_of_paper
        plan.authors_of_paper = None
        if _CITING_RE.search(query or "") and misplaced not in plan.must_cite:
            plan.must_cite.append(misplaced)

    # A query defining its answers by SET relations (citing X, by author Y,
    # authors-of-paper Z) is not asking for one known paper - a stray
    # known_title would collapse the whole pool to a single id.
    if plan.known_title and (
        plan.must_cite or plan.must_not_cite or plan.authors
        or plan.authors_of_paper or plan.cites_author or plan.cites_venue
    ):
        plan.known_title = None

    if not plan.has_constraints():
        plan.topic_terms = _clean_str(query)
    return plan, max_citations


# ---------------------------------------------------------------------------
# Author de-fragmentation
# ---------------------------------------------------------------------------

def _record_paper_count(record: dict) -> int:
    try:
        return int(record.get("paperCount") or 0)
    except (TypeError, ValueError):
        return 0


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def _name_compatible(query_name: str, record_name: str) -> bool:
    """True when ``record_name`` plausibly denotes the queried person.

    Deterministic string logic: exact normalized match; reversed token order
    ("Surname Firstname"); or initial-form first name with the same
    surname ("F. Surname" ~ "Firstname Surname"). Two DIFFERENT full
    first names are incompatible ("Dana Surname" != "David Surname"); a
    different surname is
    always incompatible.
    """
    qt = _norm_name(query_name).split()
    rt = _norm_name(record_name).split()
    if not qt or not rt:
        return False
    if qt == rt or sorted(qt) == sorted(rt):
        return True
    if qt[-1] != rt[-1]:
        return False
    qf, rf = qt[0], rt[0]
    if qf[0] != rf[0]:
        return False
    if len(qf) > 1 and len(rf) > 1 and qf != rf:
        return False
    return True


class _AuthorUnionClient:
    """Client proxy that DE-FRAGMENTS author records for the executor.

    Measured corpus behavior: the author index shards one person across
    many records. A full-name query can return a dozen exact-name
    fragments of a few papers each alongside a much larger record filed
    under the initialism form. IRIS's ``_resolve_author`` picks exactly
    one record (exact-name first, then paperCount), so whichever way it
    chooses it loses every paper stamped on the other shards, and an
    author-constrained query can score zero because the wanted papers
    all hang off the record that was not selected.

    Fix, general and query-independent: when an author search finds MORE
    than one name-compatible record, prepend ONE synthetic record whose
    ``authorId`` encodes the top-``_MAX_MERGE_RECORDS`` compatible records
    (by paperCount) and whose paperCount is their sum (IRIS's exact-name +
    max-paperCount rule then deterministically selects it), and serve
    ``get_author_papers`` for that synthetic id as the DEDUPED UNION of the
    component pulls. Incompatible records (a different surname, or a
    different full first name) never
    join the merge, so this cannot conflate different people beyond what
    the name itself underdetermines. Every other method delegates raw.
    """

    _PREFIX = "pfbmax-merge:"
    _MAX_MERGE_RECORDS = 8

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def search_authors_by_name(self, name: str):
        records = self._inner.search_authors_by_name(name) or []
        compat = [
            r for r in records
            if isinstance(r, dict) and r.get("authorId") is not None
            and _name_compatible(name, str(r.get("name") or ""))
        ]
        compat.sort(key=_record_paper_count, reverse=True)
        compat = compat[: self._MAX_MERGE_RECORDS]
        if len(compat) < 2:
            return records
        merged_id = self._PREFIX + "+".join(str(r["authorId"]) for r in compat)
        synthetic = {
            "authorId": merged_id,
            "name": name,  # verbatim -> exact-name match in _resolve_author
            "paperCount": sum(_record_paper_count(r) for r in compat),
        }
        _log(f"author merge for {name!r}: {len(compat)} records -> {merged_id}")
        return [synthetic] + list(records)

    def get_author_papers(self, author_id, limit: int = 1000, **kwargs):
        aid = str(author_id)
        if not aid.startswith(self._PREFIX):
            return self._inner.get_author_papers(author_id, limit=limit, **kwargs)
        out: list = []
        seen: set[str] = set()
        for component in aid[len(self._PREFIX):].split("+"):
            try:
                papers = self._inner.get_author_papers(
                    component, limit=limit, **kwargs
                )
            except TypeError:
                raise  # capability probe (publication_date_range) must surface
            except Exception:
                continue  # partial union beats none; components are independent
            for paper in papers or []:
                cid = _cid(paper)
                key = cid if cid else f"#{len(out)}"
                if key not in seen:
                    seen.add(key)
                    out.append(paper)
        return out


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _solve_venue_cites_venue_local(plan) -> "Submission | None":
    """Local-graph execution for the venue+cites_venue plan shape.

    Returns None (fall through to the normal executor) unless: the plan has
    both a venue constraint and cites_venue, AND the local graph databases
    exist. Venue matching uses the same generic long/short-form expansion
    for any venue string (substring both ways), no query-specific logic.
    """
    import os
    import sqlite3
    cg_path = (os.environ.get("PFBMAX_CITEGRAPH") or "").strip()
    pm_path = (os.environ.get("PFBMAX_PMETA") or "").strip()
    if not (cg_path and pm_path and plan.cites_venue and plan.venues):
        return None
    if not (os.path.exists(cg_path) and os.path.exists(pm_path)):
        return None
    try:
        from iris_asta.solvers.pfb import _canonical_venue
        cg = sqlite3.connect(cg_path)
        pm = sqlite3.connect(pm_path)
        like = lambda v: f"%{v}%"

        def acronym(name: str) -> str:
            # initials of capitalized words; general long-form <-> acronym
            # bridge (e.g. a conference's long official name vs its short
            # name), applied identically to every venue string
            return "".join(w[0] for w in name.replace(":", " ").split()
                           if w[:1].isupper()).upper()

        pool: list[tuple[str, str]] = []
        for v0 in plan.venues:
            for v in {v0, _canonical_venue(v0)}:
                vu = v.upper().strip()
                alias = set()
                if len(vu) >= 4 and vu.isalpha():
                    for (vn,) in pm.execute(
                        "SELECT DISTINCT venue FROM m WHERE venue IS NOT "
                        "NULL AND length(venue)>15"):
                        if vu in acronym(vn):
                            alias.add(vn)
                for av in list(alias)[:40]:
                    q2 = "SELECT cid, venue FROM m WHERE venue = ?"
                    a2 = [av]
                    if plan.year_min:
                        q2 += " AND year>=?"
                        a2.append(int(plan.year_min))
                    if plan.year_max:
                        q2 += " AND year<=?"
                        a2.append(int(plan.year_max))
                    pool.extend((str(r[0]), r[1] or "")
                            for r in pm.execute(q2, a2))
                q = "SELECT cid, venue FROM m WHERE venue LIKE ?"
                args = [like(v)]
                if plan.year_min:
                    q += " AND year>=?"
                    args.append(int(plan.year_min))
                if plan.year_max:
                    q += " AND year<=?"
                    args.append(int(plan.year_max))
                pool.extend((str(r[0]), r[1] or "") for r in pm.execute(q, args))
        if not pool:
            return None
        cv_can = _canonical_venue(plan.cites_venue)
        cv = plan.cites_venue.upper().strip()
        cited_ok = set(
            int(r[0]) for r in pm.execute(
                "SELECT cid FROM m WHERE venue LIKE ?",
                (like(plan.cites_venue),)))
        if cv_can != plan.cites_venue:
            cited_ok.update(int(r[0]) for r in pm.execute(
                "SELECT cid FROM m WHERE venue LIKE ?", (like(cv_can),)))
        if len(cv) >= 4 and cv.isalpha():
            cv_alias = set()
            for (vn,) in pm.execute(
                    "SELECT DISTINCT venue FROM m WHERE venue IS NOT NULL "
                    "AND length(venue)>15"):
                if cv in acronym(vn):
                    cv_alias.add(vn)
            for av in list(cv_alias)[:40]:
                cited_ok.update(int(r[0]) for r in pm.execute(
                    "SELECT cid FROM m WHERE venue = ?", (av,)))
        out = []
        seen_cids = set()
        for cid, venue in pool:
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            refs = [r[0] for r in cg.execute(
                "SELECT cited FROM edges WHERE citing=?", (int(cid),))]
            if any(x in cited_ok for x in refs):
                out.append((cid, f"Published in {venue}; cites a "
                                 f"{plan.cites_venue} paper (citation "
                                 f"graph)."))
        _log(f"local venue+cites_venue: pool {len(pool)} -> {len(out)}")
        return out or None
    except Exception as exc:
        _log(f"local venue+cites_venue failed: {exc}")
        return None


def solve_metadata(
    query: str,
    client: Any,
    llm: Any = None,
    inserted_before: str | None = None,
) -> Submission:
    """Plan -> deterministic execution -> evidence; returns the believed set.

    Contract: ``Submission = list[(corpus_id, markdown_evidence)]``, ranked
    most-relevant first (snapshot-safe citation count, from the executor).
    Emits ONLY the ids the constraint program believes in; precision is half
    of exact-set F1, so there is no padding channel. [] means "nothing
    deterministic was constructible" (router may fall back to semantic).
    """
    plan, max_citations = build_plan(query, llm)
    _log(f"plan for {query[:60]!r}: {plan} max_citations={max_citations}")
    client = _AuthorUnionClient(client)
    # GENERAL local-graph branch: venue-constrained plans with a cites_venue
    # predicate execute against the local citation graph + paper-metadata
    # side-table when PFBMAX_CITEGRAPH / PFBMAX_PMETA are set. Keyed on plan
    # SHAPE only (any venue names). Motivation, measured: the public corpus
    # API cannot serve some papers' references at all (one gold paper proved
    # unfetchable from every network and key combination tried), while the
    # S2 bulk graph contains them (local F1 0.000 -> 0.400).
    # The local branch implements venues, cites_venue and the year bounds only.
    # Any other predicate must go through the normal executor, otherwise the
    # extra constraint would be silently dropped from the answer set.
    _LOCAL_UNSUPPORTED = ("years", "publication_types", "min_citations",
                          "min_authors", "authors", "authors_of_paper",
                          "must_cite", "must_not_cite", "cites_author",
                          "exclude_author", "known_title", "topic_terms")
    if max_citations is None and not any(
            getattr(plan, f, None) for f in _LOCAL_UNSUPPORTED):
        local = _solve_venue_cites_venue_local(plan)
        if local is not None:
            return local
    try:
        ids = execute_plan(plan, client, inserted_before=inserted_before)
    except Exception as exc:  # a dead corpus must not crash the solve
        _log(f"execute_plan failed: {exc}")
        return []
    if not ids:
        return []
    if plan.cites_author:
        ids = _verify_cites_author(ids, client, plan.cites_author)
    if plan.exclude_author:
        ids = _drop_authored_by(ids, client, plan.exclude_author)
    if max_citations is not None:
        ids = _filter_max_citations(ids, client, max_citations)
    evidence = _evidence_map(client, ids[:EVIDENCE_TOP])
    return [(cid, evidence.get(cid, "")) for cid in ids]


#: Reference pulls the strict cites-author verifier may spend. Sized above the
#: executor's own walk cap so the pass is normally cache-warm and near-free.
VERIFY_REF_BUDGET = 400
VERIFY_REF_LIMIT = 200


def _verify_cites_author(ids: list[str], client: Any, name: str) -> list[str]:
    """Keep only results we can PROVE cite a paper by ``name``.

    The upstream executor evaluates this predicate inclusive-on-error: a
    candidate whose references fail to load survives. That is the right
    default when a missing answer is the worst outcome, but exact-set F1
    weighs a false positive exactly as heavily as a miss, and development
    runs showed the failure mode this creates -- recall 1.00 with precision 0.09
    (173 emitted against 16 gold), i.e. the predicate silently stopped
    filtering and we shipped the unfiltered author pool.

    So this pass re-checks each candidate and is EXCLUSIVE on error. The
    asymmetry is quantitative, not stylistic: at recall 1.0 every unverified
    candidate we keep is a guaranteed false positive, while dropping one
    costs at most one true positive.

    Safety valve: if verification proves nothing at all (a dead corpus rather
    than genuine non-citation), the unfiltered set is returned unchanged --
    never trade a real answer for an artifact of an outage.
    """
    if not (name or "").strip() or not ids:
        return ids
    kept: list[str] = []
    spent = 0
    checked = 0
    for cid in ids:
        if spent >= VERIFY_REF_BUDGET:
            break
        spent += 1
        try:
            refs = client.get_citations(cid, "references", limit=VERIFY_REF_LIMIT)
        except Exception:
            continue                      # unverifiable -> excluded
        checked += 1
        for ref in refs or []:
            authors = getattr(ref, "authors", None)
            if authors is None and isinstance(ref, dict):
                authors = ref.get("authors")
            for author in authors or []:
                nm = author.get("name") if isinstance(author, dict) else \
                    getattr(author, "name", None)
                if nm and _name_compatible(name, str(nm)):
                    kept.append(cid)
                    break
            else:
                continue
            break
    if not kept:
        _log(f"cites_author verify proved nothing for {name!r} "
             f"(checked {checked}/{len(ids)}) -- keeping unfiltered set")
        return ids
    _log(f"cites_author verify: {len(ids)} -> {len(kept)} (checked {checked})")
    return kept


def _drop_authored_by(ids: list[str], client: Any, name: str) -> list[str]:
    """Drop result ids that appear in the excluded author's own paper set.

    "not self-citations of Y" = the citing paper is not BY Y. The executor
    already excludes by author-name equality on each paper's author records;
    this second pass excludes by MEMBERSHIP in Y's de-fragmented paper set
    (all merged shards), which catches papers whose author list renders the
    name differently (an initialism form versus the full name). The pulls
    are the same
    cache keys the cites_author walk already made, so this is near-free.
    Inclusive-on-error: an unresolvable exclusion set drops nothing.
    """
    try:
        records = client.search_authors_by_name(name) or []
    except Exception:
        return ids
    author_id = None
    for record in records:  # synthetic merge record (if any) sorts first
        if isinstance(record, dict) and record.get("authorId") is not None \
                and _name_compatible(name, str(record.get("name") or "")):
            author_id = record["authorId"]
            break
    if author_id is None:
        return ids
    try:
        papers = client.get_author_papers(author_id, limit=1000)
    except Exception:
        return ids
    owned = {c for c in (_cid(p) for p in papers or []) if c}
    return [cid for cid in ids if cid not in owned]


def _filter_max_citations(ids: list[str], client: Any, max_citations: int) -> list[str]:
    """Drop ids whose citation count exceeds the cap; unknown counts KEPT.

    Inclusive-on-error mirrors the executor's reference-walk policy: a
    transient batch failure must not silently delete candidates.
    """
    counts: dict[str, int] = {}
    for start in range(0, len(ids), _BATCH_CHUNK):
        chunk = ids[start:start + _BATCH_CHUNK]
        try:
            try:
                papers = client.get_paper_batch(chunk, fields="corpusId,citationCount")
            except TypeError:  # duck-typed client without the fields kwarg
                papers = client.get_paper_batch(chunk)
        except Exception:
            continue  # unknown counts stay in
        for paper in papers or []:
            cid = _cid(paper)
            count = _citations(paper)
            if cid and count is not None:
                counts[cid] = count
    return [cid for cid in ids if counts.get(cid) is None or counts[cid] <= max_citations]


def _evidence_map(client: Any, ids: list[str]) -> dict[str, str]:
    """cid -> ``«Title» (year): abstract[:300]`` for the top ids (one batch).

    Evidence is unscored on the metadata slice; this exists to satisfy the
    submission contract with verbatim corpus text at minimum cost. Any
    failure degrades to empty evidence, never a crashed solve.
    """
    out: dict[str, str] = {}
    if not ids:
        return out
    try:
        try:
            papers = client.get_paper_batch(list(ids), fields="corpusId,title,year,abstract")
        except TypeError:
            papers = client.get_paper_batch(list(ids))
    except Exception as exc:
        _log(f"evidence batch failed: {exc}")
        return out
    for paper in papers or []:
        cid = _cid(paper)
        if not cid:
            continue
        title = (str(getattr(paper, "title", "") or "")).strip() or "Untitled"
        year = getattr(paper, "year", None)
        year_str = str(year) if isinstance(year, int) and not isinstance(year, bool) else "n.d."
        abstract = " ".join(str(getattr(paper, "abstract", "") or "").split())
        line = f"«{title}» ({year_str})"
        if abstract:
            line += ": " + abstract[:_EVIDENCE_ABSTRACT_CHARS]
        out[cid] = line
    return out


def _cid(paper: Any) -> str | None:
    """Digits-only corpus id of a paper-like object (Paper or namespace)."""
    return normalize_corpus_id(
        getattr(paper, "corpus_id", None) or getattr(paper, "corpusId", None)
    )


def _citations(paper: Any) -> int | None:
    """Citation count, or None when genuinely unknown (never a fake 0)."""
    extra = getattr(paper, "extra", None)
    value = extra.get("citationCount") if isinstance(extra, dict) else None
    if value is None:
        value = getattr(paper, "citationCount", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _log(msg: str) -> None:
    if os.environ.get("PFBMAX_METADATA_DEBUG"):
        print(f"[pfbmax.metadata] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Default LLM: shared pfbmax backbone, with a stdlib inline fallback so this
# module never blocks on pfbmax/llm.py availability.
# ---------------------------------------------------------------------------

def _default_llm() -> Any:
    try:
        from pfbmax.llm import LLM  # shared gpt-4o-mini backbone
        return LLM()
    except Exception:
        try:
            return _InlineMiniLLM()
        except Exception:
            return None


class _InlineMiniLLM:
    """Minimal gpt-4o-mini JSON caller (stdlib urllib), fallback only.

    Reads OPENAI_API_KEY from the environment or iris_asta/.env. Never used
    when ``pfbmax.llm.LLM`` imports (the normal case); unit tests inject
    fakes and never construct this.
    """

    _ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = api_key or self._find_key()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not found (env or iris_asta/.env)")

    @staticmethod
    def _find_key() -> str | None:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            return key
        env_path = os.path.join(_IRIS_DIR, ".env")
        try:
            with open(env_path, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        value = line.split("=", 1)[1].strip().strip("'\"")
                        if value:
                            return value
        except OSError:
            pass
        return None

    def json(self, system: str, user: str, max_tokens: int = 900) -> dict | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self._ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=90) as resp:
                    body = json.loads(resp.read().decode("utf-8", "replace"))
                text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                try:
                    obj = json.loads(text)
                    return obj if isinstance(obj, dict) else None
                except ValueError:
                    start, end = text.find("{"), text.rfind("}")
                    if 0 <= start < end:
                        try:
                            obj = json.loads(text[start:end + 1])
                            return obj if isinstance(obj, dict) else None
                        except ValueError:
                            return None
                    return None
            except Exception:
                if attempt == 2:
                    return None
        return None

# ---------------------------------------------------------------------------
# Venue acronym matching (additive override of the read-only executor's rule)
# ---------------------------------------------------------------------------

_VENUE_STOP = {"of", "for", "and", "the", "on", "in", "at", "to", "a", "an",
               "acm", "ieee", "sigplan", "sigmod", "sigir", "annual", "joint"}


def _acronym_matches_venue(acronym: str, venue: str) -> bool:
    """True when ``acronym`` is spelled by the initials of a contiguous run of
    the venue's content words.

    Corpora store conferences under their expanded names, so a substring test
    for the acronym fails exactly where it matters. Measured: one query asks
    for SPLASH papers, and the gold venue string is "ACM SIGPLAN International
    Conference on Systems, Programming, Languages and Applications: Software
    for Humanity". That spells S-P-L-A-S-H across six consecutive content
    words with no literal "SPLASH" anywhere, and the query scored 0.000
    purely on that.

    Contiguity keeps it honest (initials in order, no skipping), and the
    4-character floor stops short acronyms matching by chance.
    """
    a = "".join(ch for ch in (acronym or "").lower() if ch.isalnum())
    if len(a) < 4 or not venue:
        return False
    words = [w for w in re.split(r"[^A-Za-z0-9]+", venue.lower()) if w]
    words = [w for w in words if w not in _VENUE_STOP]
    initials = "".join(w[0] for w in words)
    return a in initials


def _install_venue_acronym_support() -> None:
    """Teach the read-only executor about acronym venues, additively.

    We never edit iris_asta; this wraps its matcher in-process so the rule can
    only ever ACCEPT more venues, never reject one it used to accept.
    """
    try:
        from iris_asta.solvers import pfb as _pfb
    except Exception:
        return
    if getattr(_pfb, "_pfbmax_venue_patch", False):
        return
    _orig = _pfb._venue_matches

    def _patched(paper_venue: str, wanted: list) -> bool:
        if _orig(paper_venue, wanted):
            return True
        return any(_acronym_matches_venue(w, paper_venue or "") for w in (wanted or []))

    _pfb._venue_matches = _patched
    _pfb._pfbmax_venue_patch = True


#: chosen authorId -> alias sibling ids, filled by the resolver patch.
_ALIAS_SIBLINGS: dict = {}


def _alias_norm(text: str) -> str:
    """Lowercased, punctuation-flattened name for alias comparison."""
    return " ".join(
        "".join(ch if ch.isalnum() else " " for ch in (text or "").lower()).split())


def _name_is_alias(candidate: str, wanted: str) -> bool:
    """Is ``candidate`` an initialism alias of ``wanted``?

    ("F. Surname" is an alias of "Firstname Surname".) Semantic Scholar
    fragments one researcher across several author records, and the
    FULL-NAME record is frequently the small one: the exact-name records
    may hold a handful of papers while the initialism record holds
    hundreds, including the ones a query wants. Preferring an exact
    string match therefore resolves the wrong record and empties the
    author pool.

    Alias test: same surname, and the candidate's leading token is either the
    same first name or its initial. Deliberately narrow -- it never merges two
    different surnames, so it cannot pull in an unrelated author.
    """
    c = _alias_norm(candidate).split()
    w = _alias_norm(wanted).split()
    if len(c) < 2 or len(w) < 2:
        return False
    if c[-1] != w[-1]:
        return False
    return c[0] == w[0] or (len(c[0]) == 1 and c[0] == w[0][:1])


def _install_author_alias_support() -> None:
    """Let author resolution see initialism aliases, additively.

    iris_asta stays read-only; this wraps its resolver in-process. It only
    ever WIDENS the candidate pool (exact matches remain eligible) and still
    picks by paperCount, so it cannot resolve an author the original rule
    would have resolved better.
    """
    try:
        from iris_asta.solvers import pfb as _pfb
    except Exception:
        return
    if getattr(_pfb, "_pfbmax_author_patch", False):
        return
    _orig = _pfb._resolve_author

    def _patched(client, name):
        try:
            candidates = client.search_authors_by_name(name) or []
        except Exception:
            return _orig(client, name)
        candidates = [c for c in candidates if isinstance(c, dict)]
        if not candidates:
            return _orig(client, name)
        want = _alias_norm(name)
        compat = [c for c in candidates
                  if _alias_norm(str(c.get("name") or "")) == want
                  or _name_is_alias(str(c.get("name") or ""), name)]
        if not compat:
            return _orig(client, name)

        def _count(rec):
            try:
                return int(rec.get("paperCount") or 0)
            except (TypeError, ValueError):
                return 0

        best = max(compat, key=_count)
        # Remember the sibling records so the author's paper set can be
        # DE-FRAGMENTED later. S2 splits one researcher across records
        # (a full name may hold 88 + 12 + 4 papers across records, plus a
        # reversed-order record with 1), and a
        # membership test built from only the largest record silently fails
        # for every paper living in a fragment.
        best_id = best.get("authorId") or best.get("id")
        if best_id is not None:
            sibs = [c.get("authorId") or c.get("id") for c in compat]
            _ALIAS_SIBLINGS[str(best_id)] = [str(x) for x in sibs
                                             if x is not None and str(x) != str(best_id)]
        return best

    _pfb._resolve_author = _patched

    _orig_papers = _pfb._author_papers

    def _patched_papers(client, author_id, date_range=None):
        papers = list(_orig_papers(client, author_id, date_range) or [])
        # MEASURED: de-fragmenting the author's paper set fixes RECALL
        # completely (recall 0.31 -> 1.00 on the affected query, all 16
        # golds found) but explodes the candidate pool, and exact-set F1
        # punishes that harder than it rewards the recall: p 0.71 -> 0.089
        # (179 returned for 16 gold), F1 0.435 -> 0.164. The downstream
        # filters (JournalArticle / min_citations / exclude_author) cannot
        # discriminate the widened pool, so this stays OFF until they can.
        # Opt in with PFBMAX_AUTHOR_DEFRAG=1.
        if os.environ.get("PFBMAX_AUTHOR_DEFRAG", "").strip() not in ("1", "true", "yes"):
            return papers
        seen = {id(p) for p in papers}
        for sib in _ALIAS_SIBLINGS.get(str(author_id), []):
            try:
                for p in _orig_papers(client, sib, date_range) or []:
                    if id(p) not in seen:
                        seen.add(id(p))
                        papers.append(p)
            except Exception:
                continue
        return papers

    _pfb._author_papers = _patched_papers
    _pfb._pfbmax_author_patch = True


_install_venue_acronym_support()
_install_author_alias_support()
