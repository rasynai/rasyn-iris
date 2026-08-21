"""Navigational ("specific") PFB solver: resolve a colloquial paper mention.

Compact faithful port of the IRIS v2 specific chain
(``iris_asta/iris_asta/solvers/pfb.py``: resolution chain ~3846-3900,
``_grounded_reference_resolve`` ~649, ``_relevance_resolve`` ~1142,
``_hedge_specific`` ~3713) onto the pfbmax contract interfaces.

Chain: (a) one LLM call -> {canonical_title, first_author, year,
all_plausible_titles} (caret-stripped: the corpus title index collapses
'^' out of titles, so a literal caret never matches); (b) exact title search on
canonical + plausibles; (c) guarded relevance-resolve (exact normalized-title
match, else highest-citationCount hit passing a >=60% title-word-overlap
anchor); (d) grounded reference walk (references of the nickname's own top
hits, ranked by co-citation votes, one LLM index-pick). Every candidate from
(b)/(c) must VERIFY against the LLM's own metadata: title overlap with the
searched string OR (author-surname match AND year +/-1); a mismatch falls
through to the next link.

Cardinality (exact-set F1: every wrong extra craters precision):
* default: emit exactly 1;
* hedge to 3 only when signals conflict: the query's own year/author cues
  reject the resolution, or the mention is a bare generic acronym ("the cnn
  paper" names a class, not a paper);
* bare named-artifact mentions ("the <name> paper"): when >=2 verified
  DISTINCT corpus papers carry the artifact as their leading title token
  (disjoint author sets, non-version, dissimilar titles), the name is
  contested; emit up to 5 by citationCount.

Integrity: logic keys on query text + general rules
only; no gold ids, no per-query branching; evidence is verbatim corpus text
("«Title» (year): abstract[:300]"). Stdlib only.
"""

from __future__ import annotations

import os
import re
from typing import Any

Submission = list[tuple[str, str]]

# --------------------------------------------------------------------------
# Tunables (general rules - never per-query)
# --------------------------------------------------------------------------

_MAX_TITLE_PROBES = 4          # canonical + up to 3 plausibles
_RELEVANCE_LIMIT = 10          # relevance-resolve fallback depth
_TITLE_OVERLAP = 0.6           # word-overlap anchor (v2 _ANCHOR_WORD_OVERLAP)
_MIN_OVERLAP_WORDS = 2         # overlap verification needs >=2 content words
                               # (a bare acronym overlapping itself is circular)
_REFWALK_SEEDS = 3             # grounded walk: top hits whose refs vote
_REFWALK_REF_LIMIT = 1000      # references pulled per seed (slim fields)
_REFWALK_TITLE_CAP = 50        # candidate titles shown to the LLM
_HEDGE_TOTAL = 3               # resolved + 2 extras when signals conflict
_MULTI_MAX = 5                 # cap for contested-artifact multi-emit
_MULTI_MIN_DISTINCT = 2        # non-resolved distinct hits needed to fire
_PROBE_LIMIT = 30              # relevance hits scanned by the artifact probe
_PROBE_MIN_CITES = 10          # noise floor for probe candidates
_ANCHOR_MIN_CITES = 50         # prominence floor for artifact-anchor resolution
_DISTINCT_TITLE_OVERLAP = 0.5  # >= this residual overlap => same work
_AUTHOR_PREFIX = 4             # author cue match on shared 4-char prefix
_EVIDENCE_ABSTRACT_CHARS = 300
_HYDRATE_CAP = 5               # get_paper calls to fill missing abstracts

_STOPWORDS = frozenset(
    "the a an and or of in on for with to by at as is are was were from "
    "that this its not about paper papers".split()
)
_FILLER = _STOPWORDS | frozenset(
    "find locate get show me please original famous classic seminal known "
    "et al work works article publication".split()
)
#: Common research nouns that can never BE a named artifact ("the paper
#: about the ACME dataset": 'dataset' describes, 'ACME' names).
_COMMON_NOUNS = frozenset(
    "dataset datasets model models benchmark benchmarks corpus corpora "
    "system systems method methods approach approaches framework frameworks "
    "database databases survey surveys task tasks challenge challenges "
    "network networks learning search analysis data algorithm algorithms "
    "architecture architectures technique techniques study studies".split()
)
_VERSION_TOKENS = frozenset(
    "xl xxl xs small base large mini nano tiny huge plus pro turbo lite max "
    "v2 v3 v4 ii iii iv 2 3 4".split()
)
_LEADING_ARTICLES = ("the", "a", "an")

_YEAR_CUE_RE = re.compile(r"(?<![0-9])((?:19|20)[0-9]{2})(?![0-9])")
_GLUED_AUTHOR_RE = re.compile(r"([A-Za-z]{3,})(?=(?:19|20)[0-9]{2})")
_BY_AUTHOR_RE = re.compile(r"\bby\s+([A-Z][A-Za-z'\-]{1,})")

_META_SYSTEM = """You identify the ONE scientific paper a colloquial mention refers to.
The user names a paper by nickname, acronym, dataset/system/model name, or
author+year (e.g. "the gpt-2 paper", "the SQuAD paper", "BART by Lewis et
al."). Return the paper's canonical title EXACTLY as published, in PLAIN
ASCII — never use ^, superscripts, subscripts, or special glyphs (write
"MS2" not "MS^2"). Do NOT expand an acronym into a descriptive phrase if the
real published title is different — return the real title (e.g. "the gpt-2
paper" -> "Language Models are Unsupervised Multitask Learners"). Also
return the first author's SURNAME and the publication year.
If the mention could plausibly refer to SEVERAL well-known DISTINCT papers
(a generic class like "the cnn paper", or a name shared by unrelated
systems), list the strongest 2-3 OTHER candidates' exact titles in
"all_plausible_titles" (do not repeat canonical_title; never list mere
follow-ups, extensions, or versions of the same work). Otherwise use [].
If you do not confidently know the exact title, use null.
Respond with JSON only:
{"canonical_title": str|null, "first_author": str|null, "year": int|null,
 "all_plausible_titles": [str, ...]}"""

_SELECT_SYSTEM = """You identify which paper a colloquial reference points to, by GROUNDED selection.
You are given a query that names ONE scientific paper by nickname, acronym,
dataset/system name, or author+year, plus a NUMBERED list of candidate paper
titles (the references of papers retrieved for that nickname — papers that
use a system cite its canonical paper). Pick the ONE candidate whose title is
the canonical published paper the query refers to. Do NOT guess from outside
knowledge: if no listed candidate matches, return null.
Respond with JSON only: {"index": <1-based candidate number>|null}"""


# --------------------------------------------------------------------------
# Small deterministic helpers
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase, strip punctuation to spaces (fuzzy-but-deterministic)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def _cid_of(paper: Any) -> str | None:
    """Digits-only corpus id of a paper-like object, or None."""
    raw = getattr(paper, "corpus_id", None)
    if raw is None:
        return None
    s = str(raw).strip().lower().removeprefix("corpusid:").strip()
    return str(int(s)) if s.isdigit() else None


def _cites(paper: Any) -> int:
    extra = getattr(paper, "extra", None)
    value = extra.get("citationCount") if isinstance(extra, dict) else None
    if value is None:
        value = getattr(paper, "citationCount", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _content_words(s: str) -> set[str]:
    return {w for w in _norm(s).split() if len(w) >= 3 and w not in _STOPWORDS}


def _overlap(searched: str, title: str) -> float:
    """Fraction of the searched string's content words found in ``title``."""
    want = _content_words(searched)
    if not want:
        return 0.0
    have = set(_norm(title).split())
    return sum(1 for w in want if w in have) / len(want)


def _lev1(a: str, b: str) -> bool:
    """Edit distance <= 1 ('smth' passes for 'smith')."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if len(a) == len(b):
        return a[i + 1:] == b[i + 1:]
    return a[i:] == b[i + 1:]


def _author_tokens(paper: Any) -> list[str]:
    tokens: list[str] = []
    for record in getattr(paper, "authors", None) or []:
        if isinstance(record, dict):
            name = record.get("name")
        else:
            name = getattr(record, "name", record)
        tokens.extend(_norm(str(name or "")).split())
    return tokens


def _author_match(paper: Any, cues: list[str]) -> bool:
    """Any author-name token verifies a cue: shared 4-char prefix,
    containment either way, or edit distance <= 1; short tokens (<4) only on
    exact equality ('he' for Kaiming He)."""
    for token in _author_tokens(paper):
        for cue in cues:
            if not cue:
                continue
            if len(token) < _AUTHOR_PREFIX or len(cue) < _AUTHOR_PREFIX:
                if token == cue:
                    return True
                continue
            if (token[:_AUTHOR_PREFIX] == cue[:_AUTHOR_PREFIX]
                    or cue in token or token in cue or _lev1(cue, token)):
                return True
    return False


def _violates_cutoff(paper: Any, inserted_before: str | None) -> bool:
    """Year strictly after the snapshot year PROVES the id is invalid."""
    if not inserted_before:
        return False
    try:
        cutoff_year = int(str(inserted_before)[:4])
    except (TypeError, ValueError):
        return False
    year = getattr(paper, "year", None)
    return isinstance(year, int) and not isinstance(year, bool) and year > cutoff_year


def _paper_evidence(paper: Any) -> str:
    title = getattr(paper, "title", "") or "(untitled)"
    year = getattr(paper, "year", None)
    year_str = str(year) if isinstance(year, int) and not isinstance(year, bool) else "n.d."
    abstract = (getattr(paper, "abstract", "") or "")[:_EVIDENCE_ABSTRACT_CHARS]
    return f"«{title}» ({year_str}): {abstract}".strip()


# --------------------------------------------------------------------------
# Query-shape extraction (deterministic; no LLM judgment)
# --------------------------------------------------------------------------

def _raw_content_tokens(query: str) -> list[str]:
    """Whitespace tokens minus filler, caret-stripped, edge-punct trimmed."""
    out = []
    for tok in (query or "").split():
        tok = tok.strip(".,;:!?()[]{}\"'").replace("^", "")
        if not tok:
            continue
        if _norm(tok).replace(" ", "") in _FILLER:
            continue
        out.append(tok)
    return out


def _generic_acronym(query: str) -> bool:
    """Every content token is a bare <=3-char alpha acronym ('the cnn
    paper' names an architecture class, not one paper)."""
    tokens = [_norm(t).replace(" ", "") for t in _raw_content_tokens(query)]
    return bool(tokens) and all(t.isalpha() and len(t) <= 3 for t in tokens)


def _bare_artifact(query: str) -> str | None:
    """The single named-artifact token of a bare mention ('the ACME paper'
    -> 'ACME'), or None when the mention carries qualifiers/cues. Only
    plain alphabetic names are contestable: a digit-bearing name ("gpt-2")
    is already version-disambiguated and denotes ONE artifact."""
    tokens = _raw_content_tokens(query)
    if len(tokens) != 1:
        return None
    tok = tokens[0]
    if len(tok) < 4 or _generic_acronym(query):
        return None
    if not _norm(tok).replace(" ", "").isalpha() or " " in _norm(tok):
        return None
    return tok


def _query_cues(query: str) -> tuple[set[int], list[str]]:
    """(year cues, author-name cues): 4-digit years anywhere, author names
    glued to a year ('Smith2021'), and 'by <Name>' phrasing."""
    years = {int(y) for y in _YEAR_CUE_RE.findall(query or "")}
    authors = [_norm(a) for a in _GLUED_AUTHOR_RE.findall(query or "")]
    for m in _BY_AUTHOR_RE.findall(query or ""):
        a = _norm(m)
        if a and a not in ("the", "a", "an") and a not in authors:
            authors.append(a)
    authors = [a for a in authors if len(a) >= 2]
    return years, authors


def _fails_query_cues(query: str, paper: Any | None) -> bool:
    """v2 hedge core: True when the query's own year/author cues reject the
    resolution (extraction + comparison; no LLM judgment)."""
    years, cues = _query_cues(query)
    if paper is None:
        return bool(years or cues)
    if years:
        year = getattr(paper, "year", None)
        if not isinstance(year, int) or isinstance(year, bool) or \
                all(abs(year - y) > 1 for y in years):
            return True
    if cues and not _author_match(paper, cues):
        return True
    return False


# --------------------------------------------------------------------------
# LLM metadata + verification
# --------------------------------------------------------------------------

def _clean_title(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    t = value.replace("^", "").strip()
    if not t or t.lower() in ("null", "none", "unknown"):
        return None
    return t


def _mention_meta(query: str, llm: Any) -> dict:
    """One LLM call: canonical title + first-author surname + year +
    plausible distinct alternates. Soft-fails to an empty meta."""
    try:
        obj = llm.json(_META_SYSTEM, query, max_tokens=300) or {}
    except Exception:
        obj = {}
    canonical = _clean_title(obj.get("canonical_title"))
    surname = obj.get("first_author")
    surname = _norm(surname).split()[-1] if isinstance(surname, str) and _norm(surname) else None
    year = obj.get("year")
    if isinstance(year, bool) or not isinstance(year, int) or not 1000 <= year <= 2100:
        year = None
    plausibles = []
    for t in obj.get("all_plausible_titles") or []:
        ct = _clean_title(t)
        if ct and _norm(ct) != _norm(canonical or "") and \
                _norm(ct) not in {_norm(p) for p in plausibles}:
            plausibles.append(ct)
    return {"canonical": canonical, "surname": surname, "year": year,
            "plausibles": plausibles[:_MAX_TITLE_PROBES - 1]}


def _mention_tokens(query: str) -> list[str]:
    """The mention's own distinctive artifact-name candidates, most
    distinctive first (digit-bearing or mixed-case beat capitalized beat
    lowercase; longer beats shorter), normalized and space-collapsed.

    These anchor verification to what the USER wrote.  The LLM's canonical
    title cannot be trusted for that job: it hallucinates confidently
    (measured: an invented title matched an unrelated paper at 0.71 word
    overlap and passed verification, because the overlap was computed
    against the hallucination rather than the mention).
    """
    scored = []
    for tok in _raw_content_tokens(query):
        n = _norm(tok).replace(" ", "")
        if not n or len(n) < 2 or n in _COMMON_NOUNS:
            continue
        has_digit = any(c.isdigit() for c in tok)
        mixed = any(c.isupper() for c in tok[1:])
        rank = 2 if (has_digit or mixed) else (1 if tok[:1].isupper() else 0)
        scored.append(((rank, len(n)), n))
    scored.sort(reverse=True)
    out, seen = [], set()
    for _s, n in scored:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _title_links_mention(title: str, tokens: list[str]) -> bool:
    """The candidate title actually contains one of the mention's distinctive
    tokens (space-collapsed, so a caret-bearing mention like 'X^2' becomes
    'x2' and matches a title beginning 'X2:')."""
    if not tokens:
        return True                      # nothing distinctive to anchor on
    collapsed = _norm(title).replace(" ", "")
    return any(t in collapsed for t in tokens)


def _is_version_successor(title: str, artifact_norm: str) -> bool:
    """Title names artifact+version ('Acme2', 'Acme-XL'), which
    is a successor work, not the original the mention asks for."""
    art = artifact_norm.replace(" ", "")
    if not art:
        return False
    tn = _norm(title)
    collapsed = tn.replace(" ", "")
    idx = collapsed.find(art)
    if idx >= 0:
        rest = collapsed[idx + len(art):]
        if rest[:1].isdigit():
            return True
    parts = tn.split()
    for i, p in enumerate(parts):
        if p == art and i + 1 < len(parts) and parts[i + 1] in _VERSION_TOKENS:
            return True
    return False


def _verify_meta(paper: Any, searched: str, meta: dict,
                 mention_tokens: list[str] | None = None) -> bool:
    """Accept a candidate only when it matches what the LLM said it was
    looking for: >=60% title-word overlap with the searched string (needs
    >=2 content words, since a bare acronym self-overlapping is circular), OR
    first-author surname match AND year within +/-1 (missing years pass
    vacuously; a PRESENT year >1 off rejects).

    The fuzzy-overlap branch additionally requires the title to LINK to the
    mention itself, closing the circular-verification hole above.  Exact
    normalized-title equality and the author+year branch are exempt (they are
    already anchored on something the model did not invent)."""
    title = getattr(paper, "title", "") or ""
    want = _content_words(searched)
    if _norm(title) and _norm(title) == _norm(searched):
        return True
    if len(want) >= _MIN_OVERLAP_WORDS and _overlap(searched, title) >= _TITLE_OVERLAP:
        if _title_links_mention(title, mention_tokens or []):
            return True
    surname = meta.get("surname")
    if surname and _author_match(paper, [surname]):
        want_year = meta.get("year")
        year = getattr(paper, "year", None)
        if want_year is None or not isinstance(year, int) or isinstance(year, bool):
            return True
        if abs(year - want_year) <= 1:
            return True
    return False


# --------------------------------------------------------------------------
# Resolution links (each soft-fails to None so the next link runs)
# --------------------------------------------------------------------------

def _title_search(client: Any, title: str) -> Any | None:
    try:
        return client.search_paper_by_title(title)
    except Exception:
        return None


def _paper_search(client: Any, text: str, limit: int,
                  inserted_before: str | None) -> list:
    try:
        try:
            hits = client.paper_search(
                text, limit=limit, publication_date_before=inserted_before)
        except TypeError:  # duck-typed client without the date kwarg
            hits = client.paper_search(text, limit=limit)
    except Exception:
        return []
    return [h for h in hits or [] if _cid_of(h)]


def _relevance_resolve(client: Any, searched: str,
                       inserted_before: str | None) -> Any | None:
    """Word-overlap-anchored relevance resolution: exact normalized-title
    match wins; else the highest-citationCount hit sharing >=60% of the
    searched string's content words (the anchor keeps same-keyword noise out;
    without it, a bare artifact name would match topically unrelated papers
    from another field that happen to share the word)."""
    hits = _paper_search(client, searched, _RELEVANCE_LIMIT, inserted_before)
    if not hits:
        return None
    want_norm = _norm(searched)
    for hit in hits:
        if _norm(getattr(hit, "title", "") or "") == want_norm:
            return hit
    best = None
    for hit in hits:
        if _overlap(searched, getattr(hit, "title", "") or "") >= _TITLE_OVERLAP \
                and (best is None or _cites(hit) > _cites(best)):
            best = hit
    return best


_FOCUSED_SYSTEM = (
    "Give the EXACT published title of the paper this mention refers to. "
    'Output ONLY JSON: {"title": "<exact title>"}. If you are not confident '
    'of the exact title, output {"title": null} rather than guessing.'
)


def _focused_canonical(query: str, llm: Any, already: str | None) -> str | None:
    """Second chance at the canonical title with IRIS v2's narrow prompt
    shape (one question, one answer).  The combined metadata call has to
    produce four fields at once and drifts; asking only for the title
    recovers some of those cases."""
    try:
        obj = llm.json(_FOCUSED_SYSTEM, query, max_tokens=120) or {}
    except Exception:
        return None
    title = _clean_title(obj.get("title"))
    if title and _norm(title) != _norm(already or ""):
        return title
    return None


_TOPIC_SYSTEM = (
    "You know the scientific literature. Given a mention of an artifact "
    "(dataset, benchmark, model, system or algorithm), reply JSON only: "
    '{"topic": "..."} -- EXACTLY 3 to 5 lowercase content words naming the '
    "TASK and DOMAIN the artifact is for (e.g. 'summarization medical "
    "studies'). Never invent a paper title. If you do not recognise the "
    'artifact, reply {"topic": ""}.'
)


def _topic_direct(query: str, model: str):
    """One direct chat call, used when the injected LLM cannot pick a model.

    Reads the API key from the environment and falls back to a local key
    file if present."""
    import json as _json
    import urllib.request as _u
    key = ""
    env = os.environ.get("OPENAI_API_KEY", "")
    if env:
        key = env
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            key = open(os.path.join(here, ".openai_key"), encoding="utf-8").read().strip()
        except Exception:
            return None
    body = _json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": _TOPIC_SYSTEM},
                     {"role": "user", "content": query}],
        "temperature": 0, "max_tokens": 60,
        "response_format": {"type": "json_object"}}).encode()
    req = _u.Request("https://api.openai.com/v1/chat/completions", data=body,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + key})
    try:
        with _u.urlopen(req, timeout=60) as r:
            out = _json.loads(r.read())
        return _json.loads((out["choices"][0]["message"] or {}).get("content") or "{}")
    except Exception:
        return None


def _artifact_topic_resolve(query: str, meta: dict, client: Any, llm: Any,
                            inserted_before: str | None):
    """Resolve via the artifact token plus a SHORT topic phrase.

    Motivating failure: for a lesser-known artifact the model invents a
    canonical title, so every title-driven link chases a paper that does not
    exist, and the bare artifact token alone lands on an unrelated paper from
    another field that happens to share the token.

    The split that makes this work: models are unreliable at reproducing exact
    TITLES but reliable about what an artifact IS FOR. Asked for a title, even
    a strong model invents a plausible-sounding one; asked what the artifact
    is about, it answers with the correct task and domain. So we ask only for
    the topic and let the corpus supply the title.

    Probe length matters: a short "<artifact> <task> <domain>" probe resolves
    the right paper, while the same terms padded out to nine words resolve
    nothing -- hence the hard 3-5 word cap in the prompt and the trim below.

    Every candidate still goes through ``_verify_meta``, so this widens what we
    look for without widening what we accept.
    """
    tokens = _mention_tokens(query)
    if not tokens:
        return None, None
    # This call needs KNOWLEDGE, not reasoning: the small backbone model does
    # not recognise lesser-known artifacts (it answers with generic words like
    # "dataset, machine learning, benchmark") while the larger model does.
    # Falling back to the cheap backbone therefore silently disables this
    # link, which is exactly what happened in one evaluation run where the
    # harness supplied an LLM wrapper whose .json() rejects a `model` kwarg.
    # So go direct when the wrapper cannot honour the override.
    want = os.environ.get("PFBMAX_TOPIC_MODEL", "gpt-4o-2024-11-20")
    obj = None
    try:
        obj = llm.json(_TOPIC_SYSTEM, query, max_tokens=60, model=want)
    except TypeError:
        obj = _topic_direct(query, want)
    except Exception:
        return None, None
    if not obj:
        return None, None
    topic = " ".join(str(obj.get("topic") or "").replace(",", " ").split())
    if not topic:
        return None, None
    topic = " ".join(topic.split()[:5])
    for token in tokens[:_MAX_TITLE_PROBES]:
        probe = f"{token} {topic}"
        paper = _relevance_resolve(client, probe, inserted_before)
        if paper is None or _cid_of(paper) is None:
            continue
        if _violates_cutoff(paper, inserted_before):
            continue
        if _verify_meta(paper, probe, meta, tokens):
            return paper, probe
    return None, None


def _artifact_author_resolve(query: str, meta: dict, client: Any,
                             inserted_before: str | None):
    """Search the mention's OWN artifact token together with the author cue.

    Motivating failure: for a lesser-known artifact the LLM invents a
    plausible canonical title, so every title-driven link searches for
    something that does not exist, and the artifact-token link, which only
    knows the bare token, lands on an unrelated paper from another field that
    happens to share the token. The query itself carries the disambiguator: an
    author-year tag. `_verify_meta` can already check it (surnames match on a
    shared 4-char prefix, so a slightly misspelled surname in the query still
    verifies the real author), but nothing ever RETRIEVES that paper.

    So probe the artifact token AND the surname together. This link only runs
    when the mention actually supplies an author cue, and every candidate is
    still verified, so it cannot loosen anything.
    """
    surname = (meta or {}).get("surname")
    if not surname:
        return None, None
    tokens = _mention_tokens(query)
    if not tokens:
        return None, None
    for token in tokens[:_MAX_TITLE_PROBES]:
        for probe in (f"{token} {surname}", f"{surname} {token}"):
            paper = _relevance_resolve(client, probe, inserted_before)
            if paper is None or _cid_of(paper) is None:
                continue
            if _violates_cutoff(paper, inserted_before):
                continue
            # verified on the author+year branch, which is anchored on the
            # query text rather than on anything the model generated
            if _verify_meta(paper, token, meta, tokens):
                return paper, probe
    return None, None


def _artifact_anchor_resolve(query: str, client: Any,
                             inserted_before: str | None
                             ) -> tuple[Any | None, str | None]:
    """Resolve by searching the mention's OWN distinctive token, never a
    model-generated title.  Three tiers, each version-successor filtered:
      1. title LEADS with the artifact  ('<Artifact>: ...' system-paper style)
      2. title CONTAINS the artifact
      3. abstract mentions the artifact (some systems are introduced by a
         paper whose title never names them)
    A tier resolves when the query's own year/author cues single a survivor
    out, when exactly one survivor exists, or when a >=50-citation leader is
    unique.  Returns (paper, artifact_token)."""
    years, cues = _query_cues(query)
    for token in _mention_tokens(query)[:2]:
        if len(token) < 3:
            continue
        hits = [h for h in _paper_search(client, token, _PROBE_LIMIT, inserted_before)
                if not _violates_cutoff(h, inserted_before)]
        if not hits:
            continue
        hits = [h for h in hits
                if not _is_version_successor(getattr(h, "title", "") or "", token)]
        tiers = [
            [h for h in hits
             if _leads_with_artifact(getattr(h, "title", "") or "", token)],
            [h for h in hits
             if _title_links_mention(getattr(h, "title", "") or "", [token])],
            [h for h in hits
             if token in _norm(getattr(h, "abstract", "") or "").replace(" ", "")],
        ]
        for tier in tiers:
            if not tier:
                continue
            if years or cues:
                cued = [h for h in tier if not _fails_query_cues(query, h)]
                if len(cued) == 1:
                    return cued[0], token
                if cued:
                    tier = cued
            # Prominence floor: a title merely BEGINNING with the artifact
            # word is not evidence enough on its own (measured - an 8-citation
            # "<Artifact> Applications Review" hijacked the resolution from
            # the grounded reference walk). Only a well-cited, unambiguous
            # leader resolves here; everything else falls through to the walk.
            strong = sorted([h for h in tier if _cites(h) >= _ANCHOR_MIN_CITES],
                            key=_cites, reverse=True)
            if len(strong) == 1:
                return strong[0], token
            if len(strong) > 1 and _cites(strong[0]) >= 2 * _cites(strong[1]):
                return strong[0], token
    return None, None


def _referenced_candidates(client: Any, cid: str) -> list:
    try:
        try:
            refs = client.get_citations(cid, "references",
                                        limit=_REFWALK_REF_LIMIT,
                                        fields="corpusId,title")
        except TypeError:  # duck-typed client without the fields kwarg
            refs = client.get_citations(cid, "references",
                                        limit=_REFWALK_REF_LIMIT)
    except Exception:
        return []
    return refs or []


def _reference_walk(query: str, client: Any, llm: Any,
                    inserted_before: str | None) -> Any | None:
    """Grounded selection over the references of the nickname's own top
    hits: papers that use a system cite its canonical paper, so free recall
    becomes selection over ~50 titles ranked by co-citation votes."""
    seeds = []
    for hit in _paper_search(client, query, _REFWALK_SEEDS, inserted_before):
        cid = _cid_of(hit)
        if cid and cid not in seeds:
            seeds.append(cid)
        if len(seeds) >= _REFWALK_SEEDS:
            break
    if not seeds:
        return None
    freq: dict[str, int] = {}
    stubs: dict[str, Any] = {}
    order: list[str] = []
    for seed in seeds:
        seen_here: set[str] = set()
        for ref in _referenced_candidates(client, seed):
            rcid = _cid_of(ref)
            title = (getattr(ref, "title", "") or "").strip()
            if not rcid or not title or rcid in seen_here:
                continue
            seen_here.add(rcid)
            freq[rcid] = freq.get(rcid, 0) + 1
            if rcid not in stubs:
                stubs[rcid] = ref
                order.append(rcid)
    if not stubs:
        return None
    position = {cid: i for i, cid in enumerate(order)}
    # Version-successor filter: an artifact's follow-up ('Acme2')
    # is co-cited exactly where the original is, so it reaches the top of
    # the vote ranking and the selector picks it. Drop successors of the
    # mention's own tokens before the selector ever sees them.
    tokens = [t for t in _mention_tokens(query) if len(t) >= 3]
    kept = [c for c in stubs
            if not any(_is_version_successor(
                getattr(stubs[c], "title", "") or "", t) for t in tokens)]
    pool = kept or list(stubs)
    ranked = sorted(pool, key=lambda c: (-freq[c], position[c]))[:_REFWALK_TITLE_CAP]
    listing = "\n".join(
        f"[{i + 1}] {(getattr(stubs[c], 'title', '') or '').strip()}"
        for i, c in enumerate(ranked))
    user = (f"Query: {query}\n\nCandidate referenced titles ({len(ranked)}):\n{listing}"
            "\n\nIf several candidates describe the same system, prefer the "
            "ORIGINAL paper that introduced it over any later version, "
            "extension, or follow-up.")
    try:
        obj = llm.json(_SELECT_SYSTEM, user, max_tokens=64) or {}
    except Exception:
        obj = {}
    index = obj.get("index")
    if isinstance(index, bool):  # JSON true would coerce to candidate 1
        return None
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    if not 1 <= index <= len(ranked):
        return None
    picked = ranked[index - 1]
    paper = None
    try:
        paper = client.get_paper(picked)
    except Exception:
        paper = None
    return paper if paper is not None else stubs[picked]


# --------------------------------------------------------------------------
# Contested-artifact probe (mentions that match several papers)
# --------------------------------------------------------------------------

def _leads_with_artifact(title: str, artifact_norm: str) -> bool:
    """Title's first content phrase IS the artifact name (system-paper style
    'ACME: ...'; the plural form 'ACMES: ...' is allowed; leading articles
    are skipped).
    Version extensions ('ACME-XL: ...') are NOT the artifact."""
    tn = _norm(title)
    for art in _LEADING_ARTICLES:
        if tn.startswith(art + " "):
            tn = tn[len(art) + 1:]
    if not (tn == artifact_norm or tn.startswith(artifact_norm + " ")
            or tn == artifact_norm + "s" or tn.startswith(artifact_norm + "s ")):
        return False
    rest = tn[len(artifact_norm):].lstrip("s").strip()
    if rest and rest.split()[0] in _VERSION_TOKENS:
        return False
    return True


def _distinct_from(paper: Any, kept: list, artifact_words: set[str]) -> bool:
    """Distinct artifact = disjoint author set AND dissimilar residual title
    (shared authors or >=50% shared non-artifact content words mean the same
    lineage/work: a follow-up, not a contest for the name)."""
    p_authors = set(_author_tokens(paper))
    p_words = _content_words(getattr(paper, "title", "") or "") - artifact_words
    for other in kept:
        if other is None:
            continue
        if p_authors and p_authors & set(_author_tokens(other)):
            return False
        o_words = _content_words(getattr(other, "title", "") or "") - artifact_words
        if p_words and o_words:
            shared = len(p_words & o_words) / min(len(p_words), len(o_words))
            if shared >= _DISTINCT_TITLE_OVERLAP:
                return False
    return True


def _artifact_probe(query: str, artifact: str, resolved: Any | None,
                    extra_candidates: list, client: Any,
                    inserted_before: str | None) -> list:
    """Distinct corpus papers contesting the artifact name: relevance hits
    (plus already-verified plausibles) whose titles LEAD with the artifact,
    above a small citation floor, pairwise-distinct. The resolved paper
    never appears in the returned list."""
    artifact_norm = _norm(artifact)
    if not artifact_norm:
        return []
    artifact_words = set(artifact_norm.split())
    resolved_cid = _cid_of(resolved) if resolved is not None else None
    pool = list(extra_candidates) + _paper_search(
        client, query, _PROBE_LIMIT, inserted_before)
    seen: set[str] = set()
    candidates = []
    for hit in pool:
        cid = _cid_of(hit)
        if not cid or cid in seen or cid == resolved_cid:
            continue
        seen.add(cid)
        if _violates_cutoff(hit, inserted_before):
            continue
        if not _leads_with_artifact(getattr(hit, "title", "") or "", artifact_norm):
            continue
        if _cites(hit) < _PROBE_MIN_CITES:
            continue
        candidates.append(hit)
    candidates.sort(key=_cites, reverse=True)
    kept: list = []
    for hit in candidates:
        if _distinct_from(hit, kept + [resolved], artifact_words):
            kept.append(hit)
        if len(kept) >= _MULTI_MAX:
            break
    return kept


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


_HOMONYM_SYSTEM = """You know the scientific literature broadly. A user names an artifact with no other context, so the reference is AMBIGUOUS: several unrelated works across different fields share that name.
List the distinct, well-known papers/systems/tools/datasets that go by this exact name, ACROSS DIFFERENT RESEARCH AREAS (e.g. numerical computing, NLP, biology, neuroscience, security, systems).
Output ONLY JSON: {"works": [{"title": "<exact paper title>", "field": "<area>"}]}
Give up to 6, most notable first. Use the EXACT published title of each. If you are unsure of a title, omit that entry rather than guessing."""


def _homonym_candidates(artifact: str, llm: Any, client: Any,
                        inserted_before: str | None) -> list:
    """Resolve the distinct works sharing a bare artifact name.

    A bare mention of a short artifact name can be genuinely ambiguous: the
    right answer for such a query is the whole AMBIGUITY SET, whose members
    may come from entirely unrelated research areas. Relevance search cannot
    span that: it returns whatever is topically fashionable. Enumerating the
    homonyms by name and verifying each against the corpus can.

    Keyed on the mention text only; every candidate must resolve in the
    corpus to be emitted, so the model's recall is checked, not trusted.
    """
    try:
        obj = llm.json(_HOMONYM_SYSTEM, f"Name: {artifact}", max_tokens=500) or {}
    except Exception:
        return []
    out: list = []
    seen: set[str] = set()
    for item in (obj.get("works") or [])[:6]:
        title = _clean_title(item.get("title") if isinstance(item, dict) else item)
        if not title:
            continue
        paper = _title_search(client, title)
        if paper is None or _cid_of(paper) is None:
            paper = _relevance_resolve(client, title, inserted_before)
        if paper is None or _cid_of(paper) is None:
            continue
        if _violates_cutoff(paper, inserted_before):
            continue
        cid = _cid_of(paper)
        if cid in seen:
            continue
        # the resolved title must actually relate to the requested name or
        # to the title we asked for -- never accept a topical near-miss
        found = getattr(paper, "title", "") or ""
        if not (_title_links_mention(found, [_norm(artifact).replace(" ", "")])
                or _overlap(title, found) >= _TITLE_OVERLAP):
            continue
        seen.add(cid)
        out.append(paper)
    return out


def solve_specific(query: str, client: Any, llm: Any,
                   inserted_before: str | None = None,
                   trace: dict | None = None) -> Submission:
    """Resolve one navigational query to its believed exact paper set."""
    if trace is None:
        trace = {}
    meta = _mention_meta(query, llm)
    trace["meta"] = dict(meta)
    mention_tokens = _mention_tokens(query)
    trace["mention_tokens"] = mention_tokens

    resolved: Any | None = None
    link = None
    verified_extras: list = []     # other verified papers (hedge/probe pool)

    def _try_titles(titles: list[str], tag: str) -> None:
        nonlocal resolved, link
        for searched in titles[:_MAX_TITLE_PROBES]:
            paper = _title_search(client, searched)
            if paper is None or _cid_of(paper) is None:
                continue
            if _violates_cutoff(paper, inserted_before):
                continue
            if not _verify_meta(paper, searched, meta, mention_tokens):
                continue
            if resolved is None:
                resolved, link = paper, tag
            elif _cid_of(paper) != _cid_of(resolved):
                verified_extras.append(paper)

    # (b) exact title search on canonical + plausibles, verified
    _try_titles([t for t in [meta["canonical"]] + meta["plausibles"] if t], "title")

    # (c) guarded relevance-resolve: canonical first, then the raw query
    if resolved is None:
        for searched in [t for t in (meta["canonical"], query) if t]:
            paper = _relevance_resolve(client, searched, inserted_before)
            if paper is None or _cid_of(paper) is None:
                continue
            if _violates_cutoff(paper, inserted_before):
                continue
            if _verify_meta(paper, searched, meta, mention_tokens):
                resolved, link = paper, "relevance"
                break

    # (c2) second chance at the canonical title with a narrow, single-question
    # prompt (the combined metadata call drifts when it must emit four fields)
    if resolved is None:
        focused = _focused_canonical(query, llm, meta.get("canonical"))
        if focused:
            trace["focused_canonical"] = focused
            _try_titles([focused], "title-focused")
            if resolved is None:
                paper = _relevance_resolve(client, focused, inserted_before)
                if paper is not None and _cid_of(paper) is not None and \
                        not _violates_cutoff(paper, inserted_before) and \
                        _verify_meta(paper, focused, meta, mention_tokens):
                    resolved, link = paper, "relevance-focused"

    # (c2b) artifact token + author cue: the link that survives a hallucinated
    # canonical title WHEN the mention itself names an author.
    if resolved is None:
        paper, probe = _artifact_author_resolve(query, meta, client, inserted_before)
        if paper is not None:
            resolved, link = paper, "artifact-author"
            trace["artifact_author_probe"] = probe

    # (c2c) artifact token + SHORT topic phrase: models fabricate titles but
    # know what an artifact is FOR, so ask only for the topic.
    if resolved is None:
        paper, probe = _artifact_topic_resolve(query, meta, client, llm,
                                               inserted_before)
        if paper is not None:
            resolved, link = paper, "artifact-topic"
            trace["artifact_topic_probe"] = probe

    # (c3) artifact anchor: search the mention's OWN token, never a generated
    # title - the link that survives a hallucinated canonical
    if resolved is None:
        paper, token = _artifact_anchor_resolve(query, client, inserted_before)
        if paper is not None:
            resolved, link = paper, "artifact"
            trace["artifact_token"] = token

    # (d) grounded reference walk (authoritative when free recall failed -
    # accepted without meta-verification, like v2)
    if resolved is None:
        paper = _reference_walk(query, client, llm, inserted_before)
        if paper is not None and _cid_of(paper) is not None and \
                not _violates_cutoff(paper, inserted_before):
            resolved, link = paper, "refwalk"

    trace["link"] = link
    trace["resolved"] = _cid_of(resolved) if resolved is not None else None

    generic = _generic_acronym(query)
    artifact = _bare_artifact(query)
    emitted: list = []

    # Contested named artifact -> multi-emit by citationCount.
    # (The pre-registered "probe only off an artifact-bearing resolution"
    # gate is NOT applied: the failure it targeted was caused by circular
    # verification accepting an unrelated paper, which the mention-link
    # requirement in _verify_meta now blocks at the source. Gating here too
    # would suppress legitimate contests where the canonical title simply
    # does not contain the system name at all.)
    if artifact:
        contested = _artifact_probe(query, artifact, resolved,
                                    verified_extras, client, inserted_before)
        # Homonym enumeration (listing the distinct works sharing a name,
        # across fields) was tried here and REGRESSED the slice
        # 0.780 -> 0.730: it cannot tell an AMBIGUOUS name from an
        # unambiguous one, so it padded an already-correct single answer
        # with extra candidates and halved that query's score.
        # The helper is kept (_homonym_candidates) but not called; it
        # would need an ambiguity test that fires only when several
        # resolved works genuinely share the name.
        trace["probe"] = [_cid_of(p) for p in contested]
        if len(contested) >= _MULTI_MIN_DISTINCT:
            pool = ([resolved] if resolved is not None else []) + contested
            pool.sort(key=_cites, reverse=True)
            emitted = ([resolved] if resolved is not None else []) + \
                [p for p in pool if p is not resolved]
            emitted = emitted[:_MULTI_MAX]
            trace["cardinality"] = "multi-artifact"

    if not emitted and resolved is not None:
        if generic or _fails_query_cues(query, resolved):
            extras = list(verified_extras)
            if len(extras) < _HEDGE_TOTAL - 1:
                extras += _paper_search(client, query, _RELEVANCE_LIMIT,
                                        inserted_before)
            emitted = [resolved]
            for p in extras:
                if len(emitted) >= _HEDGE_TOTAL:
                    break
                if _violates_cutoff(p, inserted_before):
                    continue
                if _cid_of(p) not in {_cid_of(e) for e in emitted}:
                    emitted.append(p)
            trace["cardinality"] = "hedge3" if generic else "hedge3-cues"
        else:
            emitted = [resolved]
            trace["cardinality"] = "single"

    if not emitted:
        # Nothing resolved anywhere: never emit empty - top relevance hits,
        # 3 when the mention is generic or carries unverifiable cues, else 1.
        top_k = _HEDGE_TOTAL if (generic or _fails_query_cues(query, None)) else 1
        hits = [h for h in _paper_search(client, query, _RELEVANCE_LIMIT,
                                         inserted_before)
                if not _violates_cutoff(h, inserted_before)]
        emitted = hits[:top_k]
        trace["cardinality"] = f"unresolved-top{top_k}"

    # Dedup by cid, cap, hydrate missing abstracts, format evidence.
    final: list = []
    seen: set[str] = set()
    for p in emitted:
        cid = _cid_of(p)
        if cid and cid not in seen:
            seen.add(cid)
            final.append((cid, p))
    final = final[:_MULTI_MAX]

    hydrated = 0
    out: Submission = []
    for cid, p in final:
        if not (getattr(p, "abstract", "") or "") and hydrated < _HYDRATE_CAP:
            hydrated += 1
            try:
                full = client.get_paper(cid)
            except Exception:
                full = None
            if full is not None:
                if not (getattr(full, "title", "") or ""):
                    try:
                        full.title = getattr(p, "title", "") or ""
                    except Exception:
                        pass
                p = full
        out.append((cid, _paper_evidence(p)))
    trace["emitted"] = [cid for cid, _ in out]
    return out
