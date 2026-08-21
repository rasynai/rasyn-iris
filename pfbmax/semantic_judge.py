"""Semantic-slice judge + emitter (public entry point: ``rank_and_emit``).

Two jobs, both keyed to how the OFFICIAL scorer works (gpt-4o reads ONLY our
markdown_evidence, per criterion; only >0.99 weighted criteria satisfaction
earns "perfect", the sole label counting toward recall@K):

1. ORDER the pool so scorer-perfect papers land inside the top-K window
   (K ~ 2P with P unknown; the top ~40 slots matter most). A batched
   gpt-4o-mini judge (same family as the scorer, so high agreement by
   construction) grades each of the top ``judge_batches*10`` fused
   candidates per criterion, and the emission is tiered: all-criteria-pass
   first (fused order), then >=75%-pass, then everything else in fused
   order. An all-reject judge degrades to pure fused order and NEVER
   empties the submission (IRIS's verified shape).

2. ATTACH evidence that DEMONSTRATES each criterion verbatim: the judge
   also selects, in the SAME call, the single best demonstrative sentence
   per paper; the quote is kept only when it is a verbatim substring of the
   paper's own provided texts (case/whitespace-normalized containment, with
   the ORIGINAL character span recovered so the emitted text is corpus
   text, never LLM text), else the paper's own best snippet stands in.
   Each emitted line is ``«Title» (year): quote | window | window`` with up
   to 2 extra distinct verbatim windows (criterion-targeted first), capped
   at 1500 chars. Papers without a resolvable title are dropped: untitled
   evidence violates the task's format rule and can never be judged
   perfect, so it would only burn a ranked slot.

Deep tails are free on the semantic slice (the nDCG math discounts them),
so k=250 papers are emitted whenever the pool has them. Batched judging
(10 papers/call, 2048 completion-token floor) is mandatory by design; unit
tests use fakes and never touch the network.
"""

from __future__ import annotations

import re
from typing import Any

Submission = list[tuple[str, str]]

__all__ = ["rank_and_emit", "JUDGE_MARKER", "JUDGE_SYSTEM"]

# --------------------------------------------------------------------------
# Tuning constants (values inherited from the IRIS solver where noted)
# --------------------------------------------------------------------------

#: Papers per judge call. Batched judging is mandatory by design;
#: single-paper judge calls are banned (cost).
JUDGE_BATCH_SIZE = 10

#: Completion-token floor per judge call (a fixed 2048-token floor; a
#: lower ceiling yields an empty completion, a failed parse, and the strict
#: all-zero fallback - silently reducing the solve to raw retrieval order).
JUDGE_MIN_OUTPUT_TOKENS = 2048

#: Per-paper completion allowance above the floor: ~30 tokens of verdict
#: JSON + up to ~120 tokens of copied quote.
JUDGE_TOKENS_PER_PAPER = 150

#: Near-pass promotion threshold (IRIS verified shape: for 1-3 criteria the
#: fraction is only reachable by passing all of them, so short criterion
#: lists keep exact fused order below the unanimous tier).
NEAR_PASS_FRACTION = 0.75

#: Hard cap on one emitted markdown_evidence string.
EVIDENCE_MAX_CHARS = 1500

#: One verbatim window's cap (IRIS multipart-excerpt part size).
WINDOW_CHARS = 400

#: Cap on the lead (judge-selected quote) inside the evidence line; an
#: over-long validated quote is re-windowed to a criteria-targeted span of
#: itself (still verbatim: a contiguous span of a contiguous span).
LEAD_MAX_CHARS = 700

#: Extra distinct verbatim windows after the lead quote.
MAX_EXTRA_WINDOWS = 2

#: Distinct per-paper texts considered for windows/validation (+ evidence).
MAX_TEXTS = 6

#: Criterion-targeted windows computed per paper (IRIS enrich cap + slack).
MAX_CRITERIA_WINDOWS = 6

#: Judge-prompt excerpt cap - sized to approximate what the official judge
#: will read (<=1500-char evidence line), so our verdicts calibrate on the
#: same proof surface the scorer sees.
JUDGE_EXCERPT_MAX_CHARS = 1400

#: Minimum normalized quote length accepted from the judge; shorter
#: fragments ("BERT") validate trivially but demonstrate nothing, so they
#: fall back to the paper's own best snippet.
MIN_QUOTE_CHARS = 20

#: Defensive title trim inside the evidence prefix.
TITLE_MAX_CHARS = 300

#: Inert marker embedded in the system prompt so offline tests (and the
#: pipeline assembly code) can route canned replies per stage without
#: depending on call order - same pattern as IRIS's ``[pfb:judge]``.
JUDGE_MARKER = "[pfbmax:judge]"

JUDGE_SYSTEM = f"""You are a STRICT relevance judge for scientific paper retrieval. {JUDGE_MARKER}
You are given relevance criteria and a NUMBERED list of papers (title +
verbatim excerpt). For EVERY paper do BOTH tasks:
1. FOR EACH criterion separately, decide whether the excerpt EXPLICITLY
demonstrates that the paper satisfies it. "Related", "implied", or
"probably" is NOT enough — mark true only when the excerpt itself is
sufficient proof; when in doubt, mark false.
2. Select the single verbatim sentence from the provided text that best
demonstrates the criteria. Copy it EXACTLY, character for character, from
that paper's own excerpt — never paraphrase, shorten, or stitch fragments
together.
Respond with JSON only:
{{"papers": [{{"idx": 1, "criteria": [true, false, ...], "quote": "<the single verbatim sentence from the provided text that best demonstrates the criteria>"}}, ...]}}
— exactly one object per paper ("idx" = the paper's number), SAME order and
SAME count as the numbered papers; each "criteria" array holds one boolean
per criterion, in the order the criteria are listed."""


# --------------------------------------------------------------------------
# Text utilities (faithful ports of the IRIS helpers)
# --------------------------------------------------------------------------

def _norm_text(s: str) -> str:
    """Lowercase and strip punctuation for fuzzy-but-deterministic matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def _norm_ws(s: str) -> str:
    """Case/whitespace normalization used for verbatim containment checks."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _term_set(terms: list[str]) -> set[str]:
    """Normalized content words (len >= 3) of the given terms."""
    return {w for t in terms for w in _norm_text(t).split() if len(w) >= 3}


def _minimal_snippet(text: str, terms: list[str], max_chars: int = WINDOW_CHARS) -> str:
    """Trim ``text`` to a contiguous verbatim window of <= ``max_chars``.

    Faithful port of IRIS ``_minimal_snippet`` (with an O(n) two-pointer +
    prefix-sum rewrite): trimming NEVER rewrites: the returned string is a
    contiguous character span of the original, chosen as the word-boundary
    window with the most query/criteria term hits (earliest window wins
    ties, deterministically). Evidence must be verbatim corpus text.
    """
    text = text or ""
    if len(text) <= max_chars:
        return text.strip()
    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return text[:max_chars].strip()
    term_set = _term_set(terms)
    hits = [1 if _norm_text(t.group()) in term_set else 0 for t in tokens]
    prefix = [0]
    for h in hits:
        prefix.append(prefix[-1] + h)
    best_score = -1
    best_span = (tokens[0].start(), min(tokens[0].start() + max_chars, len(text)))
    j = 0
    for i in range(len(tokens)):
        if j < i:
            j = i
        while j < len(tokens) and tokens[j].end() - tokens[i].start() <= max_chars:
            j += 1
        if j == i:  # single over-long token: take a raw slice
            span = (tokens[i].start(), min(tokens[i].start() + max_chars, len(text)))
            score = 0
        else:
            span = (tokens[i].start(), tokens[j - 1].end())
            score = prefix[j] - prefix[i]
        if score > best_score:
            best_score = score
            best_span = span
    return text[best_span[0]:best_span[1]].strip()


def _window_hits(window: str, terms: list[str]) -> int:
    """Count of a window's normalized tokens that hit the term set."""
    term_set = _term_set(terms)
    return sum(1 for w in _norm_text(window).split() if w in term_set)


def _is_dup(norm_candidate: str, norm_existing: list[str]) -> bool:
    """Containment dedupe: a window adding no new text is a duplicate."""
    return any(norm_candidate in p or p in norm_candidate for p in norm_existing)


# --------------------------------------------------------------------------
# Verbatim quote validation
# --------------------------------------------------------------------------

_QUOTE_WRAP_CHARS = "\"'“”‘’«»‹›`"


def _clean_quote(quote: Any) -> str:
    """Strip LLM wrapping (quote marks, ellipses) from a candidate quote."""
    if not isinstance(quote, str):
        return ""
    q = quote.strip().strip(_QUOTE_WRAP_CHARS).strip()
    changed = True
    while changed and q:
        changed = False
        for tok in ("...", "…"):
            if q.startswith(tok):
                q = q[len(tok):].lstrip()
                changed = True
            if q.endswith(tok):
                q = q[: -len(tok)].rstrip()
                changed = True
    return q.strip()


def _norm_map(source: str) -> tuple[str, list[int]]:
    """Case/whitespace-normalized copy of ``source`` + index map back.

    Each normalized character remembers the original index it came from, so
    a normalized-containment hit can be mapped back to the ORIGINAL span;
    the emitted evidence is corpus text with its original casing and
    whitespace, never the LLM's transcription of it.
    """
    chars: list[str] = []
    idx: list[int] = []
    pending_space = False
    for i, ch in enumerate(source):
        if ch.isspace():
            pending_space = bool(chars)  # leading whitespace drops
            continue
        if pending_space:
            chars.append(" ")
            idx.append(i)
            pending_space = False
        chars.append(ch.lower())
        idx.append(i)
    return "".join(chars), idx


def _find_verbatim(quote: str, sources: list[str]) -> str | None:
    """Original-text span of ``quote`` inside any source, or None.

    Containment is case/whitespace-normalized (the judge may lowercase or
    collapse spaces); anything further (paraphrase, punctuation drift, or
    a quote stitched across two snippets) fails and triggers the
    best-snippet fallback. A validated hit returns the source's OWN
    characters for the matched span (verbatim guarantee).
    """
    nq = _norm_ws(quote)
    if len(nq) < MIN_QUOTE_CHARS:
        return None
    for source in sources:
        if not source or not isinstance(source, str):
            continue
        norm, idx = _norm_map(source)
        pos = norm.find(nq)
        if pos < 0:
            continue
        start = idx[pos]
        end = idx[pos + len(nq) - 1] + 1
        return source[start:end]
    return None


# --------------------------------------------------------------------------
# Pool accessors (duck-typed; SemPool's contract surface is minimal)
# --------------------------------------------------------------------------

def _pool_texts(pool: Any, cid: str) -> list[str]:
    """The paper's provided texts: ``pool.texts(cid)`` + ``pool.evidence``.

    Every string here is verbatim corpus text per the SemPool contract, so
    the set doubles as both the window source and the quote-validation
    corpus. Defensive: a raising accessor degrades to fewer sources, never
    a crash (an unemitted contract scores 0).
    """
    texts: list[str] = []
    try:
        raw = pool.texts(cid) or []
    except Exception:
        raw = []
    for t in raw:
        if isinstance(t, str):
            t = t.strip()
            if t and t not in texts:
                texts.append(t)
        if len(texts) >= MAX_TEXTS:
            break
    try:
        ev = pool.evidence(cid) or ""
    except Exception:
        ev = ""
    if isinstance(ev, str):
        ev = ev.strip()
        frame = _EVIDENCE_TITLE_RE.match(ev)
        if frame:
            # An already-framed evidence line: the «Title» (year) frame is
            # METADATA (recovered separately by _pool_title_year), not paper
            # text - windows and quote validation must see corpus text only.
            ev = ev[frame.end():].lstrip(" :").strip()
        if ev and ev not in texts:
            texts.append(ev)
    return texts


def _int_year(value: Any) -> int | None:
    """Plausible publication year as int, else None (bools excluded)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 3000 else None
    if isinstance(value, str) and value.strip().isdigit():
        y = int(value.strip())
        return y if 1000 <= y <= 3000 else None
    return None


_EVIDENCE_TITLE_RE = re.compile(
    r"^\s*«(?P<title>[^»]{1,400})»\s*(?:\((?P<year>\d{4}|n\.d\.)\))?"
)
_TITLE_LINE_RE = re.compile(r"^\s*title\s*:\s*(?P<title>\S.*)$", re.IGNORECASE | re.MULTILINE)


def _pool_title_year(pool: Any, cid: str) -> tuple[str, int | None]:
    """Best-effort (title, year) for a cid across plausible pool surfaces.

    The pool contract pins only SemPool's minimum surface, but the evidence format
    rule needs a title. Probe ladder (each rung guarded): callable
    ``title(cid)``/``year(cid)``; a record from callable ``meta``/``paper``/
    ``record``/``get``; mapping attributes ``metas``/``meta``/``papers``/
    ``records``/``titles``/``years``; finally a ``«Title» (year):`` or
    ``title: ...`` frame parsed off ``evidence(cid)``. Papers that resolve
    no title anywhere are dropped by the emitter (format rule).
    """
    title: str = ""
    year: int | None = None
    for name in ("title", "get_title"):
        fn = getattr(pool, name, None)
        if callable(fn):
            try:
                t = fn(cid)
            except Exception:
                t = None
            if isinstance(t, str) and t.strip():
                title = t.strip()
                break
    for name in ("year", "get_year"):
        fn = getattr(pool, name, None)
        if callable(fn):
            try:
                y = _int_year(fn(cid))
            except Exception:
                y = None
            if y is not None:
                year = y
                break
    if title and year is not None:
        return title, year
    rec: Any = None
    for name in ("meta", "paper", "record", "get"):
        fn = getattr(pool, name, None)
        if callable(fn):
            try:
                r = fn(cid)
            except Exception:
                r = None
            if r is not None:
                rec = r
                break
    if rec is None:
        for name in ("metas", "meta", "papers", "records"):
            mapping = getattr(pool, name, None)
            if isinstance(mapping, dict) and cid in mapping:
                rec = mapping.get(cid)
                break
    if rec is not None:
        if not title:
            t = rec.get("title") if isinstance(rec, dict) else getattr(rec, "title", None)
            if isinstance(t, str) and t.strip():
                title = t.strip()
        if year is None:
            y = rec.get("year") if isinstance(rec, dict) else getattr(rec, "year", None)
            year = _int_year(y)
    if not title:
        tmap = getattr(pool, "titles", None)
        if isinstance(tmap, dict):
            t = tmap.get(cid)
            if isinstance(t, str) and t.strip():
                title = t.strip()
    if year is None:
        ymap = getattr(pool, "years", None)
        if isinstance(ymap, dict):
            year = _int_year(ymap.get(cid))
    if not title or year is None:
        try:
            ev = pool.evidence(cid) or ""
        except Exception:
            ev = ""
        if isinstance(ev, str) and ev:
            m = _EVIDENCE_TITLE_RE.match(ev)
            if m:
                if not title:
                    title = m.group("title").strip()
                if year is None:
                    year = _int_year(m.group("year") or "")
            if not title:
                m2 = _TITLE_LINE_RE.search(ev)
                if m2:
                    title = m2.group("title").strip()
    return title[:TITLE_MAX_CHARS].strip(), year


# --------------------------------------------------------------------------
# Windows (shared by the judge prompt and the emitted evidence)
# --------------------------------------------------------------------------

def _windows(texts: list[str], criteria: list[str], query: str) -> tuple[list[str], list[str]]:
    """(general, criterion-targeted) verbatim windows for one paper.

    General windows: the top distinct texts trimmed to their best
    query+criteria window (IRIS multipart lead, the shape that wins
    scorer accepts). Criterion windows: for each criterion, the single
    text window with the most hits on THAT criterion's terms (IRIS enrich
    forensics: a criterion the best snippet lacks is often proven by
    another snippet). All windows are contiguous verbatim spans.
    """
    terms = list(criteria) + ([query] if query else [])
    general: list[str] = []
    for t in texts[:3]:
        w = _minimal_snippet(t, terms, WINDOW_CHARS)
        if w and w not in general:
            general.append(w)
    crit: list[str] = []
    for criterion in criteria[:MAX_CRITERIA_WINDOWS]:
        if not _term_set([criterion]):
            continue
        best, best_hits = "", 0
        for t in texts:
            w = _minimal_snippet(t, [criterion], WINDOW_CHARS)
            h = _window_hits(w, [criterion])
            if h > best_hits:
                best, best_hits = w, h
        if best:
            crit.append(best)
    return general, crit


def _judge_excerpt(general: list[str], crit: list[str]) -> str:
    """Join distinct windows into the excerpt the internal judge reads.

    General lead first, criterion-targeted appends after (IRIS's additive
    ordering; replacing the lead with targeted windows measurably flipped
    scorer accepts to rejects), containment-deduped, capped near the
    official evidence budget so the verdicts calibrate on a proof surface
    equivalent to what the scorer's judge will read.
    """
    parts: list[str] = []
    norms: list[str] = []
    total = 0
    for w in list(general) + list(crit):
        w = (w or "").strip()
        if not w:
            continue
        nw = _norm_ws(w)
        if not nw or _is_dup(nw, norms):
            continue
        cost = len(w) + (3 if parts else 0)
        if total + cost > JUDGE_EXCERPT_MAX_CHARS:
            continue
        parts.append(w)
        norms.append(nw)
        total += cost
    return " … ".join(parts)


# --------------------------------------------------------------------------
# Batched judge
# --------------------------------------------------------------------------

def _truthy(value: Any) -> bool:
    """True for JSON true or a "true"/"yes" string (LLM output tolerance)."""
    return value is True or (
        isinstance(value, str) and value.strip().lower() in ("true", "yes")
    )


def _pass_count(verdict: Any, n_criteria: int) -> int:
    """Criteria-passed count from one per-paper verdict (IRIS port).

    A bare true means "all criteria proven" (n); a boolean array counts its
    truthy entries; anything unparseable is 0, so a malformed reply can only
    demote, never promote.
    """
    if _truthy(verdict):
        return n_criteria
    if isinstance(verdict, list):
        return sum(1 for v in verdict[:n_criteria] if _truthy(v))
    return 0


def _judge_batch(
    llm: Any, criteria: list[str], items: list[tuple[str, str]]
) -> tuple[list[int], list[str]]:
    """One batched judge call: per-criterion verdicts + best quote per paper.

    ``items`` is [(title, excerpt)] for <= JUDGE_BATCH_SIZE papers. Returns
    (pass_counts, raw_quotes) aligned with ``items``. Entries are aligned
    by their self-reported 1-based "idx" when valid, positionally
    otherwise; a missing/short/unparseable reply judges every uncovered
    paper 0 with no quote (strict, demote-only). max_tokens follows a
    fixed formula: max(2048, 150 * batch).
    """
    n_criteria = len(criteria)
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
    listing = "\n\n".join(
        f"[{i + 1}] title: {(t or '(unknown)')[:TITLE_MAX_CHARS]}\n"
        f"excerpt: {e or '(no excerpt available)'}"
        for i, (t, e) in enumerate(items)
    )
    user = (
        f"Criteria ({n_criteria}; judge each SEPARATELY for every paper):\n"
        f"{numbered}\n\n"
        f"Papers ({len(items)}):\n{listing}"
    )
    max_tokens = max(JUDGE_MIN_OUTPUT_TOKENS, JUDGE_TOKENS_PER_PAPER * len(items))
    try:
        obj = llm.json(JUDGE_SYSTEM, user, max_tokens=max_tokens)
    except Exception:
        obj = None
    counts = [0] * len(items)
    quotes = [""] * len(items)
    entries = obj.get("papers") if isinstance(obj, dict) else None
    if not isinstance(entries, list):
        return counts, quotes
    filled = [False] * len(items)
    for pos, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        idx = entry.get("idx")
        if isinstance(idx, bool):
            idx = None
        if isinstance(idx, str) and idx.strip().isdigit():
            idx = int(idx.strip())
        slot = idx - 1 if isinstance(idx, int) and 1 <= idx <= len(items) else pos
        if slot >= len(items) or filled[slot]:
            continue
        counts[slot] = _pass_count(entry.get("criteria"), n_criteria)
        quote = entry.get("quote")
        quotes[slot] = quote if isinstance(quote, str) else ""
        filled[slot] = True
    return counts, quotes


# --------------------------------------------------------------------------
# Evidence assembly
# --------------------------------------------------------------------------

def _compose_evidence(
    title: str,
    year: int | None,
    lead: str,
    candidates: list[str],
    terms: list[str],
) -> str:
    """``«Title» (year): lead | window | window`` under the 1500-char cap.

    The lead is the validated judge quote (or the paper's own best snippet);
    up to MAX_EXTRA_WINDOWS additional distinct verbatim windows follow,
    ``candidates`` order preserved (criterion-targeted first). Containment
    dedupe keeps every appended window informative; the budget check keeps
    the whole line inside EVIDENCE_MAX_CHARS. Unknown years print "n.d."
    so the leading title/year frame the task demands is always present.
    """
    year_str = str(year) if isinstance(year, int) and not isinstance(year, bool) else "n.d."
    prefix = f"«{title}» ({year_str}): "
    budget = EVIDENCE_MAX_CHARS - len(prefix)
    lead = (lead or "").strip()
    cap = min(LEAD_MAX_CHARS, max(0, budget))
    if len(lead) > cap:
        lead = _minimal_snippet(lead, terms, cap)
    parts: list[str] = [lead] if lead else []
    norms: list[str] = [_norm_ws(lead)] if lead else []
    used = len(prefix) + len(lead)
    added = 0
    for w in candidates:
        if added >= MAX_EXTRA_WINDOWS:
            break
        w = (w or "").strip()
        if not w:
            continue
        nw = _norm_ws(w)
        if not nw or _is_dup(nw, norms):
            continue
        cost = len(w) + (3 if parts else 0)
        if used + cost > EVIDENCE_MAX_CHARS:
            continue
        parts.append(w)
        norms.append(nw)
        used += cost
        added += 1
    return (prefix + " | ".join(parts)).strip()


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def rank_and_emit(
    query: str,
    criteria: list[str],
    pool: "SemPool",
    llm: "LLM",
    judge_batches: int = 12,
    k: int = 250,
) -> Submission:
    """Judge the fused pool head, tier-order it, and emit evidence lines.

    Pipeline:
      1. Judge the top ``judge_batches * 10`` cids of ``pool.ranked()`` in
         batches of 10 (one ``llm.json`` call each; max_tokens =
         max(2048, 150*batch)), collecting per-criterion booleans AND the
         best demonstrative quote per paper in the same call.
      2. Validate each quote by case/whitespace-normalized containment in
         the paper's own provided texts, recovering the original span;
         invalid quotes fall back to the paper's best snippet.
      3. Order: all-criteria-pass tier (fused order), then >=75%-pass tier,
         then everything else in fused order. All-reject or failed judging
         degrades to pure fused order, so the submission never empties.
      4. Emit up to ``k`` (default 250; deep tails are free on this slice)
         ``(cid, "«Title» (year): quote | window | window")`` tuples,
         <=1500 chars each, extras criterion-targeted first; papers whose
         title cannot be resolved are dropped (format rule).

    Degradation ladder: no criteria / no llm / zero batches -> no LLM calls,
    pure fused emission with best-snippet evidence. Cost profile at the
    defaults: <=12 gpt-4o-mini calls (~4k prompt + <=2k completion tokens
    each), about $0.01/query, inside the <=$0.03/query cost budget.
    """
    criteria = [c.strip() for c in (criteria or []) if isinstance(c, str) and c.strip()]
    try:
        raw_ranked = pool.ranked() or []
    except Exception:
        raw_ranked = []
    ranked: list[str] = []
    seen: set[str] = set()
    for cid in raw_ranked:
        c = str(cid).strip()
        if c and c not in seen:
            seen.add(c)
            ranked.append(c)
    if not ranked:
        return []

    try:
        # Judge depth is the recall@K lever. Decomposition over live
        # queries: rank 0.668 but recall@K 0.083, while ~43% of golds
        # are already IN the pool -- they sit below the judged window,
        # so nothing can promote them. Deeper judging is the direct fix
        # and the only cost is gpt-4o-mini tokens (the per-query cost
        # budget has ample headroom, so there is room to spend here).
        import os as _os
        _env = _os.environ.get("PFBMAX_JUDGE_BATCHES", "").strip()
        if _env.isdigit():
            judge_batches = int(_env)
        n_batches = max(0, int(judge_batches))
    except (TypeError, ValueError):
        n_batches = 0
    try:
        k_cap = max(0, int(k))
    except (TypeError, ValueError):
        k_cap = 250
    head = ranked[: n_batches * JUDGE_BATCH_SIZE]
    terms = criteria + ([query] if query else [])

    sources: dict[str, list[str]] = {}
    windows: dict[str, tuple[list[str], list[str]]] = {}
    meta: dict[str, tuple[str, int | None]] = {}

    def _prep(cid: str) -> tuple[list[str], tuple[list[str], list[str]]]:
        if cid not in sources:
            sources[cid] = _pool_texts(pool, cid)
            windows[cid] = _windows(sources[cid], criteria, query)
        return sources[cid], windows[cid]

    def _meta(cid: str) -> tuple[str, int | None]:
        if cid not in meta:
            meta[cid] = _pool_title_year(pool, cid)
        return meta[cid]

    # -- 1+2) batched judge + quote validation ------------------------------
    counts: dict[str, int] = {}
    leads: dict[str, str] = {}
    if criteria and head and llm is not None:
        for start in range(0, len(head), JUDGE_BATCH_SIZE):
            chunk = head[start : start + JUDGE_BATCH_SIZE]
            items = []
            for cid in chunk:
                _texts, (general, crit_wins) = _prep(cid)
                items.append((_meta(cid)[0], _judge_excerpt(general, crit_wins)))
            chunk_counts, chunk_quotes = _judge_batch(llm, criteria, items)
            for cid, count, quote in zip(chunk, chunk_counts, chunk_quotes):
                counts[cid] = count
                span = _find_verbatim(_clean_quote(quote), sources[cid]) if quote else None
                if span:
                    leads[cid] = span

    # -- 3) tier ordering (IRIS verified shape) -----------------------------
    n_criteria = len(criteria)
    tier_full = [
        cid for cid in head if n_criteria and counts.get(cid, 0) >= n_criteria
    ]
    tier_near = [
        cid
        for cid in head
        if n_criteria
        and counts.get(cid, 0) < n_criteria
        and counts.get(cid, 0) / n_criteria >= NEAR_PASS_FRACTION
    ]
    promoted = set(tier_full) | set(tier_near)
    ordered = tier_full + tier_near + [cid for cid in ranked if cid not in promoted]

    # -- 4) emission --------------------------------------------------------
    out: Submission = []
    for cid in ordered:
        if len(out) >= k_cap:
            break
        title, year = _meta(cid)
        if not title:
            continue  # untitled evidence cannot satisfy the format rule
        _texts, (general, crit_wins) = _prep(cid)
        lead = leads.get(cid, "")
        if not lead:
            lead = general[0] if general else (crit_wins[0] if crit_wins else "")
        out.append(
            (cid, _compose_evidence(title, year, lead, crit_wins + general, terms))
        )
    return out
