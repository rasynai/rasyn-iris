"""Second retrieval round seeded by JUDGE-CONFIRMED papers.

Everything measured here says the remaining loss is in two places: papers we
never retrieve at all, and selection accuracy. Blind widening does not fix
the first -- a wider fanout raises pool recall (0.533 -> 0.644) yet measured
NEUTRAL-to-negative end to end, because the extra candidates dilute the
window the selector can afford to look at.

The difference here is WHERE we widen. After the pool judge runs, we know which
papers it scored as perfectly relevant. Their citation neighbourhood is not a
blind widening: a paper cited by, or citing, a confirmed-relevant paper is
drawn from a distribution far denser in golds than another BM25 page. So this
round expands only around confirmed hits, then judges the newcomers with the
same cheap selector before letting any of them into the submission.

That is what Ai2's own "diligent" mode buys with its long search budget, in a
shape that stays cheap here: corpus calls are inexpensive compared to LLM
calls, so the whole expansion costs wall-clock time plus one extra judge pass
over the NEW papers only.

Integrity: seeds come from the judge's scores on retrieved text, never from
gold ids; emitted evidence is the paper's own verbatim abstract.

Env: PFBMAX_DILIGENT=1 enables, PFBMAX_DILIGENT_SEEDS (default 8),
     PFBMAX_DILIGENT_PER_SEED (default 60), PFBMAX_DILIGENT_JUDGE (default 200).
"""

from __future__ import annotations

import os

DOC_CHARS = 1500


def _envi(name, default):
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _cid(paper):
    for attr in ("corpusId", "corpus_id"):
        v = getattr(paper, attr, None)
        if v:
            return str(v)
    return None


def expand(query, criteria, pool, client, inserted_before, judge,
           trace=None):
    """Return [(cid, evidence, score)] triples of NEW papers the judge rates
    highly.

    The score is included so the caller can merge by QUALITY. An
    earlier version appended newcomers to the TAIL: recall@FULL rose
    (0.1998 -> 0.2078, so the expansion really does find new golds) but the
    score FELL 0.0427, because tail papers cannot reach the top-K window while
    still joining the judged set and dragging nDCG down. Position is the whole
    game when the scored window is K ~ 2P.

    ``judge`` is criterion_judge; it is called only on the newcomers, so the
    marginal cost is one judge pass over ``PFBMAX_DILIGENT_JUDGE`` papers.
    """
    tr = trace if trace is not None else {}
    order = list(pool.ranked())
    if not order:
        return []
    n_seeds = _envi("PFBMAX_DILIGENT_SEEDS", 8)
    per_seed = _envi("PFBMAX_DILIGENT_PER_SEED", 60)
    n_judge = _envi("PFBMAX_DILIGENT_JUDGE", 200)

    seen = set(order)
    seeds = order[:n_seeds]          # pool is judge-sorted at this point
    fresh = {}
    for seed in seeds:
        for direction in ("references", "citations"):
            try:
                papers = client.get_citations(
                    seed, direction=direction, limit=per_seed,
                    fields="corpusId,title,abstract,year") or []
            except Exception:
                papers = []
            for p in papers:
                c = _cid(p)
                if not c or c in seen or c in fresh:
                    continue
                abstract = (getattr(p, "abstract", "") or "").strip()
                title = (getattr(p, "title", "") or "").strip()
                if not abstract:
                    continue          # no verbatim text -> cannot furnish evidence
                fresh[c] = (title, abstract)
    tr["diligent_fresh"] = len(fresh)
    if not fresh:
        return []

    items = list(fresh.items())[:n_judge]
    docs = [(c, (t + ". " + a)[:DOC_CHARS]) for c, (t, a) in items]
    # score each newcomer with the SAME replicated-grader scale the pool was
    # ranked on, so the caller can merge by quality instead of by arrival.
    # Parallel: 200 sequential judge calls is ~5 minutes a query, which makes
    # the round untestable regardless of whether it helps.
    from concurrent.futures import ThreadPoolExecutor
    model = os.environ.get("PFBMAX_CJ_MODEL", "gpt-4o-mini")
    workers = _envi("PFBMAX_CJ_WORKERS", 8)

    def _one(item):
        cid, text = item
        try:
            return cid, judge.score_paper(query, criteria, text, model)
        except Exception:
            return cid, None

    scored = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for cid, v in ex.map(_one, docs):
            if v is not None:
                scored[cid] = v
    lookup = dict(items)
    try:
        gate = float(os.environ.get("PFBMAX_DILIGENT_GATE") or 0.66)
    except ValueError:
        gate = 0.66
    keep = _envi("PFBMAX_DILIGENT_KEEP", 60)
    out = []
    for cid in sorted(scored, key=lambda c: -scored[c])[:keep]:
        if scored[cid] < gate:
            break   # below the grader's own "highly relevant" threshold: not worth a slot
        t, a = lookup.get(cid, ("", ""))
        ev = (a or t)[:1200]
        if ev:
            out.append((cid, ev, scored[cid]))
    tr["diligent_kept"] = len(out)
    tr["diligent_judged"] = len(scored)
    return out
