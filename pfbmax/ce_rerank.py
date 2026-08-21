"""Cross-encoder reranking of a semantic pool (free: local GPU sidecar).

Why this stage exists: IRIS measured the cross-encoder as the dominant
ranking channel behind its semantic doubling (0.121 -> 0.244), fused at ~3.3x
the weight of any retrieval channel.  The sidecar costs nothing per query
(owned/rented GPU, already running), so on the score-per-dollar objective it
is free accuracy: the one lever that improves the numerator without touching
the denominator.

Contract: wraps any SemPool (duck-typed: .ranked(), .evidence(cid),
.texts(cid), plus optional .title/.year accessors) and returns an object with
the same surface whose .ranked() is CE-reordered.  Soft-fails to the original
pool on any transport error, so a dropped tunnel degrades rather than breaks.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# Docs sent to the CE. This was the real bottleneck, and it is not a ranking
# problem at all -- measured on the full slice:
#     nDCG 0.705 | recall@K 0.1497 | recall@FULL 0.2145
# i.e. 69.8% of the golds we SUBMIT already reach the tiny top-K window, so
# ordering inside the submission is fine. But pool gold recall is 0.56-0.69
# while only 21% of golds reach the 250 submitted: the pool -> submission
# truncation throws away two thirds of what retrieval already found.
# With a cap of 400 over pools of 2000-4400, the cross-encoder never saw ~90%
# of the pool, and everything below kept raw fusion order. The CE runs on our
# own GPU (600 docs in ~0.6s), so extra depth here is negligible under the
# benchmark's cost accounting.
RERANK_CAP = int((os.environ.get("PFBMAX_CE_CAP") or "400").strip() or 400)
DOC_CHARS = 1400          # per-doc text budget
TIMEOUT_S = 180


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/rerank",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def rerank_scores(url: str, query: str, docs: list[dict],
                  timeout: float = TIMEOUT_S) -> dict[str, float]:
    """{id: score} from the sidecar; {} on any failure (caller keeps order)."""
    if not url or not docs:
        return {}
    try:
        body = _post(url, {"query": query, "documents": docs}, timeout)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return {}
    out: dict[str, float] = {}
    for row in body.get("scores") or []:
        if isinstance(row, dict) and "id" in row:
            try:
                out[str(row["id"])] = float(row["score"])
            except (TypeError, ValueError):
                continue
    return out


class RerankedPool:
    """SemPool passthrough with a CE-reordered .ranked()."""

    def __init__(self, pool, order: list[str], trace: dict | None = None):
        self._pool = pool
        self._order = order
        self.ce_trace = trace or {}

    def ranked(self) -> list[str]:
        return list(self._order)

    def __getattr__(self, name):        # evidence/texts/title/year/size/...
        return getattr(self._pool, name)


def ce_rerank_pool(query: str, criteria: list[str], pool, url: str | None = None,
                   cap: int = RERANK_CAP, trace: dict | None = None):
    """Reorder the pool's head by cross-encoder score, keeping the tail in
    fused order.  The CE query carries the criteria: PFB relevance is
    conjunctive (a paper must satisfy every criterion to earn the scorer's
    top label), so scoring against the bare query alone under-weights the
    constraints that decide the label."""
    url = url or os.environ.get("IRIS_ASTA_RERANKER_URL", "")
    tr = trace if trace is not None else {}
    order = list(pool.ranked())
    if not url or not order:
        tr["ce"] = {"status": "skipped-no-url" if not url else "empty-pool"}
        return pool

    head = order[:cap]
    docs = []
    for cid in head:
        try:
            text = pool.evidence(cid) or ""
        except Exception:
            text = ""
        title = ""
        for attr in ("title",):
            getter = getattr(pool, attr, None)
            if callable(getter):
                try:
                    title = getter(cid) or ""
                except Exception:
                    title = ""
        blob = (title + "\n" + text).strip()[:DOC_CHARS]
        if blob:
            docs.append({"id": cid, "text": blob})

    # Criteria ARE appended to the CE query by default (PFBMAX_CE_CRITERIA
    # defaults to on; set it to 0 to score against the bare query). The
    # evidence cuts both ways, so the history is worth recording. Against
    # appending: bge is trained on natural (query, passage) pairs, and gluing
    # four criteria onto the query makes a long unnatural string that dilutes
    # the signal. Measured on held-out labelled pairs (recall@K, K=2P), 14
    # queries:
    #   bare query 0.2710 | concat query+criteria 0.2404  (-0.0306)
    # Scoring each criterion separately and taking the MIN, the true
    # conjunction, is worse still (0.2045): the per-criterion scores are
    # noisy and a min over five of them just selects the worst noise.
    # Per-criterion MEAN (0.2518) also loses to the bare query. However,
    # dropping the criteria, after winning that pair-level test (+0.031),
    # LOST end-to-end (-0.0365, 3up/4dn, n=8). That was the third time a
    # pair-level win failed to transfer end-to-end, after the fine-tuned CE
    # (-0.083) and its distilled successor. So the default stays ON, and the
    # lesson stands: trust only end-to-end A/Bs.
    use_crit = os.environ.get("PFBMAX_CE_CRITERIA", "1").strip() in ("1", "true", "yes")
    ce_query = (query + " || " + " || ".join(criteria[:4])
                if (use_crit and criteria) else query)
    scores = rerank_scores(url, ce_query, docs)
    if not scores:
        tr["ce"] = {"status": "error", "sent": len(docs)}
        return pool

    # Optional second reranker, fused by reciprocal rank rather than replacing.
    # Measured motivation: the fine-tuned cross-encoder beats stock on 9 of 12
    # held-out queries (+0.0302 excluding its worst) but occasionally produces a
    # pathological ordering that takes a query from 0.2660 to 0.0000 outright.
    # Letting either model alone decide the head is what makes that failure
    # total; RRF lets one model's confident ranking survive the other's
    # collapse, so the blend keeps the upside without the cliff.
    url2 = os.environ.get("PFBMAX_RERANKER_URL_2", "").strip()
    fused_order = None
    if url2:
        scores2 = rerank_scores(url2, ce_query, docs)
        if scores2:
            # LEXICOGRAPHIC, not additive. The fine-tuned model suffers score
            # COLLAPSE -- measured sd 0.72 vs stock's 3.23, with only 29-47%
            # distinct values against stock's 91-98% -- so large groups tie and
            # their order inside a tie is arbitrary. Since K ~ 2P, the scored
            # window sits entirely inside that arbitrary zone.
            # So: the primary model sets a coarse band, the secondary orders
            # WITHIN the band. Averaging the two (RRF / z-sum) instead lets a
            # degenerate ordering corrupt the head of both, which is exactly
            # what it did live (0.1014 vs 0.1588 stock).
            # Selected offline on held-out labelled pairs (recall@K, K=2P):
            #   stock 0.2710 | rrf 0.2908 | z-sum 0.2858
            #   trained 0.3314 | lex band .25 0.3319 | lex band .10 0.3364
            band = float(os.environ.get("PFBMAX_CE_BAND", "0.10") or 0.10)
            fused_order = sorted(
                (c for c in head if c in scores or c in scores2),
                key=lambda c: (-round(scores.get(c, -1e9) / band),
                               -scores2.get(c, -1e9)))
            tr.setdefault("ce_fuse", {}).update(
                {"mode": "lex", "band": band, "n2": len(scores2)})

    if fused_order is not None:
        scored = fused_order
        unscored = [c for c in head if c not in set(fused_order)]
        new_order = scored + unscored + order[cap:]
        tr["ce"] = {"status": "ok-lex", "sent": len(docs), "scored": len(scored)}
        return RerankedPool(pool, new_order, tr["ce"])

    scored = [c for c in head if c in scores]
    unscored = [c for c in head if c not in scores]
    scored.sort(key=lambda c: -scores[c])
    new_order = scored + unscored + order[cap:]
    tr["ce"] = {"status": "ok", "sent": len(docs), "scored": len(scored),
                "moved_into_top50": len([c for c in scored[:50]
                                         if order.index(c) >= 50])}
    return RerankedPool(pool, new_order, tr["ce"])
