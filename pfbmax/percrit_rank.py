"""Rank by a per-criterion student -- free, deep, grader-aligned filtering.

This is the capability every previous attempt lacked. The measured bottleneck
is that the scored window is tiny (K ~ 2P, e.g. 18 of 250 submitted) while the
golds sit deeper, so something must promote them into K. The criterion judge
can do it but costs ~$0.005/paper, which caps it at the top ~80. Everything
cheap enough to run deep -- RRF, the stock cross-encoder, a fine-tuned one --
ranks by something like an AVERAGE, and the grader's label is a CONJUNCTION.

The reason the cross-encoder could never be fixed is representational: it
scores one (query, passage) pair and has nowhere to put per-criterion
satisfaction. Fine-tuning it inverted live (-0.083) and distilling the judge
into it moved agreement only +0.168 -> +0.199.

So the student here predicts the right unit: (criterion, paper) -> one of the
grader's own three labels. Per-criterion probabilities combine as a SMOOTH
conjunction, product_i P(criterion_i perfect), which is the grader's rule
without the brittleness of a hard MIN (a minimum over ~5 noisy estimates just
selects the worst noise -- measured worst of all aggregators at 0.2045).

Because it runs locally it is negligible under the benchmark's cost accounting
(solver LLM tokens only), so unlike the judge it can score the whole pool.

Promote-only by default, which is the shape that has consistently worked here:
papers whose predicted conjunction clears a threshold move up, keeping their
prior order, and nothing is demoted -- so a wrong student cannot wreck an
ordering the cross-encoder already got right.

Env: PFBMAX_PERCRIT_URL (enables), PFBMAX_PERCRIT_DEPTH (default 400),
     PFBMAX_PERCRIT_MODE (promote|sort, default promote),
     PFBMAX_PERCRIT_THRESH (default 0.5), PFBMAX_PERCRIT_AGG (product|min|mean).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEPTH = 400
DOC_CHARS = 1400
TIMEOUT_S = 240.0


def _post(url: str, payload: dict, timeout: float = TIMEOUT_S) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/score",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def score_pool(url: str, criteria: list[str],
               docs: list[dict]) -> dict[str, dict]:
    """{id: {product,min,mean,per}}; {} on any failure (caller keeps order)."""
    if not url or not docs or not criteria:
        return {}
    try:
        body = _post(url, {"criteria": list(criteria)[:8], "documents": docs})
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return {}
    out = {}
    for row in body.get("scores") or []:
        if isinstance(row, dict) and "id" in row:
            out[str(row["id"])] = row
    return out


def rerank(query: str, criteria: list[str], docs: list[tuple[str, str]],
           url: str | None = None, trace: dict | None = None) -> list[str]:
    """Return cids reordered by the per-criterion student."""
    url = url or os.environ.get("PFBMAX_PERCRIT_URL", "")
    tr = trace if trace is not None else {}
    if not url or not docs or not criteria:
        return [c for c, _t in docs]

    d = (os.environ.get("PFBMAX_PERCRIT_DEPTH") or "").strip()
    depth = int(d) if d.isdigit() else DEPTH
    agg = (os.environ.get("PFBMAX_PERCRIT_AGG") or "product").strip()
    mode = (os.environ.get("PFBMAX_PERCRIT_MODE") or "promote").strip()
    try:
        thresh = float(os.environ.get("PFBMAX_PERCRIT_THRESH") or 0.5)
    except ValueError:
        thresh = 0.5

    head = docs[:depth]
    tail = [c for c, _t in docs[depth:]]
    payload = [{"id": c, "text": (t or "")[:DOC_CHARS]} for c, t in head if t]
    scored = score_pool(url, criteria, payload)
    if not scored:
        tr["percrit"] = {"status": "error", "sent": len(payload)}
        return [c for c, _t in docs]

    def val(cid):
        row = scored.get(cid)
        return float(row.get(agg, 0.0)) if row else 0.0

    order = [c for c, _t in head]
    if mode == "sort":
        new = sorted(order, key=lambda c: -val(c))
    else:
        hi = [c for c in order if val(c) >= thresh]
        rest = [c for c in order if val(c) < thresh]
        new = hi + rest
    tr["percrit"] = {"status": "ok", "sent": len(payload), "mode": mode,
                     "agg": agg, "promoted": sum(1 for c in order if val(c) >= thresh)}
    return new + tail


class _StudentPool:
    """SemPool passthrough whose .ranked() is student-reordered."""

    def __init__(self, pool, order):
        self._pool = pool
        self._order = order

    def ranked(self):
        return list(self._order)

    def __getattr__(self, name):
        return getattr(self._pool, name)


def rerank_pool(query, criteria, pool, url=None, trace=None):
    """Reorder the POOL (not the finished submission) with the student.

    This is where the measured loss actually is. On the full slice:
        nDCG 0.705 | recall@K 0.1497 | recall@FULL 0.2145
    69.8% of the golds we submit already reach the top-K window, so ordering
    inside the submission is fine -- but pool gold recall is 0.56-0.69 while
    only 21% of golds reach the 250 we submit. The submission is drawn from
    the fused order's head, so golds sitting deeper never get a chance.

    Reranking that depth with the cross-encoder made it strictly worse
    (recall@FULL 0.186 -> 0.109): the CE is a good local reranker of an
    already-good head and a bad global ranker, while RRF's multi-channel
    agreement is a strong deep prior. The student is the only free ranker
    aligned with the grader's own per-criterion decision, so it is the one
    candidate worth pointing at the depth.

    Promote-only: papers clearing the threshold move to the front keeping
    their fused order; nothing is demoted. A wrong student can then only fail
    to help, not destroy an ordering RRF got right.
    """
    url = url or os.environ.get("PFBMAX_PERCRIT_URL", "")
    tr = trace if trace is not None else {}
    order = list(pool.ranked())
    if not url or not order or not criteria:
        return pool

    d = (os.environ.get("PFBMAX_PERCRIT_POOL_DEPTH") or "").strip()
    depth = int(d) if d.isdigit() else 1500
    agg = (os.environ.get("PFBMAX_PERCRIT_AGG") or "product").strip()
    try:
        thresh = float(os.environ.get("PFBMAX_PERCRIT_THRESH") or 0.5)
    except ValueError:
        thresh = 0.5

    head = order[:depth]
    docs = []
    for cid in head:
        try:
            title = pool.title(cid) or ""
        except Exception:
            title = ""
        try:
            text = pool.evidence(cid) or ""
        except Exception:
            text = ""
        blob = (title + ". " + text).strip()
        if blob:
            docs.append({"id": cid, "text": blob[:DOC_CHARS]})
    scored = score_pool(url, criteria, docs)
    if not scored:
        tr["percrit_pool"] = {"status": "error", "sent": len(docs)}
        return pool

    def val(cid):
        row = scored.get(cid)
        return float(row.get(agg, 0.0)) if row else 0.0

    hi = [c for c in head if val(c) >= thresh]
    rest = [c for c in head if val(c) < thresh]
    tr["percrit_pool"] = {"status": "ok", "sent": len(docs),
                          "promoted": len(hi), "depth": depth}
    return _StudentPool(pool, hi + rest + order[depth:])
