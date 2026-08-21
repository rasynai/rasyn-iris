"""Token/cost accounting for PFB-MAX.

Stdlib only.
Not thread-safe by design (the pipeline is single-threaded).

Usage:
    meter = CostMeter()
    meter.start_query("pfb_validation_12")
    meter.add("gpt-4o-mini", prompt_tokens=1200, completion_tokens=180)
    meter.end_query()
    meter.total_usd()      # running total across everything
    meter.per_query        # {qid: {"calls", "prompt_tokens", "completion_tokens", "usd"}}
    meter.report()         # JSON-serializable summary dict
"""
from __future__ import annotations

# USD per 1,000,000 tokens: model -> (input_price, output_price).
# gpt-4o-mini is the solver backbone; gpt-4o is listed for scorer-side
# (judge) cost estimates only.
# Standard (non-batch, non-cached) rates, verified against OpenAI's pricing
# page 2026-08-13. Standard rates over-report slightly when prompt caching
# kicks in, which is the safe direction for a cost claim.
# Reasoning models bill their hidden thinking as output tokens, so the output
# price is what dominates a listwise ranking run.
PRICING_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.5": (5.00, 30.00),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
}


def resolve_price(model):
    """Return (input_usd_per_mtok, output_usd_per_mtok) or None if unknown.

    Exact match first, then longest-prefix match so dated API model ids
    ("gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06") price correctly.
    """
    model = model or ""
    if model in PRICING_PER_MTOK:
        return PRICING_PER_MTOK[model]
    best_name = None
    for name in PRICING_PER_MTOK:
        if model.startswith(name) and (best_name is None or len(name) > len(best_name)):
            best_name = name
    return PRICING_PER_MTOK[best_name] if best_name else None


def cost_usd(model, prompt_tokens, completion_tokens):
    """USD cost of one call. Unknown models cost 0.0 (tokens still tracked)."""
    price = resolve_price(model)
    if price is None:
        return 0.0
    return (int(prompt_tokens) * price[0] + int(completion_tokens) * price[1]) / 1_000_000.0


def _bucket():
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}


def _bump(bucket, pt, ct, usd):
    bucket["calls"] += 1
    bucket["prompt_tokens"] += pt
    bucket["completion_tokens"] += ct
    bucket["usd"] += usd


class CostMeter:
    """Accumulates LLM usage, with per-model and per-query breakdowns."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._usd = 0.0
        self._per_model = {}
        self._per_query = {}
        self._unpriced = set()
        self._active_qid = None

    # -- recording ---------------------------------------------------------
    def add(self, model, prompt_tokens, completion_tokens, qid=None):
        """Record one call. Returns the USD cost of this call.

        Attribution: explicit ``qid`` wins; otherwise the query opened by
        the last ``start_query()`` (if any) is charged.
        """
        model = model or "unknown"
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        usd = cost_usd(model, pt, ct)
        if resolve_price(model) is None:
            self._unpriced.add(model)

        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self._usd += usd
        _bump(self._per_model.setdefault(model, _bucket()), pt, ct, usd)

        q = qid if qid is not None else self._active_qid
        if q is not None:
            _bump(self._per_query.setdefault(q, _bucket()), pt, ct, usd)
        return usd

    # -- per-query windows -------------------------------------------------
    def start_query(self, qid):
        """Begin attributing subsequent adds to ``qid``.

        Implicitly ends any query still open. Repeated qids accumulate.
        A snapshot exists even for queries that end with zero LLM calls.
        """
        if self._active_qid is not None:
            self.end_query()
        self._active_qid = qid
        self._per_query.setdefault(qid, _bucket())

    def end_query(self):
        """Stop attributing. Returns a copy of the finished query's snapshot
        (or None if no query was active)."""
        qid = self._active_qid
        self._active_qid = None
        if qid is None:
            return None
        return dict(self._per_query.get(qid, _bucket()))

    @property
    def per_query(self):
        """{qid: snapshot} — copies, safe to mutate."""
        return {q: dict(v) for q, v in self._per_query.items()}

    # -- reporting ---------------------------------------------------------
    def total_usd(self):
        return self._usd

    def report(self):
        """JSON-serializable summary of everything recorded so far."""
        n_queries = len(self._per_query)
        return {
            "total_usd": round(self._usd, 6),
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "queries": n_queries,
            "mean_usd_per_query": round(self._usd / n_queries, 6) if n_queries else 0.0,
            "per_model": {m: {**v, "usd": round(v["usd"], 6)}
                          for m, v in self._per_model.items()},
            "per_query": {q: {**v, "usd": round(v["usd"], 6)}
                          for q, v in self._per_query.items()},
            "unpriced_models": sorted(self._unpriced),
        }
