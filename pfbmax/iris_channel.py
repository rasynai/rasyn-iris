"""Use IRIS's proven semantic retrieval as a candidate source.

We rebuilt semantic retrieval from scratch in pfbmax and reached 0.2107 on
the slice. IRIS's own `semantic_channel` measured **0.2443** on the same 48
queries (its retrieval-v3, the result the whole predecessor project was built
around) and it has been sitting in the read-only tree unused this whole time.
Its fanout is far deeper than ours -- ~168 corpus calls per query with a
125-deep raw snippet budget, criterion-targeted evidence enrichment, and
citation-graph expansion -- which is exactly the pool-recall advantage we
could not reproduce by widening our own probes.

So: take IRIS's emitted order as the candidate pool, then apply our listwise
reranker (which IRIS never had) on top. The two are complementary -- IRIS
wins on recall, listwise wins on ordering, and the semantic metric is
harmonic(ordering, recall@K), so it needs both.

Backbone note: IRIS's Backbone enforces an open-weight allowlist at
construction (the predecessor project's thesis was open-weight-only). That
constraint is not ours, so we disable it and point the client at gpt-4o-mini,
which is both cheaper and stronger than the Qwen it was built for.

iris_asta stays READ-ONLY: this only imports and calls it.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_BUNDLE, "iris_asta")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _prepare_env() -> None:
    """Point IRIS's backbone at OpenAI and lift its open-weight guard."""
    os.environ["IRIS_ASTA_OPEN_WEIGHT_ONLY"] = "false"
    os.environ.setdefault("IRIS_ASTA_LLM_BASE_URL", "https://api.openai.com/v1")
    os.environ.setdefault("IRIS_ASTA_LLM_MODEL", "gpt-4o-mini")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        try:
            key = open(os.path.join(_HERE, ".openai_key"), encoding="utf-8").read().strip()
        except Exception:
            key = ""
    if key:
        os.environ["IRIS_ASTA_LLM_API_KEY"] = key
    # IRIS prices usage under a together_ai alias by default (it was serving
    # self-hosted Qwen); we are on OpenAI, so report the real model.
    os.environ["IRIS_ASTA_USAGE_MODEL"] = "openai/gpt-4o-mini"


def available() -> bool:
    try:
        _prepare_env()
        from iris_asta.solvers import pfb  # noqa: F401
        return True
    except Exception:
        return False


def retrieve(query: str, client, inserted_before: str | None = None,
             trace: dict | None = None) -> list[tuple[str, str]]:
    """Run IRIS's semantic channel; returns its (cid, evidence) order.

    Returns [] on any failure so the caller keeps its own pool.
    """
    tr = trace if trace is not None else {}
    try:
        _prepare_env()
        from iris_asta.backbone import Backbone
        from iris_asta.config import load_config
        from iris_asta.solvers import pfb
    except Exception as exc:
        tr["iris_channel"] = {"status": f"import-error:{type(exc).__name__}"}
        return []
    try:
        cfg = load_config()
        bb = Backbone(cfg, task="pfb")
        sub = pfb.semantic_channel(query, bb, client,
                                   inserted_before=inserted_before,
                                   ids_top_k=None)
        out = [(str(cid), ev) for cid, ev in (sub or []) if cid]
        tr["iris_channel"] = {"status": "ok", "n": len(out)}
        return out
    except Exception as exc:
        tr["iris_channel"] = {"status": f"error:{type(exc).__name__}:{str(exc)[:120]}"}
        return []
