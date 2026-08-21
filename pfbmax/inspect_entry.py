"""File-reference entry point that runs the PFB-MAX stack under the OFFICIAL
astabench harness (so submissions and live gpt-4o grading use the real path).

    inspect eval astabench/paper_finder_validation \
        --solver "<abs>/pfbmax/inspect_entry.py@pfbmax_solver" \
        --model openai/gpt-4o-mini

Why a file shim (same reasoning as iris_asta/inspect_solver.py): inspect
discovers solvers by AST-scanning for a LITERAL top-level @solver function,
and its loader execs the file without registering it in sys.modules — so this
module stays dataclass-free at top level and pulls the implementation in
through ordinary imports.

Corpus access: the harness hands the solver its own date-wrapped MCP tools,
but those are async and PFB is search-heavy (the IRIS project measured the
async tool bridge deadlocking on this task). We therefore use a direct
AstaClient like IRIS does, and enforce the sample's snapshot date ourselves
from state.metadata["insertion_date"] — the same contract the task tools
enforce, so results stay date-legal.
"""

import json
import os
import sys

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import solver

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_BUNDLE, "iris_asta")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_env() -> None:
    env_path = os.path.join(_BUNDLE, "iris_asta", ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def _extract_query(state) -> str:
    meta = getattr(state, "metadata", None) or {}
    raw = meta.get("raw_query")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    text = str(getattr(state, "input_text", "") or "")
    marker = "find papers relevant to the following query"
    low = text.lower()
    idx = low.find(marker)
    if idx >= 0:
        text = text[idx + len(marker):]
    return text.strip().strip(":").strip()


@solver
def pfbmax_solver(**kwargs):
    """PFB-MAX: compile -> route -> (metadata | specific | semantic) -> emit."""
    _load_env()
    # Fail fast on missing credentials. Every stage below soft-fails by design
    # (an exception must never cost a sample), which means a missing key would
    # otherwise produce a complete, hours-long, all-zero run with no diagnostic.
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it, or copy "
            "iris_asta/.env.example to iris_asta/.env and fill it in.")
    if not (os.environ.get("ASTA_TOOL_KEY") or "").strip():
        raise RuntimeError(
            "ASTA_TOOL_KEY is not set: corpus search would run against the "
            "anonymous rate-limit pool and score near zero. Request a free key "
            "and put it in iris_asta/.env (see iris_asta/.env.example).")
    os.environ.setdefault("IRIS_ASTA_MCP_DEADLINE_S", "15")
    os.environ.setdefault("IRIS_ASTA_TIMEOUT_S", "45")
    os.environ.setdefault("IRIS_ASTA_RATE_LIMIT_RPS", "2.0")

    async def solve(state, generate):
        import asyncio

        def _run():
            # inspect's loader execs this file without a stable __file__/sys.path
            # contract, so re-assert the import roots inside the worker.
            here = os.environ.get("PFBMAX_DIR") or _HERE
            bundle = os.path.dirname(here)
            for p in (here, os.path.join(bundle, "iris_asta")):
                if p not in sys.path:
                    sys.path.insert(0, p)

            from iris_asta.asta_client import AstaClient
            from iris_asta.config import load_config
            import router
            from costmeter import CostMeter
            from llm import LLM
            try:
                from corpus_cache import CachedClient
            except Exception:
                CachedClient = None

            meta = getattr(state, "metadata", None) or {}
            inserted_before = meta.get("insertion_date") or "2025-06-01"
            query = _extract_query(state)

            # PFBMAX_USE_MCP=1: force the Asta MCP gateway. Measured
            # 2026-08-18: api.semanticscholar.org hard-429s this key for
            # >24h while asta-tools.allen.ai/mcp answers in 0.8s.
            _mcp = bool((os.environ.get("PFBMAX_USE_MCP") or "").strip())
            if _mcp:
                # MCP truncates responses over ~400KB; giant citation pages
                # (limit 1000) always truncate -> exception -> bisect ->
                # truncate again, burning the whole 48-call budget at ~60s a
                # call. Smaller pages never truncate. In-process override
                # only; iris_asta itself stays untouched on disk.
                try:
                    import iris_asta.solvers.pfb as _pfb
                    _pfb._CITATION_LIMIT = min(
                        getattr(_pfb, "_CITATION_LIMIT", 1000), 200)
                except Exception:
                    pass
            client = AstaClient(load_config(),
                                use_mcp=True if _mcp else None)
            if CachedClient is not None:
                client = CachedClient(client)
            meter = CostMeter()
            llm = LLM(meter=meter)
            trace = {}
            try:
                results = router.solve(query, client, llm, inserted_before,
                                       trace=trace)
            except Exception as exc:                 # never fail the sample
                trace["fatal"] = repr(exc)[:300]
                results = []
            return results, meter.total_usd(), trace

        results, usd, trace = await asyncio.to_thread(_run)

        payload = {"output": {"results": [
            {"paper_id": str(pid), "markdown_evidence": ev}
            for pid, ev in results]}}
        completion = json.dumps(payload, ensure_ascii=True)

        state.output.completion = completion
        state.messages.append(ChatMessageAssistant(content=completion))
        try:
            state.metadata["pfbmax_usd"] = round(usd, 6)
            state.metadata["pfbmax_route"] = trace.get("route")
            state.metadata["pfbmax_n"] = len(results)
        except Exception:
            pass
        return state

    return solve
