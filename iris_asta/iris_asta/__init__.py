"""IRIS-Asta: routed scientific-agent system for the ASTABench suite.

Core modules are stdlib-only; ``inspect_ai``/``astabench`` imports are guarded
so ``import iris_asta`` succeeds on a bare Python 3.11 machine (no pip deps,
no keys). Every sibling-module re-export below is guarded: modules that are
present import normally; missing or broken optional modules are silently
skipped and their names are simply absent from the package namespace.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__: list[str] = []

try:
    from iris_asta.contracts import (
        emit_adt,
        emit_discoverybench,
        emit_e2e,
        emit_litqa2,
        emit_pfb,
        emit_sqa,
        normalize_corpus_id,
        repair_sqa_sections,
        validate_single_json_object,
        validate_sqa_sections,
    )

    __all__ += [
        "emit_adt",
        "emit_discoverybench",
        "emit_e2e",
        "emit_litqa2",
        "emit_pfb",
        "emit_sqa",
        "normalize_corpus_id",
        "repair_sqa_sections",
        "validate_single_json_object",
        "validate_sqa_sections",
    ]
except Exception:  # pragma: no cover - contracts is stdlib-only and present
    pass

try:
    from iris_asta.config import Config, load_config

    __all__ += ["Config", "load_config"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.backbone import Backbone, ChatResult, extract_json

    __all__ += ["Backbone", "ChatResult", "extract_json"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.asta_client import AstaClient, Paper, Snippet

    __all__ += ["AstaClient", "Paper", "Snippet"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.asta_mcp import McpTransport, parse_sse_json

    __all__ += ["McpTransport", "parse_sse_json"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.solver_entry import iris_solver

    __all__ += ["iris_solver"]
except Exception:  # pragma: no cover - inspect_ai integration is optional
    pass

try:
    from iris_asta import solvers

    __all__ += ["solvers"]
except Exception:  # pragma: no cover - subpackage may not exist yet
    pass
