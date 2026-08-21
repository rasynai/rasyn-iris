"""Task solvers for iris_asta (PFB, LitQA2, ADT, DiscoveryBench, ...).

Guarded re-exports only: ``import iris_asta.solvers`` must succeed on a bare
Python 3.11 machine even when individual solver modules are absent or their
optional dependencies (``inspect_ai``/``astabench``) are not installed.
Missing or broken solver modules are silently skipped and their names are
simply absent from the subpackage namespace, so a failure in one solver can
never break ``import iris_asta``.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from iris_asta.solvers.pfb import (
        MetadataPlan,
        execute_plan,
        plan_metadata,
        route_query,
        semantic_channel,
        solve_pfb,
    )

    __all__ += [
        "MetadataPlan",
        "execute_plan",
        "plan_metadata",
        "route_query",
        "semantic_channel",
        "solve_pfb",
    ]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.solvers.sqa import solve_sqa

    __all__ += ["solve_sqa"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.solvers.code_execution import solve_ds1000, solve_workspace_task

    __all__ += ["solve_ds1000", "solve_workspace_task"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.solvers.e2e import solve_e2e

    __all__ += ["solve_e2e"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.solvers.adt import solve_adt

    __all__ += ["solve_adt"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.solvers.litqa2 import solve_litqa2

    __all__ += ["solve_litqa2"]
except Exception:  # pragma: no cover - module may not exist yet
    pass

try:
    from iris_asta.solvers.discoverybench import (
        AnalysisResult,
        compose_hypothesis,
        solve_discoverybench,
    )

    __all__ += ["AnalysisResult", "compose_hypothesis", "solve_discoverybench"]
except Exception:  # pragma: no cover - module may not exist yet
    pass
