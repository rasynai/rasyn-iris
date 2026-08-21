"""File-reference entry point for the inspect_ai harness (NOT stdlib-safe).

Use with the official asta-bench harness as:

    inspect eval astabench/<task> \
        --solver "<abs path>/iris_asta/iris_asta/inspect_solver.py@iris_solver"

Why this file exists (benchmark facts served):

* ``inspect eval --solver file.py@name`` discovers solvers by AST-scanning the
  referenced file for a LITERAL top-level ``@solver``-decorated function, then
  creates it via the registry. ``iris_asta.solver_entry`` cannot host that
  (its inspect_ai import must stay guarded so ``import iris_asta`` works on a
  bare Python 3.11 machine), so this shim carries the registration.
* inspect's ``load_module`` execs the referenced file WITHOUT inserting it in
  ``sys.modules``, which breaks ``@dataclass`` under ``from __future__ import
  annotations`` (PEP 563 ``_is_type`` looks up ``sys.modules[cls.__module__]``).
  This shim keeps its own top level dataclass-free and pulls the real
  implementation in through a normal package import, which registers
  ``iris_asta.solver_entry`` in ``sys.modules`` correctly.

This module is intentionally NOT imported by ``iris_asta/__init__`` and is
excluded from the offline-import contract: it requires inspect_ai.
"""

from inspect_ai.solver import solver

from iris_asta.solver_entry import _iris_solver_impl


@solver
def iris_solver(**kwargs):
    """IRIS ASTABench solver: fingerprint the task, run the IRIS solver stack.

    Recognized kwargs (all optional; used for testing/injection): ``backbone``,
    ``adapter``, ``cfg``. Unknown kwargs are ignored.
    """
    return _iris_solver_impl(**kwargs)
