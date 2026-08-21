"""Deterministic ASTABench task routing for IRIS v2.

The public benchmark currently contains eleven scored cells but only nine
distinct solver interfaces: Paper Finder is also used for LitQA2 Search and
the E2E solver is used for both the ordinary and hard splits.  Routing is
metadata-first because task metadata is less likely to change than prose
templates; prompt fingerprints are retained for offline use and compatibility
with older harness versions.

This module is stdlib-only and deliberately contains no benchmark imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

TASK_PFB = "pfb"
TASK_SQA = "sqa"
TASK_ADT = "adt"
TASK_LITQA2 = "litqa2"
TASK_DISCOVERYBENCH = "discoverybench"
TASK_DS1000 = "ds1000"
TASK_CORE = "core"
TASK_SUPER = "super"
TASK_E2E = "e2e"
TASK_GENERIC = "generic"

TASKS = (
    TASK_PFB,
    TASK_SQA,
    TASK_ADT,
    TASK_LITQA2,
    TASK_DISCOVERYBENCH,
    TASK_DS1000,
    TASK_CORE,
    TASK_SUPER,
    TASK_E2E,
    TASK_GENERIC,
)


@dataclass(frozen=True)
class RouteSpec:
    """One task route and its operational policy."""

    task: str
    category: str
    specialist: str
    contract: str
    default_max_steps: int = 1


ROUTES: dict[str, RouteSpec] = {
    TASK_PFB: RouteSpec(TASK_PFB, "literature", "paper_finder", "pfb_json"),
    TASK_SQA: RouteSpec(TASK_SQA, "literature", "scholar_qa", "sqa_json", 3),
    TASK_ADT: RouteSpec(TASK_ADT, "literature", "table_synthesis", "adt_json"),
    TASK_LITQA2: RouteSpec(TASK_LITQA2, "literature", "litqa_ensemble", "mc_json", 3),
    TASK_DISCOVERYBENCH: RouteSpec(
        TASK_DISCOVERYBENCH, "data", "typed_discovery", "discovery_json", 6
    ),
    TASK_DS1000: RouteSpec(TASK_DS1000, "code", "code_repair", "code_tags", 3),
    TASK_CORE: RouteSpec(TASK_CORE, "code", "reproducibility_agent", "report_file", 18),
    TASK_SUPER: RouteSpec(TASK_SUPER, "code", "repository_agent", "answer_json", 18),
    TASK_E2E: RouteSpec(TASK_E2E, "discovery", "experiment_agent", "e2e_json", 22),
    TASK_GENERIC: RouteSpec(TASK_GENERIC, "other", "fallback", "free_text"),
}


_MC_LINE_RE = re.compile(r"^\s*([A-Z])[\)\].:]\s+(.+?)\s*$", re.M)


def _has_multiple_choice_block(text: str) -> bool:
    letters = [m.group(1) for m in _MC_LINE_RE.finditer(text or "")]
    if not letters or "A" not in letters:
        return False
    start = letters.index("A")
    return len(letters[start:]) >= 2 and letters[start + 1] == "B"


def detect_task(
    input_text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    has_choices: bool = False,
) -> str:
    """Return the IRIS route for an ASTABench sample.

    Metadata signatures mirror the fields constructed by ASTABench v1.0.0.
    They intentionally precede text checks so arbitrary E2E research prompts
    and SUPER repository instructions do not have to contain magic wording.
    """

    meta = metadata if isinstance(metadata, Mapping) else {}
    keys = set(meta)

    # Unique task metadata (strongest signals first).
    if {"results_dir", "dataset_name"} <= keys or (
        "structured_question" in keys and "results_dir" in keys
    ):
        return TASK_E2E
    if "pre_execute_cells" in keys or "landmarks" in keys or "entrypoint" in keys:
        return TASK_SUPER
    if "capsule_id" in keys and "results" in keys:
        return TASK_CORE
    if "raw_query" in keys:
        return TASK_PFB
    if "corpus_ids" in keys:
        return TASK_ADT
    if "initial_prompt" in keys and ("case_id" in keys or "annotator" in keys):
        return TASK_SQA
    if "metadata" in keys and (
        "query" in keys
        or isinstance(meta.get("metadata"), Mapping)
        and "datasets" in meta.get("metadata", {})
    ):
        return TASK_DISCOVERYBENCH

    text = input_text or ""
    lowered = text.lower()

    # Stable prompt markers from the public task implementations.
    if "task: codeocean_" in lowered or (
        "save your answers to a file named report.json" in lowered
        and "capsule" in lowered
    ):
        return TASK_CORE
    if (
        "write the remaining python code to append to the program above" in lowered
        or ("<code>" in lowered and "just write the new code" in lowered)
    ):
        return TASK_DS1000
    if "find papers relevant to the following query" in lowered:
        return TASK_PFB
    if "generate a report answering the following research question" in lowered:
        return TASK_SQA
    if "build a table that has each paper as a row" in lowered:
        return TASK_ADT
    if "make sure you load the dataset" in lowered:
        return TASK_DISCOVERYBENCH
    if has_choices or (
        "insufficient information" in lowered and _has_multiple_choice_block(text)
    ):
        return TASK_LITQA2

    # E2E prompts are intentionally open-ended. These markers are conservative
    # fallbacks; production routing normally uses the metadata branch above.
    if "scientific result" in lowered and any(
        marker in lowered
        for marker in ("conduct", "experiment", "evaluate", "investigate", "analyze")
    ):
        return TASK_E2E
    return TASK_GENERIC


def route_spec(task: str) -> RouteSpec:
    """Return a route policy, falling back safely for unknown task strings."""

    return ROUTES.get(task, ROUTES[TASK_GENERIC])


__all__ = [
    "ROUTES",
    "TASKS",
    "RouteSpec",
    "detect_task",
    "route_spec",
] + [name for name in globals() if name.startswith("TASK_")]
