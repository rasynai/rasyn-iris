"""End-to-end scientific experiment agent for IRIS v2."""

from __future__ import annotations

import json
from typing import Any, Callable

from iris_asta.backbone import extract_json
from iris_asta.contracts import emit_e2e


def _ask(bb: Any, messages: list[dict], max_tokens: int) -> dict:
    try:
        result = bb.chat(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return extract_json(getattr(result, "text", "")) or {}
    except Exception:
        return {}


def _ask_text(bb: Any, messages: list[dict], max_tokens: int) -> str:
    """Plain-text fallback for long final reports.

    Provider JSON modes can return an empty or truncated completion when a
    long report is embedded inside one JSON string.  The outer benchmark
    contract is assembled deterministically, so report synthesis itself does
    not need constrained decoding.
    """
    try:
        result = bb.chat(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=False,
        )
        return str(getattr(result, "text", "") or "").strip()
    except Exception:
        return ""


def _collect_artifacts(run_code: Callable[[str], str]) -> list[tuple[str, str, str]]:
    collector = r'''
import json
from pathlib import Path
items = []
patterns = ("iris_*", "results.json", "results.csv", "results.txt", "report.md")
seen = set()
for pattern in patterns:
    for p in Path(".").glob(pattern):
        if not p.is_file() or str(p) in seen:
            continue
        seen.add(str(p))
        try:
            if p.stat().st_size > 9000:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        items.append({
            "filename": str(p),
            "format": p.suffix.lstrip(".") or "text",
            "artifact": text,
        })
print(json.dumps({"iris_artifacts": items}))
'''
    try:
        raw = str(run_code(collector))
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start : end + 1]) if 0 <= start < end else {}
        items = obj.get("iris_artifacts", [])
    except Exception:
        items = []
    output = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        output.append(
            (
                str(item.get("filename", "artifact.txt")),
                str(item.get("format", "text")),
                str(item.get("artifact", "")),
            )
        )
    return output[:12]


def solve_e2e(
    prompt: str,
    metadata: dict,
    bb: Any,
    run_code: Callable[[str], str],
    *,
    max_steps: int = 14,
) -> str:
    """Execute a bounded experiment and return report, code, and artifacts."""

    system = """You are IRIS, an autonomous but skeptical research scientist in an isolated
benchmark sandbox. Complete the scientific task with real computation. Inspect available data
and software, translate the request into explicit measurable criteria, run appropriate baselines
and controls, quantify uncertainty where possible, and check for leakage or confounding. Never
fabricate a run or numerical result.

At each turn return one JSON object:
{"action":"execute","reason":"...","code":"Python for the stateful python_session"}
or
{"action":"final","report":"a self-contained scientific report grounded in observed results"}

Python may use pathlib, subprocess, installed scientific libraries, and shell commands through
subprocess. Save compact result tables or logs as iris_results.json/csv/txt so they can be included
as artifacts. Prefer reproducible scripts and deterministic seeds. Execute rather than merely
describing planned work.

Execution-budget policy:
- Spend at most THREE actions total on environment inspection, dependency installation, repository
  reading, or searching for data. After two failed attempts to obtain the ideal dataset, use the
  closest legally accessible subset/proxy and disclose that limitation.
- By action 4, write an end-to-end minimal experiment. Prioritize the task's scored deliverables:
  required baselines, proposed method, metrics, edge/error analysis, and saved artifacts.
- Prefer one coarse experiment that actually runs over an elaborate unfinished setup. Use small
  subsets, cached/precomputed features, lightweight models, and bounded subprocess timeouts.
- Reserve the final two actions for checking saved results and returning the report. Do not start
  a new download, dependency, or model training job in those actions.
- Never claim a metric that was not printed by executed code. A proxy experiment is acceptable only
  when it is clearly labeled and its exact data/features are reported.
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    code_cells: list[tuple[str, str]] = []
    observations: list[str] = []
    report = ""

    # One action is reserved for report synthesis after the execution loop.
    execution_steps = max(1, max_steps - 1)
    for step in range(execution_steps):
        remaining = execution_steps - step
        if step < 3:
            directive = (
                "Setup allowance is active, but converge on one viable data path now."
            )
        elif remaining > 2:
            directive = (
                "Setup allowance is exhausted. Implement or run the smallest complete "
                "experiment covering the requested baselines and method; do not keep searching."
            )
        else:
            directive = (
                "Final execution phase: inspect/save results and artifacts only. Do not begin "
                "new downloads, installs, or long training. Return action=final if evidence is ready."
            )
        budget_message = {
            "role": "user",
            "content": (
                f"BUDGET STATUS: action {step + 1} of {execution_steps}; "
                f"{remaining - 1} execution actions remain after this one. {directive}"
            ),
        }
        action = _ask(bb, messages[-14:] + [budget_message], 4200)
        kind = str(action.get("action", "")).strip().lower()
        if kind == "final" and isinstance(action.get("report"), str):
            report = action["report"].strip()
            if report:
                break
        code = action.get("code")
        if not isinstance(code, str) or not code.strip():
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return an executable next step, or a final "
                        "evidence-grounded report."
                    ),
                }
            )
            continue
        code_cells.append((f"analysis_step_{step + 1:02d}.py", code))
        try:
            observation = str(run_code(code))
        except Exception as exc:
            observation = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
        observations.append(observation[-12000:])
        messages.extend(
            [
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=True)},
                {"role": "user", "content": "PYTHON_SESSION RESULT:\n" + observation[-12000:]},
            ]
        )

    if not report:
        final = _ask(
            bb,
            messages[-14:]
            + [
                {
                    "role": "user",
                    "content": (
                        "The execution budget is exhausted. Return action=final with a rigorous "
                        "report using only observed evidence. Explicitly disclose "
                        "failed or missing analyses."
                    ),
                }
            ],
            5200,
        )
        report = str(final.get("report", "")).strip()
    if not report:
        report = _ask_text(
            bb,
            messages[-12:]
            + [
                {
                    "role": "user",
                    "content": (
                        "Write ONLY a concise, self-contained scientific report with Title, "
                        "Abstract, Introduction, Approach, Experiments, Results, Limitations, "
                        "Conclusion, and References. Use only results observed in the execution "
                        "trace. Explicitly say which requested metrics or baselines were not "
                        "completed. Do not wrap the report in JSON or a code fence."
                    ),
                }
            ],
            7000,
        )
    if not report:
        evidence_tail = "\n\n".join(_clip for _clip in observations[-3:])
        report = (
            "Execution-Limited Pilot Report\n\n"
            "Abstract\nThe requested investigation did not reach a completed metric-bearing "
            "experiment within the execution budget.\n\n"
            "Approach and observed execution\nThe agent performed the recorded setup and "
            "analysis actions. The final observed outputs were:\n"
            + evidence_tail[-6000:]
            + "\n\nResults and limitations\nNo unobserved numerical result is claimed. "
            "Any requested baseline, metric, or ablation absent from the trace remains "
            "incomplete.\n\nConclusion\nThe execution trace and code are retained for "
            "reproduction, but the scientific hypothesis is not supported or rejected by "
            "this incomplete run."
        )

    artifacts = _collect_artifacts(run_code)
    # Tool observations are also evidence artifacts, even when the experiment
    # did not create a standalone file.
    for index, observation in enumerate(observations[-6:], 1):
        artifacts.append((f"execution_{index:02d}.txt", "text", observation))
    trace = "\n".join(
        f"{filename}: executed; observation captured" for filename, _ in code_cells
    )
    return emit_e2e(report, code_cells, artifacts, trace=trace)


__all__ = ["solve_e2e"]
