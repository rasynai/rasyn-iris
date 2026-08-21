"""Runtime configuration for the IRIS-Asta solver stack.

Solver models sit behind one OpenAI-compatible gateway and can be routed per
task. The library supports an open-weights-only mode: with the default-on
``open_weight_only`` guard, every solver-side model must be on the verified
open-weight allowlist (or a local endpoint). The IRIS paper-finder
configuration uses hosted models, so those runs set
``IRIS_ASTA_OPEN_WEIGHT_ONLY=0`` to route through hosted endpoints.

Stdlib only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_ROUTED_TASKS = (
    "pfb",
    "sqa",
    "adt",
    "litqa2",
    "discoverybench",
    "ds1000",
    "core",
    "super",
    "e2e",
)

_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_MODEL = "qwen2.5-72b-instruct"
_DEFAULT_API_KEY = "EMPTY"  # vLLM ignores the key but the OpenAI schema requires one
_DEFAULT_USAGE_MODEL = "together_ai/Qwen/Qwen2.5-72B-Instruct"
_DEFAULT_VERIFIED_OPEN_MODELS = frozenset(
    {
        "qwen2.5-72b-instruct",
        "qwen2.5-14b-instruct",
        "qwen/qwen-2.5-72b-instruct",  # OpenRouter id for the same weights
        "deepseek/deepseek-v4-flash",
        "moonshotai/kimi-k2.5",
        "qwen/qwen3.5-122b-a10b",
        "z-ai/glm-5.2",
    }
)


@dataclass
class Config:
    """Frozen interface (docs/INTERFACES.md). All fields have defaults that
    match the env-var defaults so tests can construct ``Config(...)`` with
    only the fields they care about; ``load_config()`` is the env-driven
    constructor used in production."""

    llm_base_url: str = _DEFAULT_BASE_URL          # env IRIS_ASTA_LLM_BASE_URL
    llm_model: str = _DEFAULT_MODEL                # env IRIS_ASTA_LLM_MODEL
    # env IRIS_ASTA_LLM_CHEAP_MODEL (default = llm_model)
    llm_cheap_model: str = _DEFAULT_MODEL
    llm_api_key: str = _DEFAULT_API_KEY            # env IRIS_ASTA_LLM_API_KEY
    asta_tool_key: str = ""                        # env ASTA_TOOL_KEY
    usage_model_name: str = _DEFAULT_USAGE_MODEL   # env IRIS_ASTA_USAGE_MODEL
    rate_limit_rps: float = 2.0  # env IRIS_ASTA_RATE_LIMIT_RPS (MCP ceiling 10 req/s
    # SHARED across all concurrent samples - each PFB sample builds its own
    # uncoordinated limiter, so the per-sample default must stay <= 10 /
    # max_samples; the eval scripts run --max-samples 4 and sustained overrun
    # has hard-throttled the key server-side. 2.0 matches what
    # run_pfb_iris.sh already exports; raise via env for single-sample dev.)
    timeout_s: float = 45.0  # env IRIS_ASTA_TIMEOUT_S; fail a stuck call fast, not 2min
    # IRIS v2 routes can use different models behind the same OpenAI-compatible
    # gateway (vLLM for open-weight runs; LiteLLM or another gateway for mixed
    # frontier runs). Keys are the canonical task names from iris_asta.router.
    task_models: dict[str, str] = field(default_factory=dict)
    task_usage_models: dict[str, str] = field(default_factory=dict)
    task_max_steps: dict[str, int] = field(default_factory=dict)
    # Gateway-specific extra request-body fields (env
    # IRIS_ASTA_LLM_EXTRA_BODY, JSON object), e.g. OpenRouter provider
    # pinning. Merged into chat payloads with setdefault (never overrides
    # core fields).
    llm_extra_body: dict = field(default_factory=dict)
    # On a routing GATEWAY (OpenRouter), a 400/404/408/409 is frequently a
    # transient "no upstream provider could serve this right now" rather than a
    # malformed request - a retry lands on a different provider and succeeds.
    # Against a single local vLLM a 4xx is a real schema bug, so this is OFF by
    # default (preserving fail-fast semantics) and enabled only for gateway runs
    # via IRIS_ASTA_RETRY_GATEWAY_ERRORS=1.
    retry_gateway_errors: bool = False
    # The open-weight guard is ON by default: it restricts solver models to
    # the verified open-weight allowlist or a local endpoint. Runs that use
    # hosted models, including the IRIS paper-finder configuration, disable
    # it with IRIS_ASTA_OPEN_WEIGHT_ONLY=0.
    open_weight_only: bool = True
    verified_open_models: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_VERIFIED_OPEN_MODELS
    )

    def model_for_task(self, task: str, *, cheap: bool = False) -> str:
        """Select a task-specific model, preserving the old global fallback."""

        model = self.llm_cheap_model if cheap else self.task_models.get(
            task, self.llm_model
        )
        if self.open_weight_only and not self._is_verified_open_model(model):
            raise ValueError(
                f"model {model!r} is not in the verified open-weight allowlist"
            )
        return model

    def _is_verified_open_model(self, model: str) -> bool:
        """Accept verified hosted IDs or any model on a local endpoint.

        The loopback bypass is a documented trust assumption, not an
        oversight: any model served from 127.0.0.1/localhost passes the
        guard because self-hosting implies the weights are on disk.
        """

        base = self.llm_base_url.lower()
        local_endpoint = any(
            marker in base
            for marker in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]")
        )
        return local_endpoint or model.lower() in {
            value.lower() for value in self.verified_open_models
        }

    def usage_model_for_task(self, task: str) -> str:
        """Return the honest Inspect usage/pricing identifier for ``task``."""

        return self.task_usage_models.get(task, self.usage_model_name)

    def max_steps_for_task(self, task: str, default: int) -> int:
        """Return a positive route budget with a safe caller-provided default."""

        value = self.task_max_steps.get(task, default)
        return value if isinstance(value, int) and value > 0 else default


def load_config() -> Config:
    """Build a :class:`Config` from environment variables.

    Benchmark fact served: env-driven config keeps the solver runnable both
    offline (tests, no keys) and on the eval box (ASTA_TOOL_KEY for the
    ~4 req/s Semantic Scholar tools; IRIS_ASTA_LLM_* for the vLLM endpoint)
    without code changes - the frozen system runs the test split in one go.

    ``IRIS_ASTA_LLM_CHEAP_MODEL`` defaults to ``IRIS_ASTA_LLM_MODEL`` so a
    single-model deployment needs only one variable.
    """
    model = os.environ.get("IRIS_ASTA_LLM_MODEL", _DEFAULT_MODEL)
    task_models = {}
    task_usage_models = {}
    task_max_steps = {}
    for task in _ROUTED_TASKS:
        suffix = task.upper()
        if value := os.environ.get(f"IRIS_ASTA_MODEL_{suffix}"):
            task_models[task] = value
        if value := os.environ.get(f"IRIS_ASTA_USAGE_MODEL_{suffix}"):
            task_usage_models[task] = value
        if raw_steps := os.environ.get(f"IRIS_ASTA_MAX_STEPS_{suffix}"):
            try:
                steps = int(raw_steps)
            except (TypeError, ValueError):
                steps = 0
            if steps > 0:
                task_max_steps[task] = steps

    open_allowlist = set(_DEFAULT_VERIFIED_OPEN_MODELS)
    for value in os.environ.get("IRIS_ASTA_OPEN_MODEL_ALLOWLIST", "").split(","):
        if value.strip():
            open_allowlist.add(value.strip())

    return Config(
        llm_base_url=os.environ.get("IRIS_ASTA_LLM_BASE_URL", _DEFAULT_BASE_URL),
        llm_model=model,
        llm_cheap_model=os.environ.get("IRIS_ASTA_LLM_CHEAP_MODEL", model),
        llm_api_key=os.environ.get("IRIS_ASTA_LLM_API_KEY", _DEFAULT_API_KEY),
        asta_tool_key=os.environ.get("ASTA_TOOL_KEY", ""),
        usage_model_name=os.environ.get("IRIS_ASTA_USAGE_MODEL", _DEFAULT_USAGE_MODEL),
        rate_limit_rps=_env_float("IRIS_ASTA_RATE_LIMIT_RPS", 2.0),
        timeout_s=_env_float("IRIS_ASTA_TIMEOUT_S", 45.0),
        task_models=task_models,
        task_usage_models=task_usage_models,
        task_max_steps=task_max_steps,
        llm_extra_body=_env_json_obj("IRIS_ASTA_LLM_EXTRA_BODY"),
        retry_gateway_errors=_env_bool("IRIS_ASTA_RETRY_GATEWAY_ERRORS", False),
        open_weight_only=_env_bool("IRIS_ASTA_OPEN_WEIGHT_ONLY", True),
        verified_open_models=frozenset(open_allowlist),
    )


def _env_json_obj(name: str) -> dict:
    """JSON-object env var; soft-fails to {} (a typo must not crash init)."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return {}
    try:
        import json

        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _env_float(name: str, default: float) -> float:
    """Float env var with a soft fallback (a typo must not crash solver init)."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
