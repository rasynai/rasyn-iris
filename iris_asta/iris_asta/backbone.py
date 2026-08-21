"""OpenAI-compatible chat backbone for routed hosted or self-hosted models.

Benchmark facts served:

* Every solver-side LLM call goes through :class:`Backbone` so that token
  usage is recorded via ``astabench.util.model.record_model_usage_with_inspect``
  under ``cfg.usage_model_name``. Hosted model IDs are recorded directly;
  self-hosted runs can use a ``together_ai/*`` proxy because litellm cannot
  otherwise price them.
* ``json_mode=True`` sends ``response_format={"type": "json_object"}`` —
  parse failures score 0 on SQA/PFB/ADT/E2E/DiscoveryBench, so structured
  stages must constrain decoding at the server.
* Retries with exponential backoff on HTTP 429/5xx, connection errors, and
  empty completions — vLLM under load or during warmup must not turn into a
  scored-0 sample.

Stdlib only; ``astabench``/``inspect_ai`` imports are guarded inside
``_record_usage`` so this module imports cleanly on a bare machine.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Config

# Base for exponential retry backoff (seconds): 0.5, 1, 2, ... capped below.
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 30.0


@dataclass
class ChatResult:
    """One chat-completion outcome: text plus token counts for usage pricing."""

    text: str
    prompt_tokens: int
    completion_tokens: int


#: Gateway 4xx codes that are transient on a routing proxy (OpenRouter): a
#: retry re-routes to another upstream provider. Meaningless against a single
#: local vLLM, where the same codes are real schema bugs - so only honored when
#: ``retry_gateway`` is set (cfg.retry_gateway_errors, gateway runs only).
_GATEWAY_RETRY_CODES = frozenset({400, 404, 408, 409})


def _is_retryable(exc: Exception, retry_gateway: bool = False) -> bool:
    """True for transient failures worth a backoff-retry.

    Benchmark fact served: HTTP 429 (rate limit) and 5xx (vLLM overload /
    proxy hiccup) are transient — abandoning the call would forfeit the
    sample's score. Non-HTTP ``URLError`` (connection refused/reset during
    server warmup) is retried for the same reason. Against a single local
    vLLM a 4xx is a real schema defect and propagates immediately; on a
    routing gateway (OpenRouter) 400/404/408/409 are frequently transient
    provider-availability errors, retried only when ``retry_gateway`` is set.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code == 429 or 500 <= code <= 599:
            return True
        if retry_gateway and code in _GATEWAY_RETRY_CODES:
            return True
        return False
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))


class Backbone:
    """Chat client for an OpenAI-schema ``/chat/completions`` endpoint.

    ``transport`` (test injection) is a callable ``transport(payload: dict)
    -> response: dict`` replacing the real HTTP POST; it may raise
    ``urllib.error.HTTPError`` to simulate HTTP failures. ``cheap=True``
    selects ``cfg.llm_cheap_model`` (the 14B stage model) instead of
    ``cfg.llm_model``.
    """

    def __init__(
        self,
        cfg: Config,
        cheap: bool = False,
        transport=None,
        *,
        task: str = "generic",
    ):
        self.cfg = cfg
        self.task = task
        self.model = cfg.model_for_task(task, cheap=cheap)
        self.usage_model_name = (
            cfg.usage_model_name if cheap else cfg.usage_model_for_task(task)
        )
        self._transport = transport if transport is not None else self._http_transport
        self._sleep = time.sleep  # test hook: replace to avoid real backoff sleeps

    def _http_transport(self, payload: dict) -> dict:
        """Default transport: POST ``{base_url}/chat/completions`` via urllib.

        Benchmark fact served: the solver must run with zero pip deps, so raw
        ``urllib.request`` speaks the OpenAI chat-completions schema to vLLM.
        """
        url = self.cfg.llm_base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.llm_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Fold the response body into the error so a gateway's reason
            # ("no allowed provider", quota, etc.) reaches the trace instead of
            # a bare "HTTP Error 400". Re-raise an HTTPError preserving .code so
            # the retry classifier still sees the status.
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            reason = f"{exc.reason} | {body}" if body else exc.reason
            raise urllib.error.HTTPError(
                exc.url, exc.code, reason, exc.headers, None
            ) from None

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
        retries: int = 3,
    ) -> ChatResult:
        """Run one chat completion with retries; record usage on every success.

        Benchmark facts served: ``json_mode`` maps to
        ``response_format={"type": "json_object"}`` (contract-valid JSON or the
        sample scores 0); ``retries`` extra attempts with exponential backoff
        cover HTTP 429/5xx, connection errors, and empty completions; after
        each successful HTTP round-trip ``_record_usage`` logs prompt/completion
        tokens so both hosted and self-hosted routes get honest cost records.

        If every attempt succeeds at the HTTP level but yields empty text, the
        last (empty) :class:`ChatResult` is returned — callers treat empty
        text as a soft failure rather than crashing the solver.
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Gateway-specific extra body (e.g. OpenRouter provider pinning:
        # {"provider": {"order": ["DeepInfra"], "allow_fallbacks": false}} -
        # default routing mixes providers, some of which reject a
        # model/endpoint combo or serve different quantizations). setdefault
        # so core request fields can never be clobbered by config.
        for key, value in (getattr(self.cfg, "llm_extra_body", None) or {}).items():
            payload.setdefault(key, value)

        attempts = max(1, retries + 1)
        empty_result: ChatResult | None = None
        for attempt in range(attempts):
            if attempt:
                self._sleep(min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** (attempt - 1))))
            try:
                resp = self._transport(payload)
            except Exception as exc:
                retry_gw = bool(getattr(self.cfg, "retry_gateway_errors", False))
                if _is_retryable(exc, retry_gateway=retry_gw) and attempt < attempts - 1:
                    continue
                raise
            text = self._extract_text(resp)
            usage = resp.get("usage") or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            self._record_usage(self.usage_model_name, pt, ct)
            result = ChatResult(text=text, prompt_tokens=pt, completion_tokens=ct)
            if text.strip():
                return result
            empty_result = result  # empty completion: retry if attempts remain
        assert empty_result is not None  # loop ran >= 1 successful iteration
        return empty_result

    @staticmethod
    def _extract_text(resp: dict) -> str:
        """Pull ``choices[0].message.content`` out of an OpenAI-schema response."""
        try:
            choices = resp.get("choices") or []
            message = (choices[0] or {}).get("message") or {}
            content = message.get("content")
            return content if isinstance(content, str) else ""
        except (IndexError, AttributeError, TypeError):
            return ""

    @staticmethod
    def _record_usage(usage_model_name: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage with Inspect's ledger; silent no-op if unavailable.

        Benchmark fact served: agent-eval prices ``together_ai/*`` model names
        by parameter-count fallback, so recording usage under
        ``usage_model_name`` gives each routed model an honest $/problem on the
        leaderboard instead of NULL. Guarded imports keep the package
        importable (and this call harmless) when astabench/inspect_ai are
        absent, e.g. in offline tests.
        """
        try:
            from astabench.util.model import record_model_usage_with_inspect
            from inspect_ai.model import ModelUsage
        except Exception:
            return
        try:
            usage = ModelUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            record_model_usage_with_inspect(usage_model_name, usage)
        except Exception:
            return  # usage accounting must never crash a solver run


def extract_json(text: str) -> dict | None:
    """Tolerantly parse the first-``{``-to-last-``}`` span of ``text`` as JSON.

    Benchmark fact served: this mirrors the benchmark-side parser (which takes
    ``completion[find('{') : rfind('}')+1]`` of the model output) — using the
    same tolerance on OUR model's intermediate outputs means anything that
    passes here will also parse on the scorer side. On ``JSONDecodeError`` a
    second attempt strips one brace from each end (``text[start+1:end]``, i.e.
    the spec's ``[start+1:end-1]`` in inclusive-index notation) to survive a
    stray outer-brace wrap like ``{{...}}``; otherwise returns ``None``.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        try:
            obj = json.loads(text[start + 1 : end])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None
