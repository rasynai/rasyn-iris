"""Minimal OpenAI chat-completions client for PFB-MAX. Stdlib urllib only.

Owned by the cost accounting part of the pipeline, alongside the cost meter.

    from pfbmax.costmeter import CostMeter
    from pfbmax.llm import LLM

    llm = LLM(meter=CostMeter())                 # backbone: gpt-4o-mini
    obj = llm.json("system prompt", "user prompt")   # dict | None
    txt = llm.text("system prompt", "user prompt")   # str

Behavior:
- temperature 0; ``.json()`` uses response_format json_object plus a
  tolerant first-``{``-to-last-``}`` parse (returns None if unparseable).
- retries (default 2, exponential backoff) on 429 / 5xx / timeouts /
  connection errors; other HTTP errors raise immediately.
- HTTP error bodies are folded into the raised LLMError (key-scrubbed).
- usage (prompt/completion tokens) is recorded into the CostMeter passed
  to the constructor (duck-typed: anything with ``.add(model, pt, ct)``).

API key: OPENAI_API_KEY from the environment, else parsed from
``iris_asta/.env`` found by walking up from this file. The key is NEVER
printed or logged, and key-shaped substrings are scrubbed from all error
messages. Tests inject ``transport=`` and an explicit fake ``api_key=``
so they never touch the real key or the network.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{4,}")


def _record_inspect_usage(model: str, prompt_tokens: int,
                          completion_tokens: int) -> None:
    """Report our own token spend to the harness's usage ledger.

    We call OpenAI directly (raw urllib) rather than through inspect's model
    API, so inspect sees none of it: the official eval log recorded only the
    SCORER's tokens and our solver cost read as $0.00 -- an understatement
    that would put a false cost on the leaderboard, where cost is half the
    ranking. This mirrors iris_asta.backbone._record_usage: push a ModelUsage
    into inspect's ledger under the real model name so agent-eval prices it.

    Guarded twice over -- absent harness, or any recording failure, must
    never disturb a solve (usage accounting is bookkeeping, not the answer).
    """
    if not (prompt_tokens or completion_tokens):
        return
    try:
        from astabench.util.model import record_model_usage_with_inspect
        from inspect_ai.model import ModelUsage
    except Exception:
        return
    try:
        record_model_usage_with_inspect(
            model if "/" in model else f"openai/{model}",
            ModelUsage(input_tokens=prompt_tokens,
                       output_tokens=completion_tokens,
                       total_tokens=prompt_tokens + completion_tokens),
        )
    except Exception:
        return


class LLMError(RuntimeError):
    """Raised on transport failure or non-retryable / exhausted HTTP errors.

    ``status`` is the HTTP status code, or None for network-level failures.
    The (scrubbed, truncated) HTTP error body is folded into the message.
    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _scrub(text):
    """Redact anything shaped like an OpenAI API key."""
    return _KEY_RE.sub("sk-***", text or "")


def _parse_env_file(path):
    """Tolerant KEY=VALUE .env parser (comments, export, quotes, CRLF, BOM)."""
    out = {}
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def _find_env_file():
    here = Path(__file__).resolve()
    for base in here.parents:
        cand = base / "iris_asta" / ".env"
        if cand.is_file():
            return cand
    return None


def _load_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_file = _find_env_file()
    if env_file is not None:
        key = _parse_env_file(env_file).get("OPENAI_API_KEY", "").strip()
        if key:
            return key
    return None


def _default_transport(url, data, headers, timeout):
    """POST ``data`` to ``url``; return (status_code, body_text).

    HTTP error statuses are returned (body read from the error stream), so
    the caller owns retry/raise policy. Network errors and timeouts raise.
    """
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        try:
            body = err.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return err.code, body


def _loose_json(text):
    """Contract-mandated tolerant parse: whole string, else first-{ to last-}.

    Returns a dict, or None if no dict can be recovered.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    return None


class LLM:
    """gpt-4o-mini via raw OpenAI chat completions (no SDK).

    Constructor (contract): ``LLM(meter=None, model="gpt-4o-mini")``.
    Extra keyword-only knobs:
      api_key   -- explicit key (tests use a fake); default: env / iris_asta/.env
      endpoint  -- full chat-completions URL
      timeout   -- per-request seconds (default 90)
      retries   -- extra attempts after the first (default 2)
      backoff   -- base seconds between attempts: backoff * 2**(attempt-1)
      transport -- callable(url, data_bytes, headers, timeout) -> (status, body_str);
                   injected by unit tests to avoid the network
    """

    def __init__(self, meter=None, model=DEFAULT_MODEL, *, api_key=None,
                 endpoint=DEFAULT_ENDPOINT, timeout=90.0, retries=2,
                 backoff=1.0, transport=None):
        self.meter = meter
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = int(retries)
        self.backoff = backoff
        self.transport = transport if transport is not None else _default_transport
        self._custom_transport = transport is not None
        self._api_key = api_key
        self.calls = 0
        self.last_text = None  # raw content of the last successful completion

    # -- public API (contract) -------------------------------------------
    def json(self, system, user, max_tokens=900, model=None):
        """JSON-mode completion -> dict, or None if the reply isn't parseable.

        ``model`` overrides the backbone for this one call. Used where the
        cheap backbone lacks the needed KNOWLEDGE rather than the needed
        reasoning: the cheap model does not recognise lesser-known artifact
        names, while gpt-4o does, and one such call per specific
        query is negligible against that slice's ~$0.001 total.
        """
        content = self._complete(system, user, max_tokens, json_mode=True,
                                 model=model)
        return _loose_json(content)

    def text(self, system, user, max_tokens=900):
        """Plain completion -> stripped text ('' if the reply was empty)."""
        content = self._complete(system, user, max_tokens, json_mode=False)
        return (content or "").strip()

    # -- internals ---------------------------------------------------------
    def _key(self):
        if self._api_key is None:
            self._api_key = _load_api_key()
        if self._api_key is None:
            if self._custom_transport:
                self._api_key = "sk-fake-for-injected-transport"
            else:
                raise LLMError(
                    "OPENAI_API_KEY not found in environment or iris_asta/.env")
        return self._api_key

    def _complete(self, system, user, max_tokens, json_mode, model=None):
        system = system or ""
        user = user or ""
        if json_mode and "json" not in (system + " " + user).lower():
            # OpenAI rejects json_object mode unless "json" appears in messages.
            system = (system + "\nRespond with a single valid JSON object.").strip()
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": int(max_tokens),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self._key(),
        }

        attempts = self.retries + 1
        err = None
        for attempt in range(attempts):
            if attempt:
                time.sleep(self.backoff * (2 ** (attempt - 1)))
            try:
                status, body = self.transport(self.endpoint, data, headers,
                                              self.timeout)
            except Exception as exc:  # timeouts, DNS, resets -> retryable
                err = LLMError("transport error: "
                               + (_scrub(str(exc)) or type(exc).__name__))
                continue
            if status == 200:
                return self._on_success(body)
            err = LLMError("OpenAI HTTP %d: %s" % (status, _scrub(body)[:800]),
                           status=status)
            if status == 429 or status >= 500:
                continue
            raise err
        raise LLMError("giving up after %d attempts: %s" % (attempts, err),
                       status=getattr(err, "status", None))

    def _on_success(self, body):
        try:
            resp = json.loads(body)
        except ValueError:
            raise LLMError("non-JSON 200 response: " + _scrub(body)[:300],
                           status=200)
        usage = resp.get("usage") or {}
        _pt = int(usage.get("prompt_tokens") or 0)
        _ct = int(usage.get("completion_tokens") or 0)
        if self.meter is not None:
            self.meter.add(resp.get("model") or self.model, _pt, _ct)
        _record_inspect_usage(resp.get("model") or self.model, _pt, _ct)
        self.calls += 1
        content = ""
        choices = resp.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
        self.last_text = content
        return content
