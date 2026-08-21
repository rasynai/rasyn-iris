"""MCP (streamable-HTTP) transport for the Asta Scientific Corpus tools.

The ASTA_TOOL_KEY handed out for ASTABench is only valid against the Asta
MCP endpoint (``https://asta-tools.allen.ai/mcp/v1``), NOT against
``api.semanticscholar.org`` — this module gives :class:`~iris_asta.asta_client.AstaClient`
a second wire path speaking JSON-RPC 2.0 over streamable-HTTP MCP.

Verified live protocol facts encoded here:

* POST ``https://asta-tools.allen.ai/mcp/v1`` with headers
  ``Content-Type: application/json``,
  ``Accept: application/json, text/event-stream``,
  ``x-api-key: <cfg.asta_tool_key>``.
* Responses are SSE-framed even for single JSON-RPC responses
  (``event: message`` then ``data: {json-rpc response}`` lines);
  :func:`parse_sse_json` extracts the LAST ``data:`` line's JSON.  Plain
  (non-SSE) JSON bodies are accepted as a fallback.
* ``initialize`` (protocolVersion ``2025-03-26``) may return an
  ``Mcp-Session-Id`` response header; when present it is echoed back as an
  ``Mcp-Session-Id`` request header on every subsequent request, and the
  client sends the ``notifications/initialized`` notification (no id)
  after ``initialize`` and before any ``tools/call`` — per streamable-HTTP
  MCP.
* Tool results arrive as ``result.content[0].text`` containing JSON (MCP
  text content) — that inner JSON is parsed and returned;
  ``result.structuredContent`` is the fallback when the text payload is
  absent or not JSON.
* Rate limit: 10 req/s per endpoint (429 on excess) — the transport reuses
  the client's token-bucket :class:`~iris_asta.asta_client.RateLimiter`
  (at ``cfg.rate_limit_rps``, below that ceiling) and the same
  429/504/529 retry policy (exponential backoff, 60 s cap, max 8
  attempts).

Stdlib only.  ``http`` (test injection) is a callable
``http(url: str, headers: dict, body: bytes) -> (status: int, headers: dict, body: str)``.
JSON-RPC ``error`` responses raise ``RuntimeError`` carrying the error
message; MCP tool-level errors (``result.isError``) also raise
``RuntimeError``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .asta_client import RateLimiter, _backoff_delay
from .config import Config

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
PROTOCOL_VERSION = "2025-03-26"
CLIENT_INFO = {"name": "iris-asta", "version": "0.1"}

_RETRY_CODES = (429, 504, 529)
_MAX_ATTEMPTS = 8
#: Wall-clock budget for one MCP call attempt. Typical corpus responses are
#: ~0.8s; the server occasionally throttles a request to 30s+ while sending
#: SSE ": ping" heartbeats. Abort past this and retry (a retry near-always
#: hits the fast path); also the urlopen socket timeout so a ping-less silent
#: connection can't block forever. Overridable via IRIS_ASTA_MCP_DEADLINE_S:
#: during a server-side throttling episode the "fast path" a retry would hit
#: is itself 12-30s, so the default cuts off every attempt and the run stalls;
#: raising the budget lets a slow-but-real response land instead of burning
#: all 8 attempts. Keep the default at 12s for healthy-corpus runs.
def _env_deadline(default: float = 12.0) -> float:
    try:
        v = float(os.environ.get("IRIS_ASTA_MCP_DEADLINE_S", "") or default)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


_MCP_CALL_DEADLINE = _env_deadline()

# Live finding: the server sometimes wraps a transient backend failure
# ("ConnectionRefusedError ... RetryError") as an MCP tool error. ONLY those
# are worth a client-level retry (max 2 retries, 5 s pause); every other tool
# error is a real bug and raises immediately.
_TRANSIENT_TOOL_ERROR_MARKERS = ("ConnectionRefused", "RetryError")
_TRANSIENT_TOOL_RETRIES = 2  # max extra attempts after the first
_TRANSIENT_TOOL_PAUSE_S = 5.0


def _is_transient_tool_error(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in _TRANSIENT_TOOL_ERROR_MARKERS)


def parse_sse_json(body: str) -> dict:
    """Extract the JSON object from the LAST ``data:`` line of an SSE body.

    Verified live fact served: the Asta MCP endpoint frames even single
    JSON-RPC responses as SSE (``event: message`` / ``data: {...}``); the
    JSON-RPC response is the last ``data:`` line.  Junk lines (``event:``,
    ``:`` comments, empties) are ignored.  A body with no ``data:`` line is
    parsed as plain JSON (fallback for servers answering
    ``application/json`` directly); an empty body raises ``ValueError``.
    """
    last = None
    for line in body.splitlines():
        if line.startswith("data:"):
            last = line[len("data:"):].strip()
    if last is None:
        stripped = body.strip()
        if not stripped:
            raise ValueError("empty MCP response body (no data: line, no JSON)")
        return json.loads(stripped)
    return json.loads(last)


def _header_value(headers: dict | None, name: str) -> str | None:
    """Case-insensitive header lookup (urllib/servers vary header casing)."""
    lname = name.lower()
    for k, v in (headers or {}).items():
        if str(k).lower() == lname:
            return v
    return None


class McpTransport:
    """JSON-RPC 2.0 client over streamable-HTTP MCP for the Asta corpus tools.

    ``initialize()`` performs the MCP handshake (initialize +
    notifications/initialized, capturing ``Mcp-Session-Id`` when the server
    issues one); ``call_tool(name, arguments)`` lazily initializes on first
    use, then unwraps the MCP tool result into the inner JSON dict.
    """

    def __init__(self, cfg: Config, http=None, *, max_attempts: int | None = None):
        self.cfg = cfg
        self._max_attempts = max(
            1, min(_MAX_ATTEMPTS, int(max_attempts or _MAX_ATTEMPTS))
        )
        self._http = http if http is not None else self._default_http
        self._limiter = RateLimiter(cfg.rate_limit_rps)
        self._sleep = time.sleep  # test hook: replace to avoid real backoff sleeps
        self._session_id: str | None = None
        self._initialized = False
        self._next_id = 0

    # ---------------------------------------------------------------- HTTP --

    def _default_http(self, url: str, headers: dict, body: bytes):
        """Default transport: POST ``body`` to ``url`` via urllib.

        Returns ``(status, headers, body_text)`` even for HTTP error
        statuses (HTTPError is caught and flattened) so the retry loop can
        classify status codes uniformly with injected fakes.
        """
        req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        deadline = time.monotonic() + _MCP_CALL_DEADLINE
        sock_timeout = min(self.cfg.timeout_s, _MCP_CALL_DEADLINE)
        try:
            with urllib.request.urlopen(req, timeout=sock_timeout) as resp:
                status = resp.status
                resp_headers = dict(resp.headers.items())
                ctype = resp_headers.get("Content-Type", "") or resp_headers.get(
                    "content-type", ""
                )
                # Streamable-HTTP MCP replies with text/event-stream and then
                # HOLDS the connection open (sending ": ping" heartbeats) for
                # possible server-initiated messages; resp.read() would block
                # until EOF. Worse, the server's response time is INTERMITTENT
                # (~0.8s typical, occasionally 30s+ under throttle) and the
                # per-read socket timeout never fires because pings keep
                # resetting it. So enforce a WALL-CLOCK deadline: read
                # line-by-line, return the instant our JSON-RPC ``data:`` event
                # arrives, and if the deadline passes first, return 504 so the
                # retry loop reissues (a retry almost always hits the fast
                # path). Non-SSE bodies (plain JSON / 202) read whole.
                if "text/event-stream" in ctype.lower():
                    data_parts: list[str] = []
                    for raw in resp:  # iterates lines as they arrive
                        if time.monotonic() > deadline:
                            return 504, resp_headers, ""  # slow -> retry fresh
                        line = raw.decode("utf-8", "replace").rstrip("\r\n")
                        if line.startswith("data:"):
                            data_parts.append(line[5:].lstrip())
                        elif line == "" and data_parts:
                            payload = "".join(data_parts)
                            try:
                                obj = json.loads(payload)
                            except ValueError:
                                data_parts = []
                                continue
                            # return only once we have THIS request's response
                            if isinstance(obj, dict) and (
                                "result" in obj or "error" in obj or "id" in obj
                            ):
                                return status, resp_headers, "data: " + payload
                            data_parts = []
                    # stream ended without a matching event
                    return status, resp_headers, "data: " + "".join(data_parts)
                return status, resp_headers, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", "replace")
            except Exception:
                err_body = ""
            err_headers = dict(exc.headers.items()) if exc.headers is not None else {}
            return exc.code, err_headers, err_body
        except (TimeoutError, urllib.error.URLError, OSError):
            # socket read timed out (silent/ping-less slow connection) or the
            # connection dropped -> treat as a retryable 504 so _rpc reissues.
            return 504, {}, ""

    def _request_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.cfg.asta_tool_key:
            headers["x-api-key"] = self.cfg.asta_tool_key
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _rpc(self, payload: dict, expect_response: bool = True) -> dict:
        """POST one JSON-RPC message; retry 429/504/529 with exp backoff.

        Captures ``Mcp-Session-Id`` from response headers for echo on later
        requests.  Notifications (``expect_response=False``) skip body
        parsing — per streamable-HTTP MCP the server answers them with a
        bare 202 Accepted.  JSON-RPC ``error`` responses raise
        ``RuntimeError`` with the server's error message.
        """
        body = json.dumps(payload).encode("utf-8")
        status, resp_headers, resp_body = 0, {}, ""
        for attempt in range(self._max_attempts):
            if attempt:
                self._sleep(_backoff_delay(attempt - 1))
            self._limiter.acquire()
            status, resp_headers, resp_body = self._http(
                MCP_URL, self._request_headers(), body
            )
            if status in _RETRY_CODES and attempt < self._max_attempts - 1:
                continue
            break
        if status >= 400:
            raise RuntimeError(
                f"MCP HTTP {status} for {payload.get('method')}: {resp_body[:500]}"
            )
        session_id = _header_value(resp_headers, "Mcp-Session-Id")
        if session_id:
            self._session_id = session_id
        if not expect_response:
            return {}
        message = parse_sse_json(resp_body)
        error = message.get("error")
        if error:
            raise RuntimeError(
                f"MCP JSON-RPC error {error.get('code')}: {error.get('message')}"
            )
        return message.get("result") or {}

    # ----------------------------------------------------------------- MCP --

    def initialize(self) -> dict:
        """MCP handshake: initialize, then the initialized notification.

        Verified live facts served: initialize works with protocolVersion
        2025-03-26; the server may return an ``Mcp-Session-Id`` header
        (captured by ``_rpc``); ``notifications/initialized`` (no id) must
        be sent before ``tools/call``.
        """
        self._next_id += 1
        result = self._rpc(
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": dict(CLIENT_INFO),
                },
            }
        )
        self._rpc(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            expect_response=False,
        )
        self._initialized = True
        return result

    def call_tool(self, name: str, arguments: dict) -> dict:
        """``tools/call`` one Asta corpus tool; returns the inner JSON dict.

        Lazily runs :meth:`initialize` on first use.  Tool results arrive
        as ``result.content[0].text`` containing JSON — that inner JSON is
        parsed and returned (a bare JSON list is wrapped as
        ``{"data": [...]}`` so callers always get a dict);
        ``result.structuredContent`` is the fallback.  MCP tool errors
        (``result.isError``) raise ``RuntimeError`` with the server's
        message intact.  Live finding: transient backend failures surface
        as tool errors containing ``ConnectionRefused``/``RetryError`` —
        ONLY those get a client-level retry (max 2 retries, 5 s pause);
        any other tool error raises immediately.
        """
        if not self._initialized:
            self.initialize()
        for attempt in range(1 + _TRANSIENT_TOOL_RETRIES):
            if attempt:
                self._sleep(_TRANSIENT_TOOL_PAUSE_S)
            self._next_id += 1
            result = self._rpc(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": dict(arguments)},
                }
            )
            try:
                return _unwrap_tool_result(name, result)
            except RuntimeError as exc:
                if attempt < _TRANSIENT_TOOL_RETRIES and _is_transient_tool_error(exc):
                    continue
                raise
        raise RuntimeError("unreachable: tool retry loop exited without return/raise")


def _unwrap_tool_result(name: str, result: dict) -> dict:
    """Unwrap an MCP tools/call result into the tool's inner JSON payload."""
    content = result.get("content") or []
    if result.get("isError"):
        texts = [
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("text")
        ]
        raise RuntimeError(f"MCP tool {name!r} error: {' '.join(texts) or result}")
    inners = []
    for block in content:
        text = block.get("text") if isinstance(block, dict) else None
        if text is None:
            continue
        try:
            inner = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        inners.append(inner)
    if len(inners) == 1:
        inner = inners[0]
        if isinstance(inner, dict):
            return inner
        if isinstance(inner, list):
            return {"data": inner}
    elif inners:
        # Live finding (2026-07-15): list-returning tools
        # (search_authors_by_name, get_author_papers, get_citations) stream
        # ONE JSON object per content block - taking content[0] silently
        # collapsed every author/citation list to a single item and zeroed
        # the PFB metadata slice. Flatten multi-block results into the same
        # {"data": [...]} envelope the single-block list tools use.
        items: list = []
        for inner in inners:
            if isinstance(inner, list):
                items.extend(inner)
            else:
                items.append(inner)
        return {"data": items}
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    raise RuntimeError(
        f"MCP tool {name!r} returned no parseable JSON payload: {result!r}"
    )
