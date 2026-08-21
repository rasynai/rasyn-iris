"""Client for an optional local corpus mirror (SQLite FTS5).

Why this exists: the public corpus API is heavily rate limited, and this
mirror is the only fully self-hosted path through the pipeline. The mirror
is the same underlying Semantic Scholar data, pre-filtered to the
benchmark's snapshot date, so it is date-legal by construction and answers
reliably.

Measured on the 46 gold-bearing validation queries: ONE OR query reaches
37.3% pool recall, against 43.2% for IRIS's ~168-remote-call fanout. The
match is deliberately WIDE -- narrowing it to an AND of the rarest terms
collapsed recall to 1.6%, because BM25 ranking is what discriminates and a
narrow match just deletes candidates before the ranker sees them.

Transport: HTTP to an SSH-tunnelled loopback port. Every failure mode is
soft: no service, timeout, or bad payload returns [] so callers fall back to
the remote corpus rather than losing the query.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace

DEFAULT_URL = "http://127.0.0.1:8899"
TIMEOUT_S = 120.0


def _url() -> str:
    return os.environ.get("PFBMAX_LOCAL_SEARCH_URL", DEFAULT_URL).rstrip("/")


def available(timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(_url() + "/health", timeout=timeout) as r:
            return bool(json.loads(r.read()).get("ok"))
    except Exception:
        return False


def _to_paper(d: dict) -> SimpleNamespace:
    """Shape a row like an AstaClient Paper (duck-typed by every consumer)."""
    return SimpleNamespace(
        corpusId=str(d.get("corpusId")),
        corpus_id=str(d.get("corpusId")),
        title=d.get("title") or "",
        abstract=d.get("abstract") or "",
        year=d.get("year"),
        venue=d.get("venue") or "",
        authors=d.get("authors") or [],
        citationCount=d.get("citationCount") or 0,
        publicationDate=d.get("publicationDate"),
        text=d.get("abstract") or "",   # abstract stands in for a snippet
        score=None,
        extra={"citationCount": d.get("citationCount") or 0},
    )


def _get(path: str, params: dict, timeout: float) -> list[dict]:
    url = _url() + path + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read()).get("data") or []
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []


def search(query: str, limit: int = 1000,
           timeout: float = TIMEOUT_S) -> list[SimpleNamespace]:
    """BM25 search over title+abstract; [] on any failure."""
    if not (query or "").strip():
        return []
    return [_to_paper(d) for d in _get("/search", {"q": query, "limit": limit}, timeout)]


def batch(corpus_ids, timeout: float = TIMEOUT_S) -> list[SimpleNamespace]:
    ids = [str(c) for c in (corpus_ids or []) if str(c).strip()][:500]
    if not ids:
        return []
    return [_to_paper(d) for d in _get("/batch", {"cids": ",".join(ids)}, timeout)]
