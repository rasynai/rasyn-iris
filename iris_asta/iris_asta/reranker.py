"""Dependency-free client for an optional local cross-encoder reranker."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RerankDocument:
    id: str
    text: str


def rerank_documents(
    url: str,
    query: str,
    documents: list[RerankDocument],
    *,
    timeout_s: float = 300.0,
) -> list[tuple[str, float]]:
    """Return ``(id, score)`` pairs best-first from a reranker sidecar.

    The wire contract is intentionally tiny and provider-neutral.  A local
    open-weight service, a GPU deployment, or another compatible scorer can
    sit behind the same endpoint.  Invalid/missing ids are ignored and any
    transport error is raised to the caller, which decides its fallback.
    """

    endpoint = str(url or "").strip().rstrip("/")
    if not endpoint or not documents:
        return []
    payload = json.dumps({
        "query": str(query or ""),
        "documents": [
            {"id": str(doc.id), "text": str(doc.text)} for doc in documents
        ],
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint + "/rerank",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
        body: Any = json.loads(response.read().decode("utf-8"))
    raw = body.get("scores") if isinstance(body, dict) else None
    if not isinstance(raw, list):
        raise ValueError("reranker response must contain a scores list")
    allowed = {doc.id for doc in documents}
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("id") or "").strip()
        if not doc_id or doc_id not in allowed or doc_id in seen:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        seen.add(doc_id)
        out.append((doc_id, score))
    out.sort(key=lambda pair: (-pair[1], pair[0]))
    return out


__all__ = ["RerankDocument", "rerank_documents"]
