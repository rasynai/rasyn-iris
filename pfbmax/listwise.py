"""RankGPT-style listwise reranking with a sliding window.

Why listwise: the existing judge was POINTWISE, meaning it graded each paper
independently against the criteria and returned booleans. A pointwise grader
cannot express "this paper is better than that one", so it cannot produce an
ordering; ours bucketed papers into three coarse tiers and left the fused
retrieval score to break ties. Measured consequence: rank 0.668 but
recall@K 0.083, and deleting the judge outright *improved* the score.

Listwise reranking (Sun et al., RankGPT 2023; RankZephyr; the 2025-26
follow-ups) instead shows the model a window of candidates at once and asks
for a permutation, so relative judgements are the output rather than a
by-product. It is the reported state of the art for exactly our failure mode,
and the sliding window keeps it O(n) in candidates: rank the bottom window,
carry the winners up, repeat toward the top.

That matters here because the benchmark's semantic score is
harmonic(nDCG, recall@K) with K ~ 2x the number of perfect papers, which is
a pure ordering problem over a pool that already contains ~43% of the golds.

This stage is opt-in: the router runs it only when PFBMAX_LISTWISE=1 is set.
Model choice is deliberate: the official scorer is gpt-4o-2024-11-20, and a
reranker from the same family agrees with it far better than gpt-4o-mini did
(the mini judge's disagreement at the "perfect vs merely highly relevant"
boundary is what made it worthless). The default model is gpt-4o-2024-11-20;
set PFBMAX_RANK_MODEL to override.

Integrity: ranks on query + criteria text only; no gold ids, no per-query
branching. Papers the model omits keep their prior relative order.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

WINDOW = 20               # candidates shown per call (RankGPT default range)
STEP = 10                 # window overlap -> each doc seen ~twice
DEPTH = 100               # rerank this many top candidates
DOC_CHARS = 380           # per-doc budget; windows must fit comfortably
TIMEOUT_S = 180.0
ENDPOINT = "https://api.openai.com/v1/chat/completions"

_PERM_RE = re.compile(r"\[(\d+)\]")

_SYSTEM = (
    "You rank scientific papers by how well they satisfy a search request. "
    "You will be given a query, its relevance criteria, and a numbered list "
    "of paper excerpts. Rank ALL of them from most to least relevant.\n"
    "A paper is most relevant only if it satisfies EVERY criterion; a paper "
    "that satisfies most criteria but clearly misses one ranks below every "
    "paper that satisfies all of them.\n"
    "Respond with the identifiers in descending relevance, e.g. "
    "[4] > [1] > [7] > ... , including every identifier exactly once. "
    "Output only that ranking line."
)


def _key() -> str:
    k = os.environ.get("OPENAI_API_KEY", "")
    if k:
        return k
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".openai_key"),
                 os.path.join(os.path.dirname(here), "iris_asta", ".env")):
        try:
            txt = open(path, encoding="utf-8").read()
        except Exception:
            continue
        if path.endswith(".openai_key"):
            return txt.strip()
        for line in txt.splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _is_reasoning(model: str) -> bool:
    """gpt-5.x / o-series speak a different dialect than the gpt-4 chat API."""
    m = (model or "").lower()
    return m.startswith("gpt-5") or re.match(r"^o[1-9]", m) is not None


def _chat(model: str, messages: list[dict], max_tokens: int,
          meter=None, timeout: float = TIMEOUT_S) -> str:
    if _is_reasoning(model):
        # Reasoning models reject temperature, rename the token cap, and spend
        # hidden reasoning tokens against it -- so the cap must cover thinking
        # plus the permutation line, not just the permutation line.
        payload = {"model": model, "messages": messages,
                   "max_completion_tokens": max(2048, max_tokens * 12),
                   "reasoning_effort": os.environ.get(
                       "PFBMAX_RANK_EFFORT", "low")}
    else:
        payload = {"model": model, "messages": messages,
                   "temperature": 0.0, "max_tokens": max_tokens}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _key()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    usage = obj.get("usage") or {}
    pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    if meter is not None:
        try:
            meter.add(obj.get("model") or model, pt, ct)
        except Exception:
            pass
    try:
        from llm import _record_inspect_usage
        _record_inspect_usage(obj.get("model") or model, pt, ct)
    except Exception:
        pass
    choices = obj.get("choices") or []
    return (choices[0].get("message") or {}).get("content", "") if choices else ""


def _parse_permutation(text: str, n: int) -> list[int]:
    """Identifiers in the model's stated order, 0-based, deduped and bounded."""
    out, seen = [], set()
    for m in _PERM_RE.findall(text or ""):
        try:
            i = int(m) - 1
        except ValueError:
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def rerank(query: str, criteria: list[str], docs: list[tuple[str, str]],
           model: str | None = None, depth: int = DEPTH,
           window: int = WINDOW, step: int = STEP, meter=None,
           trace: dict | None = None) -> list[str]:
    """Return cids reordered by listwise relevance.

    ``docs`` is [(cid, text)] in current (fused) order. Only the top ``depth``
    are reranked; the tail keeps its order, since the scored window is far
    shallower than the submission and reranking the tail buys nothing.
    """
    model = model or os.environ.get("PFBMAX_RANK_MODEL", "gpt-4o-2024-11-20")
    _d = os.environ.get("PFBMAX_RANK_DEPTH", "").strip()
    if _d.isdigit():
        depth = int(_d)
    tr = trace if trace is not None else {}
    if not docs:
        return []
    head = docs[:depth]
    tail = [cid for cid, _t in docs[depth:]]
    order = list(range(len(head)))
    crit = "\n".join(f"- {c}" for c in (criteria or [])[:6]) or "- (none given)"

    passes = 1
    _p = os.environ.get("PFBMAX_RANK_PASSES", "").strip()
    if _p.isdigit():
        passes = max(1, int(_p))

    calls = 0
    # Repeated bottom-to-top sweeps. One sweep only lets a document climb
    # ~window/step positions, so a gold buried deep cannot reach the scored
    # top-K in a single pass; RankZephyr reports repeated passes as a real
    # gain for exactly this reason. Each extra pass costs another sweep.
    for _pass in range(passes):
      start = max(0, len(order) - window)
      while True:
          idxs = order[start:start + window]
          if len(idxs) < 2:
              if start == 0:
                  break
              start = max(0, start - step)
              continue
          lines = []
          for n, oi in enumerate(idxs, 1):
              _cid, text = head[oi]
              lines.append(f"[{n}] {' '.join((text or '').split())[:DOC_CHARS]}")
          user = (f"Query: {query}\n\nCriteria:\n{crit}\n\n"
                  f"Papers ({len(idxs)}):\n" + "\n".join(lines) +
                  f"\n\nRank all {len(idxs)} identifiers, best first.")
          try:
              reply = _chat(model, [{"role": "system", "content": _SYSTEM},
                                    {"role": "user", "content": user}],
                            max_tokens=16 * len(idxs) + 64, meter=meter)
              calls += 1
          except (urllib.error.URLError, OSError, ValueError, TimeoutError):
              reply = ""
          perm = _parse_permutation(reply, len(idxs))
          if perm:
              ranked = [idxs[p] for p in perm]
              ranked += [i for i in idxs if i not in set(ranked)]
              order[start:start + window] = ranked
          if start == 0:
              break
          start = max(0, start - step)

    tr["listwise"] = {"model": model, "calls": calls, "depth": len(head),
                      "passes": passes}
    return [head[i][0] for i in order] + tail
