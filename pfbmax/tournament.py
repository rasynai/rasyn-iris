"""Listwise tournament over the judge's contested band.

Why this exists, in numbers: the pointwise judge strands 458/910 gold-perfect
papers at scores 0.3-0.6 (only 35 clear the >0.99 gate), yet blind A/B tests
prove the distinction IS visible in the text: gold vs same-score non-gold is
called correctly 71.5% of the time by a strong reasoning model at high
effort and 72.0% by gpt-5-mini (low), versus 64.7% for gpt-4o-mini.
Absolute "is it Perfect?" scoring throws that comparative signal away;
ordering papers directly recovers it. RankGPT-style sliding windows, Borda
aggregation, and the pointwise score kept as a weak prior.

Scope guard: only the CONTESTED ZONE is reordered: papers below the >0.99
gate, in score-sorted order, up to ZONE_MAX of them. Gate-passers stay
locked on top (they are 63% precise and already sorted), and the tail below
the zone is untouched, so a bad window can cost a few ranks, never a gold's
window membership from above.

Cost: ~2 passes x zone/stride windows x ~1.7k tokens with gpt-5-mini(low)
~= $0.02-0.05/query.

Env: PFBMAX_TOURN=1 enables (router wiring), PFBMAX_TOURN_MODEL,
     PFBMAX_TOURN_ZONE (default 150), PFBMAX_TOURN_WIN (12),
     PFBMAX_TOURN_STRIDE (6), PFBMAX_TOURN_PASSES (2).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ENDPOINT = "https://api.openai.com/v1/chat/completions"
TIMEOUT_S = 180.0
ALIAS = "ABCDEFGHIJKLMNOP"

SYS = ("You rank papers for a literature search. Given the request, its "
       "relevance criteria, and a lettered list of papers, order the papers "
       "from MOST to LEAST likely to satisfy EVERY criterion fully. Compare "
       "the papers against each other; do not grade them in isolation. "
       'Reply JSON only: {"ranking": ["X", "Y", ...]} using each letter '
       "exactly once.")


def _key():
    k = os.environ.get("OPENAI_API_KEY", "")
    if k:
        return k
    return open(os.path.join(HERE, ".openai_key"),
                encoding="utf-8").read().strip()


def _envi(name, default):
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _rank_window(query, criteria, docs, model, meter=None):
    """docs: [(cid, text)]; returns cids best-first, or None on failure."""
    letters = ALIAS[:len(docs)]
    crit = "\n".join(f"- {c}" for c in criteria)
    papers = "\n\n".join(f"PAPER {l}:\n{t[:900]}"
                         for l, (_c, t) in zip(letters, docs))
    user = (f"request: {query}\ncriteria:\n{crit}\n\n{papers}\n\n"
            f"Order the {len(docs)} papers best-first.")
    payload = {"model": model,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": user}],
               "response_format": {"type": "json_object"}}
    if model.startswith(("gpt-5", "o")):
        payload["max_completion_tokens"] = 4096
        payload["reasoning_effort"] = os.environ.get(
            "PFBMAX_TOURN_EFFORT", "low")
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = 120
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _key()})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            obj = json.loads(r.read())
        usage = obj.get("usage") or {}
        if meter is not None:
            try:
                meter.add(obj.get("model") or model,
                          usage.get("prompt_tokens") or 0,
                          usage.get("completion_tokens") or 0)
            except Exception:
                pass
        raw = json.loads(obj["choices"][0]["message"]["content"] or "{}")
        got = [str(x).strip().upper()[:1] for x in raw.get("ranking", [])]
        seen = []
        for g in got:
            if g in letters and g not in seen:
                seen.append(g)
        for l in letters:            # letters the model dropped keep order
            if l not in seen:
                seen.append(l)
        idx = {l: i for i, l in enumerate(letters)}
        return [docs[idx[l]][0] for l in seen]
    except Exception:
        return None


def rerank_zone(query, criteria, ordered, scores, texts,
                model=None, meter=None, trace=None):
    """Reorder the contested zone of ``ordered`` (score-sorted cids).

    ordered: full head order AFTER pointwise sort (best first)
    scores:  {cid: pointwise score}
    texts:   {cid: doc text}
    Returns the full reordered list (locked top + tournament zone + tail).
    """
    tr = trace if trace is not None else {}
    model = model or os.environ.get("PFBMAX_TOURN_MODEL", "gpt-5-mini")
    zone_max = _envi("PFBMAX_TOURN_ZONE", 150)
    win = _envi("PFBMAX_TOURN_WIN", 12)
    stride = _envi("PFBMAX_TOURN_STRIDE", 6)
    passes = _envi("PFBMAX_TOURN_PASSES", 2)

    locked = [c for c in ordered if (scores.get(c) or 0.0) > 0.99]
    rest = [c for c in ordered if (scores.get(c) or 0.0) <= 0.99]
    zone = [c for c in rest if texts.get(c)][:zone_max]
    tail = [c for c in rest if c not in set(zone)]
    if len(zone) < win:
        return ordered

    order = list(zone)
    base_rank = {c: i for i, c in enumerate(zone)}
    # Points POOL across passes: the offline sweep on dumped windows showed
    # pooled Borda (+cap 8) at +0.0151 vs +(-0.0105..-0.0001) for the
    # sequential per-pass reorder this loop originally applied. Pass 2 still
    # re-slides windows over the pass-1 order (new match-ups), but the final
    # order comes from ALL windows at once.
    pooled = {c: [] for c in zone}
    for _p in range(passes):
        windows = []
        i = 0
        while i < len(order):
            w = order[i:i + win]
            if len(w) >= 4:
                windows.append((i, w))
            if i + win >= len(order):
                break
            i += stride
        results = {}
        with ThreadPoolExecutor(max_workers=_envi("PFBMAX_TOURN_WORKERS", 6)) as ex:
            futs = {ex.submit(_rank_window, query, criteria,
                              [(c, texts[c]) for c in w], model, meter): (s, w)
                    for s, w in windows}
            for f in futs:
                s, w = futs[f]
                got = f.result()
                if got:
                    results[s] = got
        # accumulate Borda points into the cross-pass pool; the within-pass
        # reorder below only shapes the NEXT pass's window composition.
        for s, w in windows:
            got = results.get(s)
            if not got:
                continue
            for pos, c in enumerate(got):
                pooled[c].append(1.0 - pos / max(1, len(got) - 1))
        try:
            prior_w = float(os.environ.get("PFBMAX_TOURN_PRIOR") or 0.4)
        except ValueError:
            prior_w = 0.4
        def key(c):
            p = (sum(pooled[c]) / len(pooled[c])) if pooled[c] else 0.5
            prior = 1.0 - base_rank[c] / max(1, len(zone) - 1)
            return -( (1 - prior_w) * p + prior_w * prior )
        order = sorted(order, key=key)
        # dump raw window outputs: aggregation becomes a $0 offline sweep
        dump = os.environ.get("PFBMAX_TOURN_DUMP", "").strip()
        if dump:
            try:
                with open(dump, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "query": query, "pass": _p, "zone": zone,
                        "windows": [{"start": s, "in": w,
                                     "out": results.get(s)}
                                    for s, w in windows],
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass

    # DEMOTION CAP. Pilot 1 (no cap): net -0.0001 -- one +0.094 rescue but
    # two catastrophes (-0.035, -0.071 to zero) where the tournament dragged
    # base-trusted papers out of the scored window. Rescues need unlimited
    # ascent; base papers need bounded descent: a paper may fall at most
    # DEMOTE_CAP ranks below where the base order had it.
    cap = _envi("PFBMAX_TOURN_DEMOTE_CAP", 8)
    if cap > 0:
        final = list(order)
        for c in zone:
            want = base_rank[c] + cap
            cur = final.index(c)
            if cur > want:
                final.remove(c)
                final.insert(want, c)
        order = final
    tr["tournament"] = {"model": model, "zone": len(zone),
                        "windows": len(windows), "passes": passes,
                        "failed": sum(1 for s, _w in windows
                                      if s not in results)}
    return locked + order + tail
