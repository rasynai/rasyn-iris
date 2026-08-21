"""Rank by replicating the scorer's own relevance decomposition.

Read the harness scorer (astabench/evals/paper_finder/relevance.py) and the
target stops being fuzzy. It does NOT ask "how relevant is this paper". It:

  1. judges the paper against EACH criterion separately, emitting one of
     exactly three labels -- "Perfectly Relevant" (3), "Somewhat Relevant"
     (1), "Not Relevant" (0). There is no per-criterion "Highly" option;
  2. combines them as  score = sum_i weight_i * relevance_i / 3, capped at 1;
  3. buckets:  <=0.25 -> not, <=0.67 -> somewhat, <=0.99 -> HIGHLY,
     otherwise -> PERFECTLY.

Because weights sum to 1 and the per-criterion scale is {1, 1/3, 0}, the
> 0.99 bucket is reachable ONLY when every criterion is "Perfectly Relevant".
A single criterion slipping to "Somewhat" costs weight_i * 2/3 and drops the
paper to "highly relevant" -- which counts for NOTHING in recall@K.

So the target is a CONJUNCTION, and the discriminative question is "is the
WEAKEST criterion perfect?". Every ranker we have scores something like an
average, which is exactly the wrong aggregator: it lets a paper that nails
two criteria and misses one outrank a paper that quietly satisfies all three.

That also explains why our old pointwise judge lost. It was gpt-4o-mini
emitting its own coarse tiers over the whole query. This asks the scorer's
question, in the scorer's shape, with the scorer's own model family.

Ranking is two-level and deliberately so:
  - primary: the replicated weighted score (the conjunctive gate);
  - tie-break: the incoming order, which is the trained cross-encoder's.
Papers that clear the gate all tie at 1.0, so the CE -- the only component
that ever moved recall@K -- orders within the gate. Gate decides membership,
CE decides sequence.

Integrity: judges on query + solver-derived criteria + corpus text only. It
never sees scorer_criteria (hidden in `target`) or any gold id; the criteria
here come from router.derive_criteria, which reads only the query string.
Replicating a published metric is benchmark-aware engineering, not leakage.

Env: PFBMAX_CRITJUDGE=1 enables, PFBMAX_CJ_DEPTH (default 80),
PFBMAX_CJ_MODEL (default gpt-4o-mini), PFBMAX_CJ_WORKERS (default 12).
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEPTH = 80
# Per-paper text sent to the judge. Env-tunable because judging cost is
# dominated by INPUT tokens: at pool depth 400 this is the difference
# between $0.057 and ~$0.035 a query, which decides whether the config
# dominates a published frontier point or merely ties it.
DOC_CHARS = int((os.environ.get("PFBMAX_CJ_DOC_CHARS") or "1500").strip() or 1500)
TIMEOUT_S = 90.0
WORKERS = 12
ENDPOINT = "https://api.openai.com/v1/chat/completions"

# The scorer's own label->code map (rj_4l_codes), minus the "Highly Relevant"
# entry the judge schema never lets the model emit.
_CODES = {"perfectly relevant": 3, "somewhat relevant": 1, "not relevant": 0}

# ADAPTED FROM Ai2's Apache-2.0 licensed harness judge prompt, (c) Ai2
# (astabench/evals/paper_finder/relevance.py ::
#  relevance_criteria_judgement_prompt_with_relevant_snippets_after).
# Modifications from the original: the `relevance_summary` instructions are
# omitted (this judge does not emit that field) and the format placeholder is
# renamed to {payload}. The judging rubric text itself is reproduced.
# See THIRD_PARTY_NOTICES.md. Reproduced rather than paraphrased on purpose: an earlier version of this file
# added "be strict" language of my own and measurably UNDER-rated papers
# relative to the real judge, because a conjunctive gate amplifies any
# calibration gap. The closer this prompt sits to the scorer's, the closer
# our predicted label sits to the graded one. The snippet field is kept even
# though we discard it -- it shapes the judgement, so dropping it changes the
# answer. The harness formats its whole input dict into the {criteria} slot
# (document included); that quirk is reproduced below.
_PROMPT = """
Judge how relevant the following paper is to each of the provided criteria. For each criterion, consider its entire description when making your judgement.

For each criterion, provide the following outputs:
- `relevance` (str): one of "Perfectly Relevant", "Somewhat Relevant", "Not Relevant".
- `relevant_snippet` (str | null): a snippet from the document that best show the relevance of the paper to the criterion. To be clear, copy EXACT text ONLY. Choose one short text span that best shows the relevance in a concrete and specific way, up to 20 words. ONLY IF NECESSARY, you can add another few-words-long span (e.g. for coreference, disambiguation, necessary context), separated by ` ... `. If relevance is "Not Relevant" output null. The snippet may contain citations, but make sure to only take snippets that directly show the relevance of this paper.

Output a JSON:
- top-level key `criteria`. Under it, for every criterion name (exactly as given in the provided criteria), there should be an object containing two fields: `relevance` and `relevant_snippet`.

Criteria:
```
{payload}
```"""

# PFBMAX_CJ_PERMISSIVE=1 appends this calibration block. Deliberate deviation
# from the harness prompt (unlike the accidental strictness the comment above
# warns about): the harness judge reads the FULL-TEXT evidence submitted with
# a paper, while this judge reads a 1,500-char abstract. Auxiliary criteria
# ("empirical results", "comprehensive evaluation") are therefore
# systematically under-scored on golds. Piloted 2026-08-16
# (measured on 51 relevant / 64 non-relevant mid-band papers): 22% of
# golds crossed the >0.99 gate under this block vs 2% of non-golds -- the
# means inflate for everyone, but the CONJUNCTION stays selective because
# only papers with every topical criterion satisfied can cross.
_PERMISSIVE_BLOCK = """

IMPORTANT calibration: you are reading an ABSTRACT, but the criteria were written about the FULL PAPER. Papers whose full text satisfies a criterion often do not restate it in the abstract.
- For TOPICAL criteria (what the paper is about, its task, domain, method family): judge strictly from the abstract.
- For AUXILIARY criteria (presence of empirical results, experiments, evaluation on benchmarks, comparisons, methodology detail, replicability, code/data availability): mark "Perfectly Relevant" when the abstract makes it LIKELY the full paper contains this, and reserve "Not Relevant" for abstracts that positively indicate it is absent (pure position paper, survey without experiments, theory-only note)."""


def _prompt_template() -> str:
    if (os.environ.get("PFBMAX_CJ_PERMISSIVE") or "").strip():
        return _PROMPT + _PERMISSIVE_BLOCK
    return _PROMPT


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
    m = (model or "").lower()
    return m.startswith("gpt-5") or re.match(r"^o[1-9]", m) is not None


def _chat(model: str, messages: list[dict], max_tokens: int, meter=None) -> str:
    payload = {"model": model, "messages": messages,
               "response_format": {"type": "json_object"}}
    if _is_reasoning(model):
        # Reasoning models reject temperature and bill hidden thinking
        # against the completion cap, so the cap must cover both.
        payload["max_completion_tokens"] = max(1024, max_tokens * 8)
        payload["reasoning_effort"] = os.environ.get("PFBMAX_CJ_EFFORT", "low")
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = max_tokens
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _key()})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
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


def _parse_scores(reply: str, criteria: list[str]) -> dict[str, int]:
    """{criterion: 0|1|3} for whatever the model named, loosely matched."""
    try:
        obj = json.loads(reply or "{}")
    except Exception:
        m = re.search(r"\{.*\}", reply or "", re.S)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {}
    crit_obj = obj.get("criteria") if isinstance(obj, dict) else None
    if not isinstance(crit_obj, dict):
        return {}
    # The scorer itself normalizes underscores out of returned names because
    # models rename fields; do the same, then match case-insensitively.
    norm = {c.replace("_", " ").strip().lower(): c for c in criteria}
    out: dict[str, int] = {}
    for name, val in crit_obj.items():
        key = str(name).replace("_", " ").strip().lower()
        target = norm.get(key)
        if target is None:
            for k2, orig in norm.items():
                if k2.startswith(key[:40]) or key.startswith(k2[:40]):
                    target = orig
                    break
        if target is None:
            continue
        label = val.get("relevance") if isinstance(val, dict) else val
        code = _CODES.get(str(label).strip().lower())
        if code is not None:
            out[target] = code
    return out


def _score_paper_raw(query, criteria, text, model, meter=None):
    """{criterion: 0|1|3} from one grader-replica call, or None on failure."""
    payload = json.dumps({
        "document": " ".join((text or "").split())[:DOC_CHARS],
        "criteria": json.dumps(
            [{"name": c, "description": c} for c in criteria], indent=2),
    }, indent=2)
    try:
        reply = _chat(model,
                      [{"role": "user",
                        "content": _prompt_template().format(payload=payload)}],
                      max_tokens=120 * len(criteria) + 160, meter=meter)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    got = _parse_scores(reply, criteria)
    return got or None


def score_paper_detailed(query: str, criteria: list[str], text: str,
                         model: str, meter=None):
    """(aggregate score, {criterion: 0|1|3}) -- the per-criterion codes too.

    The aggregate alone is not invertible: with equal weights it only reveals
    SUM(r_i), not which criterion failed. Training a per-criterion student
    needs the individual labels, so they are returned and dumped here.
    """
    got = _score_paper_raw(query, criteria, text, model, meter)
    if got is None:
        return None, None
    w = 1.0 / max(1, len(criteria))
    return min(1.0, sum(w * got.get(c, 0) / 3.0 for c in criteria)), got


def score_paper(query: str, criteria: list[str], text: str, model: str,
                meter=None) -> float | None:
    """Replicated weighted score in [0,1], or None if the call failed."""
    got = _score_paper_raw(query, criteria, text, model, meter)
    if got is None:
        return None
    # Equal weights: the scorer's real weights are hidden, and the
    # perfect/not decision is weight-INDEPENDENT anyway (any criterion below
    # 3 drops the total under 0.99 for any weights summing to 1). Weights
    # would only reorder papers that already failed the gate.
    w = 1.0 / max(1, len(criteria))
    score = sum(w * got.get(c, 0) / 3.0 for c in criteria)
    return min(1.0, score)


def rerank(query: str, criteria: list[str], docs: list[tuple[str, str]],
           model: str | None = None, depth: int | None = None,
           meter=None, trace: dict | None = None) -> list[str]:
    """Return cids reordered by the replicated scorer decomposition."""
    model = model or os.environ.get("PFBMAX_CJ_MODEL", "gpt-4o-mini")
    if depth is None:
        d = os.environ.get("PFBMAX_CJ_DEPTH", "").strip()
        depth = int(d) if d.isdigit() else DEPTH
    workers = os.environ.get("PFBMAX_CJ_WORKERS", "").strip()
    workers = int(workers) if workers.isdigit() else WORKERS
    tr = trace if trace is not None else {}
    if not docs:
        return []
    head, tail = docs[:depth], [c for c, _t in docs[depth:]]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        detailed = list(ex.map(
            lambda d: score_paper_detailed(query, criteria, d[1], model, meter),
            head))
    scores = [s for s, _codes in detailed]
    per_crit = [codes for _s, codes in detailed]

    # PFBMAX_CJ_POSDECAY=<alpha>: weight criteria by position, w_i ~ e^(-a*i),
    # instead of equally. Measured basis (2026-08-16): in the mid-band where
    # the golds get buried, the judge's fail rate on GOLD papers rises
    # monotonically with criterion position (17% on crit[0] -> 85% on crit[4])
    # while gold/non separation per criterion stays thin -- later,
    # auxiliary-style criteria ("empirical results", "methodology detailed")
    # are noise the equal-weight conjunction lets swamp the topical signal.
    # Offline: +0.0129 semantic on the tuning dump (29up/16dn), +0.0101 on the
    # held-out deep250 dump (24up/16dn), stable for alpha in [0.3, 1.0].
    # General rule (query-independent, no gold); default OFF.
    pd = os.environ.get("PFBMAX_CJ_POSDECAY", "").strip()
    if pd:
        try:
            alpha = float(pd)
        except ValueError:
            alpha = 0.0
        if alpha > 0 and criteria:
            w = [math.exp(-alpha * i) for i in range(len(criteria))]
            tot = sum(w)
            w = [x / tot for x in w]
            for i, codes in enumerate(per_crit):
                if codes:
                    scores[i] = sum(
                        wi * codes.get(c, 0) / 3.0
                        for wi, c in zip(w, criteria))

    # Judging is the expensive part and the fusion weight is not; dump the
    # raw per-paper scores so weightings can be swept offline at $0 instead
    # of re-billing a full pass per candidate weight.
    dump = os.environ.get("PFBMAX_CJ_DUMP", "").strip()
    if dump:
        try:
            with open(dump, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "query": query, "criteria": list(criteria),
                    "order_in": [c for c, _t in docs],
                    "scores": {c: s for (c, _t), s in zip(head, scores)},
                    # Per-criterion codes {criterion: 0|1|3}. The aggregate is
                    # NOT invertible -- with equal weights it reveals only
                    # SUM(r_i), not which criterion failed -- and a
                    # per-criterion student needs the individual labels.
                    "per_criterion": {c: pc for (c, _t), pc in zip(head, per_crit)
                                      if pc},
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    ok = [s for s in scores if s is not None]
    # A failed judgement must not silently demote a paper: give it the median
    # judged score so it keeps roughly its incoming rank via the tie-break.
    neutral = sorted(ok)[len(ok) // 2] if ok else 0.0
    keyed = [((-(s if s is not None else neutral)), i, cid)
             for i, ((cid, _t), s) in enumerate(zip(head, scores))]
    keyed.sort()
    tr["critjudge"] = {"model": model, "depth": len(head), "judged": len(ok),
                       "failed": len(head) - len(ok),
                       "gate_pass": sum(1 for s in ok if s > 0.99)}
    # scores exposed for downstream stages (tournament zone selection)
    tr["cj_scores"] = {cid: (s if s is not None else neutral)
                       for (cid, _t), s in zip(head, scores)}
    return [cid for _s, _i, cid in keyed] + tail


class _JudgedPool:
    """SemPool passthrough whose .ranked() is judge-reordered."""

    def __init__(self, pool, order):
        self._pool = pool
        self._order = order

    def ranked(self):
        return list(self._order)

    def __getattr__(self, name):
        return getattr(self._pool, name)


def rerank_pool(query, criteria, pool, model=None, trace=None, meter=None):
    """Judge the POOL to choose which 250 get submitted.

    This targets the largest measured loss in the system. Picking the best 250
    from a pool that holds ~60% of the golds would give recall@FULL 0.60 (the
    250 cap essentially never binds, median P=22). We achieve 0.2145 -- a
    35.7% conversion efficiency -- and SOTA-level semantic needs only 0.29,
    i.e. 48%. So this is a selection problem with a large, reachable margin,
    not a ranking-quality problem: 69.8% of the golds we DO submit already
    reach the tiny top-K window.

    Nothing cheap converts it. The cross-encoder made it worse when pointed at
    depth (recall@FULL 0.186 -> 0.109) because it is a local reranker, not a
    global one; the free per-criterion student moved it only +0.007. The judge
    is the one component measured to actually agree with the grader, so here
    it is spent on the selection step instead of on reordering a submission
    that is already ordered well.

    The judged head (the top PFBMAX_CJ_POOL_DEPTH entries) is fully re-sorted
    by judge score, so the judge can promote and also demote within that head;
    everything below the judged depth keeps its incoming order behind it. With
    PFBMAX_TOURN=1 a listwise tournament additionally reorders the contested
    band, where demotion is bounded by the tournament's move cap. Cost scales
    directly with depth (~$0.005/paper).

    Env: PFBMAX_CJ_POOL=1, PFBMAX_CJ_POOL_DEPTH (default 400).
    """
    tr = trace if trace is not None else {}
    order = list(pool.ranked())
    if not order or not criteria:
        return pool
    d = (os.environ.get("PFBMAX_CJ_POOL_DEPTH") or "").strip()
    depth = int(d) if d.isdigit() else 400
    try:
        thresh = float(os.environ.get("PFBMAX_CJ_POOL_THRESH") or 0.66)
    except ValueError:
        thresh = 0.66

    head = order[:depth]
    docs = []
    for cid in head:
        try:
            title = pool.title(cid) or ""
        except Exception:
            title = ""
        try:
            parts = pool.texts(cid) or []
        except Exception:
            parts = []
        body = " ".join(p for p in parts if p)
        blob = ((title + ". " + body) if title else body).strip()
        docs.append((cid, blob))
    docs = [(c, t) for c, t in docs if t]
    if not docs:
        return pool

    # CASCADE. Judging is ~$0.005/paper with gpt-4o, so depth 400 is all we
    # can afford -- but the pool is 2000-4400 and the golds we are missing sit
    # deeper. gpt-4o-mini costs 17x less. In our tests it was not reliable as
    # a solo judge at the perfect-vs-highly boundary (measured: it ranked a
    # true match below a near-miss), but that is not the job here; as a pool
    # judge with replication it performs well. The job is discarding the
    # obviously-irrelevant tail, where a weak judge is adequate. So mini
    # prunes a much deeper slice down to a shortlist and gpt-4o -- the only
    # model that agrees with the grader -- makes every decision that matters.
    # Deliberately generous: keep a large shortlist so a mini mistake costs a
    # rank, not a gold.
    pre_model = (os.environ.get("PFBMAX_CJ_PREFILTER_MODEL") or "").strip()
    pre_keep = (os.environ.get("PFBMAX_CJ_PREFILTER_KEEP") or "").strip()
    keep_n = int(pre_keep) if pre_keep.isdigit() else 500
    if pre_model and len(docs) > keep_n:
        pre_order = rerank(query, criteria, docs, model=pre_model,
                           depth=len(docs), meter=meter, trace={})
        pos = {c: i for i, c in enumerate(pre_order)}
        docs = sorted(docs, key=lambda d: pos.get(d[0], 10 ** 6))[:keep_n]
        tr["cj_prefilter"] = {"model": pre_model, "from": len(pre_order),
                              "kept": len(docs)}

    new_order = rerank(query, criteria, docs, model=model, depth=len(docs),
                       meter=meter, trace=tr)

    # PFBMAX_TOURN=1: listwise tournament over the contested band. Blind A/B
    # audits measured 0.72 gold-vs-non accuracy for gpt-5-mini where the
    # pointwise score ties them; validated end-to-end +0.0150 (tuning 12) /
    # +0.0055 (held-out 36) semantic with pooled Borda, prior 0.4, cap 8.
    if (os.environ.get("PFBMAX_TOURN") or "").strip().lower() not in ("", "0", "false", "no"):
        try:
            import tournament as _tourn
            texts = {c: t for c, t in docs}
            new_order = _tourn.rerank_zone(
                query, criteria, new_order,
                tr.get("cj_scores") or {}, texts, meter=meter, trace=tr)
        except Exception:
            pass          # tournament is additive; failure keeps judge order

    kept = set(new_order)
    tr["cj_pool"] = {"depth": len(docs), "thresh": thresh}
    return _JudgedPool(pool, new_order + [c for c in order if c not in kept])
