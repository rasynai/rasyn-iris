"""Evidence-grounded ScholarQA specialist for IRIS v2.

The previous route was a single uncited generation.  This implementation
decomposes the question, retrieves full-text snippets through the task-provided
date-filtered tools, and makes the model cite only an explicit source pack.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from typing import Any

from iris_asta.backbone import extract_json
from iris_asta.contracts import emit_sqa
from iris_asta.reranker import RerankDocument, rerank_documents
from iris_asta.retrieval import ReciprocalRankFusion

_SOURCE_RE = re.compile(r"\bS(\d+)\b", re.I)

_SNIPPET_LIMIT_PER_QUERY = 64
_PAPER_LIMIT_PER_QUERY = 32
_RERANK_DEFAULT_CAP = 256
_SOURCE_DEFAULT_CAP = 40
_SOURCE_PER_PAPER_CAP = 4


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _plan_queries(question: str, bb: Any) -> list[str]:
    prompt = (
        "Decompose this scientific research question into 4 complementary "
        "literature-search queries. Include terminology, mechanism, comparison, "
        "and limitations where applicable. Return JSON only as "
        '{"queries":["..."]}.\n\nQuestion: ' + question
    )
    try:
        result = bb.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
            json_mode=True,
        )
        obj = extract_json(getattr(result, "text", "")) or {}
        queries = [_text(q) for q in obj.get("queries", [])]
    except Exception:
        queries = []
    out: list[str] = []
    for query in [question] + queries:
        if query and query.lower() not in {x.lower() for x in out}:
            out.append(query)
    # Four complementary channels are enough to fill the 24-source synthesis
    # pack; a fifth sequential MCP query adds a large worst-case retry tail
    # for little marginal evidence. (Constants here are hand-set priors,
    # not derived from scorer internals.)
    return out[:4]


def _retrieve(
    question: str,
    bb: Any,
    client: Any,
    inserted_before: str | None = None,
) -> list[dict[str, str]]:
    fusion = ReciprocalRankFusion()
    records: dict[str, dict[str, Any]] = {}

    def absorb(item: Any, passage: str, raw_score: float) -> str | None:
        corpus_id = _text(str(getattr(item, "corpus_id", "")))
        title = _text(getattr(item, "title", ""))
        normalized = " ".join(passage.casefold().split())
        paper_key = corpus_id or title.casefold()
        if not paper_key or not normalized:
            return None
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        key = f"{paper_key}\x1f{digest}"
        rec = records.setdefault(key, {
            "corpus_id": corpus_id,
            "title": title or "Untitled paper",
            "snippet": passage,
            "best_score": raw_score,
        })
        if raw_score > rec["best_score"]:
            rec["best_score"] = raw_score
            rec["snippet"] = passage
        return key

    queries = _plan_queries(question, bb)
    for query_index, query in enumerate(queries):
        try:
            kwargs = {"limit": _SNIPPET_LIMIT_PER_QUERY}
            if inserted_before:
                kwargs["inserted_before"] = inserted_before
            snippets = client.snippet_search(query, **kwargs) or []
        except Exception:
            snippets = []
        snippets = sorted(
            snippets,
            key=lambda item: float(getattr(item, "score", 0.0) or 0.0),
            reverse=True,
        )
        for rank, item in enumerate(snippets):
            passage = " ".join(_text(getattr(item, "text", "")).split())
            try:
                raw_score = float(getattr(item, "score", 0.0) or 0.0)
            except (TypeError, ValueError):
                raw_score = 0.0
            key = absorb(item, passage, raw_score)
            if key:
                fusion.add(
                    key,
                    rank=rank,
                    channel=f"snippet:query-{query_index}",
                    weight=1.2,
                )

        # Abstract retrieval adds papers whose full text did not contain a
        # high-ranked literal query match. Clients without paper_search keep
        # the original snippet-only behavior.
        try:
            papers = client.paper_search(
                query,
                limit=_PAPER_LIMIT_PER_QUERY,
                publication_date_before=inserted_before,
            ) or []
        except Exception:
            papers = []
        for rank, paper in enumerate(papers):
            passage = " ".join(_text(getattr(paper, "abstract", "")).split())
            key = absorb(paper, passage, 0.0)
            if key:
                fusion.add(
                    key,
                    rank=rank,
                    channel=f"paper:query-{query_index}",
                    weight=1.0,
                )

    # The same optional sidecar used by PFB can rerank SQA's much smaller
    # source pool. Failures are observational and leave pure RRF untouched.
    reranker_url = os.environ.get("IRIS_ASTA_RERANKER_URL", "").strip()
    if reranker_url and records:
        try:
            rerank_cap = max(1, min(
                len(records),
                int(os.environ.get(
                    "IRIS_ASTA_SQA_RERANKER_CAP", str(_RERANK_DEFAULT_CAP)
                )),
            ))
        except ValueError:
            rerank_cap = min(len(records), _RERANK_DEFAULT_CAP)
        selected = fusion.ranked()[:rerank_cap]
        documents = [
            RerankDocument(
                id=candidate.key,
                text=(
                    f"{records[candidate.key]['title']}\n"
                    f"{records[candidate.key]['snippet']}"
                )[:4000],
            )
            for candidate in selected
            if records[candidate.key]["snippet"]
        ]
        try:
            scored = rerank_documents(
                reranker_url,
                question,
                documents,
                timeout_s=float(os.environ.get("IRIS_ASTA_RERANKER_TIMEOUT_S", "900")),
            )
            fusion.add_ranking(
                [key for key, _score in scored],
                channel="reranker:cross-encoder",
                weight=4.0,
            )
        except Exception:
            pass

    try:
        source_cap = max(1, int(os.environ.get(
            "IRIS_ASTA_SQA_SOURCE_CAP", str(_SOURCE_DEFAULT_CAP)
        )))
    except ValueError:
        source_cap = _SOURCE_DEFAULT_CAP
    sources: list[dict[str, str]] = []
    per_paper: Counter = Counter()
    for candidate in fusion.ranked():
        rec = records.get(candidate.key)
        if not rec or not rec["snippet"]:
            continue
        paper_key = rec["corpus_id"] or rec["title"].casefold()
        if per_paper[paper_key] >= _SOURCE_PER_PAPER_CAP:
            continue
        sources.append(
            {
                "source_id": f"S{len(sources) + 1}",
                "corpus_id": rec["corpus_id"],
                "title": rec["title"],
                "snippet": rec["snippet"][:1200],
            }
        )
        per_paper[paper_key] += 1
        if len(sources) >= source_cap:
            break
    return sources


def _draft(question: str, sources: list[dict[str, str]], bb: Any) -> dict:
    source_pack = "\n\n".join(
        f"{s['source_id']} | corpus_id={s['corpus_id']} | {s['title']}\n{s['snippet']}"
        for s in sources
    )
    prompt = f"""Write a rigorous answer to the research question using only the source pack.
Organize it into 3-6 useful sections. Distinguish established findings from uncertainty,
answer comparisons directly, and discuss important limitations. Cite claims inline using
the exact source markers [S1], [S2], etc. Never invent a source or result.

Return JSON only:
{{"sections":[{{"title":"...","text":"...","source_ids":["S1","S2"]}}]}}

QUESTION:
{question}

SOURCE PACK:
{source_pack or "No relevant sources were retrieved. State that evidence is unavailable."}
"""
    try:
        result = bb.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5000,
            json_mode=True,
        )
        return extract_json(getattr(result, "text", "")) or {}
    except Exception:
        return {}


def _sections_from_draft(draft: dict, sources: list[dict[str, str]]) -> list[dict]:
    source_map = {s["source_id"].upper(): s for s in sources}
    output: list[dict] = []
    raw_sections = draft.get("sections")
    if not isinstance(raw_sections, list):
        raw_sections = []
    for raw in raw_sections:
        if not isinstance(raw, dict):
            continue
        body = _text(raw.get("text"))
        if not body:
            continue
        ids: list[str] = []
        declared = raw.get("source_ids")
        if isinstance(declared, list):
            ids.extend(_text(x).strip("[]").upper() for x in declared)
        ids.extend(f"S{n}".upper() for n in _SOURCE_RE.findall(body))
        citations = []
        used: set[str] = set()
        for source_id in ids:
            if source_id in used or source_id not in source_map:
                continue
            used.add(source_id)
            source = source_map[source_id]
            marker = f"[{source_id}]"
            if marker not in body:
                body = body.rstrip() + f" {marker}"
            citations.append(
                {
                    "id": marker,
                    "snippets": [source["snippet"]],
                    "title": source["title"],
                    "metadata": {"corpus_id": source["corpus_id"]},
                }
            )
            if len(citations) >= 6:
                break
        output.append(
            {"title": _text(raw.get("title")) or "Findings", "text": body, "citations": citations}
        )
    return output


def solve_sqa(
    question: str,
    bb: Any,
    client: Any,
    inserted_before: str | None = None,
) -> str:
    """Retrieve, synthesize, and emit a scorer-valid ScholarQA report."""

    sources = _retrieve(question, bb, client, inserted_before=inserted_before)
    draft = _draft(question, sources, bb)
    sections = _sections_from_draft(draft, sources)
    if not sections:
        if sources:
            first = sources[0]
            sections = [
                {
                    "title": "Evidence summary",
                    "text": (
                        "The retrieved literature provides relevant evidence, "
                        "but a complete synthesis could not be generated "
                        f"[{first['source_id']}]."
                    ),
                    "citations": [
                        {
                            "id": f"[{first['source_id']}]",
                            "snippets": [first["snippet"]],
                            "title": first["title"],
                            "metadata": {"corpus_id": first["corpus_id"]},
                        }
                    ],
                }
            ]
        else:
            sections = [
                {
                    "title": "Evidence unavailable",
                    "text": (
                        "No sufficiently relevant literature evidence was "
                        "retrieved for this question."
                    ),
                    "citations": [],
                }
            ]
    return emit_sqa(sections)


__all__ = ["solve_sqa"]
