"""Small, model-agnostic primitives for fusing heterogeneous retrieval lists.

Search backends expose scores on incompatible scales: a full-text ranker score
cannot safely be added to a title/abstract ranker score, and even two rewrites
sent to the same backend can have differently calibrated score ranges.  IRIS
therefore fuses *ranks*, not provider scores, using weighted reciprocal-rank
fusion (RRF).  The implementation is deliberately dependency-free so the same
primitive can be reused by PaperFindingBench, LitQA search, and ScholarQA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class FusionCandidate:
    """Accumulated evidence that one stable document id is worth retrieving."""

    key: str
    score: float = 0.0
    best_rank: int | None = None
    channels: dict[str, int] = field(default_factory=dict)
    families: set[str] = field(default_factory=set)

    @property
    def appearances(self) -> int:
        return len(self.channels)


class ReciprocalRankFusion:
    """Weighted reciprocal-rank fusion with per-channel deduplication.

    ``rank`` is zero-based.  A document contributes at most once per channel;
    if a backend returns a duplicate id, only its best rank counts.  Channel
    names use ``family:variant`` (for example ``snippet:rewrite-2``), which
    makes the trace useful without coupling this class to any particular task.
    """

    def __init__(self, rank_constant: float = 60.0):
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive")
        self.rank_constant = float(rank_constant)
        self._candidates: dict[str, FusionCandidate] = {}

    def add(
        self,
        key: Any,
        *,
        rank: int,
        channel: str,
        weight: float = 1.0,
    ) -> None:
        stable_key = str(key or "").strip()
        stable_channel = str(channel or "").strip()
        if not stable_key or not stable_channel or weight <= 0:
            return
        rank = max(0, int(rank))
        candidate = self._candidates.setdefault(
            stable_key, FusionCandidate(key=stable_key)
        )
        old_rank = candidate.channels.get(stable_channel)
        if old_rank is not None and old_rank <= rank:
            return
        if old_rank is not None:
            candidate.score -= float(weight) / (
                self.rank_constant + old_rank + 1
            )
        candidate.channels[stable_channel] = rank
        candidate.families.add(stable_channel.split(":", 1)[0])
        candidate.score += float(weight) / (self.rank_constant + rank + 1)
        if candidate.best_rank is None or rank < candidate.best_rank:
            candidate.best_rank = rank

    def add_ranking(
        self,
        keys: Iterable[Any],
        *,
        channel: str,
        weight: float = 1.0,
    ) -> None:
        for rank, key in enumerate(keys):
            self.add(key, rank=rank, channel=channel, weight=weight)

    def get(self, key: Any) -> FusionCandidate | None:
        return self._candidates.get(str(key or "").strip())

    def ranked(self) -> list[FusionCandidate]:
        return sorted(
            self._candidates.values(),
            key=lambda item: (
                -item.score,
                item.best_rank if item.best_rank is not None else 10**9,
                item.key,
            ),
        )

    def snapshot(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return a compact JSON-serializable diagnostic view."""

        return [
            {
                "id": item.key,
                "rrf": round(item.score, 8),
                "appearances": item.appearances,
                "families": sorted(item.families),
                "best_rank": item.best_rank,
                "channels": dict(sorted(item.channels.items())),
            }
            for item in self.ranked()[: max(0, int(limit))]
        ]


__all__ = ["FusionCandidate", "ReciprocalRankFusion"]
