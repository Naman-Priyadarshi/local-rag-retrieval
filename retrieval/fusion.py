from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

Ranking = Sequence[tuple[str, float]]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Ranking],
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Fuse ranked lists using score += weight / (rrf_k + 1-based rank)."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not rankings:
        return []

    weights = weights or {}
    fused: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    sequence = 0
    for source, ranking in rankings.items():
        weight = float(weights.get(source, 1.0))
        if weight < 0:
            raise ValueError("Fusion weights must be non-negative")
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            if doc_id not in first_seen:
                first_seen[doc_id] = sequence
                sequence += 1
            fused[doc_id] += weight / (rrf_k + rank)

    ordered = sorted(fused, key=lambda doc_id: (-fused[doc_id], first_seen[doc_id]))
    return [(doc_id, float(fused[doc_id])) for doc_id in ordered[:top_k]]


def distribution_based_score_fusion(
    rankings: Mapping[str, Ranking],
    *,
    weights: Mapping[str, float] | None = None,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Approximate DBSF by normalizing each source using mean +/- 3 std dev.

    Scores are normalized independently per source over that source's returned
    candidate window, clipped to [0, 1], and summed. This is not equivalent to
    fusing every score from a full-corpus index if `rankings` is truncated.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not rankings:
        return []

    weights = weights or {}
    fused: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    sequence = 0

    for source, ranking in rankings.items():
        if not ranking:
            continue
        weight = float(weights.get(source, 1.0))
        if weight < 0:
            raise ValueError("Fusion weights must be non-negative")
        ids = [doc_id for doc_id, _ in ranking]
        values = [float(score) for _, score in ranking]
        mean = sum(values) / len(values)
        variance = sum((score - mean) ** 2 for score in values) / len(values)
        std = variance ** 0.5
        low, high = mean - 3.0 * std, mean + 3.0 * std
        if high <= low:
            min_score, max_score = min(values), max(values)
            if max_score > min_score:
                normalized = [(score - min_score) / (max_score - min_score) for score in values]
            else:
                normalized = [0.0] * len(values)
        else:
            normalized = [min(1.0, max(0.0, (score - low) / (high - low))) for score in values]

        for doc_id, score in zip(ids, normalized, strict=True):
            if doc_id not in first_seen:
                first_seen[doc_id] = sequence
                sequence += 1
            fused[doc_id] += weight * score

    ordered = sorted(fused, key=lambda doc_id: (-fused[doc_id], first_seen[doc_id]))
    return [(doc_id, float(fused[doc_id])) for doc_id in ordered[:top_k]]
