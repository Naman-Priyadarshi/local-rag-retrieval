from __future__ import annotations

import re
from typing import Literal

import numpy as np
from rank_bm25 import BM25L, BM25Okapi, BM25Plus

BM25Variant = Literal["okapi", "bm25l", "bm25plus"]


def tokenize(text: str) -> list[str]:
    """Transparent word tokenizer shared by all BM25 variants."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25VariantRetriever:
    """A common API for BM25Okapi, BM25L, and BM25Plus."""

    def __init__(self, variant: BM25Variant = "okapi") -> None:
        if variant not in {"okapi", "bm25l", "bm25plus"}:
            raise ValueError(f"Unsupported BM25 variant: {variant}")
        self.variant = variant
        self.name = f"bm25_{variant}"
        self.model_id = {
            "okapi": "rank_bm25.BM25Okapi",
            "bm25l": "rank_bm25.BM25L",
            "bm25plus": "rank_bm25.BM25Plus",
        }[variant]
        self.doc_ids: list[str] = []
        self.index: BM25Okapi | BM25L | BM25Plus | None = None

    def fit(self, doc_ids: list[str], documents: list[str]) -> None:
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must have equal lengths")
        if not documents:
            raise ValueError("Cannot index an empty document collection")
        tokenized = [tokenize(text) for text in documents]
        index_class = {
            "okapi": BM25Okapi,
            "bm25l": BM25L,
            "bm25plus": BM25Plus,
        }[self.variant]
        self.index = index_class(tokenized)
        self.doc_ids = list(doc_ids)

    def score_all(self, query: str) -> np.ndarray:
        if self.index is None:
            raise RuntimeError("Call fit() before score_all().")
        if not query.strip():
            raise ValueError("query must not be empty")
        return np.asarray(self.index.get_scores(tokenize(query)), dtype=np.float64)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        scores = self.score_all(query)
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in ranked]


class TitleWeightedBM25Retriever:
    """Fielded BM25: normalize title/body scores separately, then weight them.

    Title scores are min-max scaled per query before combining so the title's
    shorter length and score scale do not accidentally dominate the abstract.
    `title_weight=0.35` means 35% title score and 65% abstract score.
    """

    name = "title_weighted_bm25"

    def __init__(self, title_weight: float = 0.35, variant: BM25Variant = "okapi") -> None:
        if not 0.0 <= title_weight <= 1.0:
            raise ValueError("title_weight must be between 0 and 1")
        self.title_weight = float(title_weight)
        self.variant = variant
        self.model_id = f"FieldedBM25({variant}, title_weight={self.title_weight:.2f})"
        self.doc_ids: list[str] = []
        self.title_index = BM25VariantRetriever(variant)
        self.body_index = BM25VariantRetriever(variant)

    def fit(self, doc_ids: list[str], titles: list[str], bodies: list[str]) -> None:
        if not (len(doc_ids) == len(titles) == len(bodies)):
            raise ValueError("doc_ids, titles, and bodies must have equal lengths")
        self.doc_ids = list(doc_ids)
        self.title_index.fit(doc_ids, titles)
        self.body_index.fit(doc_ids, bodies)

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        low = float(np.min(scores))
        high = float(np.max(scores))
        if high <= low:
            return np.zeros_like(scores, dtype=np.float64)
        return (scores - low) / (high - low)

    def score_all(self, query: str) -> np.ndarray:
        title_scores = self._minmax(self.title_index.score_all(query))
        body_scores = self._minmax(self.body_index.score_all(query))
        return self.title_weight * title_scores + (1.0 - self.title_weight) * body_scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        scores = self.score_all(query)
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in ranked]
