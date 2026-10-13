from __future__ import annotations

import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Small, transparent tokenizer used by this BM25 baseline."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """BM25 baseline exposing a shared fit/search API."""

    name = "bm25"
    model_id = "rank_bm25.BM25Okapi"

    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        self.index: BM25Okapi | None = None

    def fit(self, doc_ids: list[str], documents: list[str]) -> None:
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must have equal lengths")
        if not documents:
            raise ValueError("Cannot index an empty document collection")
        self.doc_ids = list(doc_ids)
        self.index = BM25Okapi([tokenize(text) for text in documents])

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        scores = self.index.get_scores(tokenize(query))
        # Stable tie-breaking follows original corpus order.
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        return [(self.doc_ids[i], float(scores[i])) for i in ranked[:top_k]]
