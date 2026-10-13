from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Common interface for interchangeable embedding models."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return one embedding vector per document."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Return a single query embedding vector."""
        ...
