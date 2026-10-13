from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_embeddings import load


MODEL_ID = "majentik/Qwen3-Embedding-0.6B-MLX-8bit"

DEFAULT_QUERY_INSTRUCTION = (
    "Given a query, retrieve relevant passages that answer the query"
)


class Qwen3MLXEmbedder:
    """Qwen3-Embedding running locally with mlx-embeddings."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 8,
        max_length: int = 8192,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        self.model_id = model_id
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.max_length = max_length
        self.dimension = 1024

        # Keep model and tokenizer explicitly nullable until loaded.
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def load_model(self) -> None:
        """Load the model once and reuse it."""
        if self.model is not None and self.tokenizer is not None:
            return

        model, tokenizer = load(self.model_id)

        # Assign only after both objects have been returned successfully.
        self.model = model
        self.tokenizer = tokenizer

    def _get_loaded_components(self) -> tuple[Any, Any]:
        """Return initialized components with a clear runtime guarantee."""
        self.load_model()

        model = self.model
        tokenizer = self.tokenizer

        if model is None or tokenizer is None:
            raise RuntimeError(
                f"Failed to initialize embedding model: {self.model_id}"
            )

        return model, tokenizer

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode text batches and return a float32 NumPy matrix."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model, tokenizer = self._get_loaded_components()
        batches: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="mlx",
            )

            outputs = model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            embeddings = outputs.text_embeds
            mx.eval(embeddings)

            batch_vectors = np.asarray(
                embeddings.tolist(),
                dtype=np.float32,
            )

            if batch_vectors.ndim != 2:
                raise RuntimeError(
                    "Expected a 2D embedding matrix, got shape "
                    f"{batch_vectors.shape}"
                )

            if batch_vectors.shape[1] != self.dimension:
                raise RuntimeError(
                    f"Expected {self.dimension} embedding dimensions, "
                    f"got {batch_vectors.shape[1]}"
                )

            if not np.isfinite(batch_vectors).all():
                raise RuntimeError("Embedding output contains NaN or infinity")

            batches.append(batch_vectors)

        return np.concatenate(batches, axis=0)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed document chunks without a query instruction."""
        return self._embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query using a retrieval instruction."""
        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        formatted_query = (
            f"Instruct: {self.query_instruction}\n"
            f"Query: {query}"
        )

        return self._embed([formatted_query])[0]

    def benchmark(self, texts: list[str]) -> dict[str, Any]:
        """Measure warm document throughput and query latency."""
        if not texts:
            raise ValueError("Provide at least one text to benchmark")

        self.load_model()

        # Warm up before timing. Model loading is excluded from measurements.
        self.embed_documents(texts[:1])

        started = time.perf_counter()
        self.embed_documents(texts)
        document_seconds = time.perf_counter() - started

        if document_seconds <= 0:
            raise RuntimeError("Invalid benchmark duration")

        query_times: list[float] = []

        for text in texts[: min(10, len(texts))]:
            started = time.perf_counter()
            self.embed_query(text)
            query_times.append(time.perf_counter() - started)

        return {
            "model": self.model_id,
            "documents": len(texts),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "embedding_dimension": self.dimension,
            "document_seconds": document_seconds,
            "documents_per_second": len(texts) / document_seconds,
            "mean_query_latency_ms": (
                1000 * sum(query_times) / len(query_times)
            ),
        }
