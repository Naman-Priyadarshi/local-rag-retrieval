from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
from sentence_transformers import SparseEncoder


class SPLADERetriever:
    """SPLADE-Mini sparse retriever with a persistent document-vector cache.

    The stored representations remain sparse. Do not densify the 30,522-wide
    token vectors: that would waste memory and disk space.
    """

    name = "splade"
    model_id = "rasyosef/splade-mini"
    cache_version = 1

    def __init__(
        self,
        cache_dir: Path,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be 'cpu' or 'mps'")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.device = device
        self.model: SparseEncoder | None = None
        self.doc_ids: list[str] = []
        self.doc_embeddings: Any | None = None
        self.cache_hit = False

    def _load_model(self) -> SparseEncoder:
        if self.model is None:
            self.model = SparseEncoder(
                self.model_id,
                device=self.device,
                similarity_fn_name="dot",
            )
            self.model.eval()
            # The model card specifies 512 tokens. SciFact abstracts are short;
            # retain that limit to avoid unnecessary sequence processing.
            self.model.max_seq_length = min(self.model.max_seq_length, 512)
            # SPLADE scores are learned term weights; use dot product, not cosine.
            self.model.similarity_fn_name = "dot"
        return self.model

    @staticmethod
    def _fingerprint(doc_ids: list[str], documents: list[str]) -> str:
        digest = hashlib.sha256()
        for doc_id, text in zip(doc_ids, documents, strict=True):
            digest.update(doc_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def fit(self, doc_ids: list[str], documents: list[str]) -> float:
        """Load or build the sparse corpus index; return encoding seconds.

        Model-loading time is not included in the returned value. The caller can
        time the complete fit() call separately if desired.
        """
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must have equal lengths")
        if not documents:
            raise ValueError("Cannot index an empty document collection")

        self.doc_ids = list(doc_ids)
        model = self._load_model()
        fingerprint = self._fingerprint(doc_ids, documents)
        cache_path = self.cache_dir / "scifact_splade_mini.pt"
        metadata_path = self.cache_dir / "scifact_splade_mini.json"
        expected_metadata = {
            "cache_version": self.cache_version,
            "model_id": self.model_id,
            "fingerprint": fingerprint,
            "document_count": len(documents),
            "device_used_for_encoding": self.device,
        }

        if cache_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                # Caches are local files created by this program.
                try:
                    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                except TypeError:  # compatibility with older PyTorch releases
                    cached = torch.load(cache_path, map_location="cpu")
                if metadata == expected_metadata and tuple(cached.shape)[0] == len(doc_ids):
                    self.doc_embeddings = cached
                    self.cache_hit = True
                    print(f"Loaded cached SPLADE representations: {cache_path}")
                    return 0.0
            except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f"Could not reuse SPLADE cache ({exc}); rebuilding it.")

        print(f"Encoding {len(documents)} documents with {self.model_id}...")
        started = time.perf_counter()
        with torch.inference_mode():
            self.doc_embeddings = model.encode_document(
                documents,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                convert_to_sparse_tensor=True,
                save_to_cpu=True,
            )
        elapsed = time.perf_counter() - started

        # Write cache atomically, so an interrupted run will not leave a valid-
        # looking partial cache behind.
        temporary_cache = cache_path.with_suffix(".pt.tmp")
        torch.save(self.doc_embeddings, temporary_cache)
        temporary_cache.replace(cache_path)
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
        temporary_metadata.replace(metadata_path)
        self.cache_hit = False
        print(f"Saved sparse cache: {cache_path}")
        return elapsed

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.doc_embeddings is None:
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query.strip():
            raise ValueError("query must not be empty")

        model = self._load_model()
        with torch.inference_mode():
            query_embedding = model.encode_query(
                [query],
                convert_to_tensor=True,
                convert_to_sparse_tensor=True,
                save_to_cpu=True,
            )
            doc_embeddings = self.doc_embeddings

        if doc_embeddings is None:
            raise RuntimeError("Document embeddings have not been initialized.")

        scores_tensor = model.similarity(
            cast(torch.Tensor, query_embedding),
            cast(torch.Tensor, doc_embeddings),
        )
        scores = scores_tensor.detach().cpu().reshape(-1).tolist()
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        return [(self.doc_ids[i], float(scores[i])) for i in ranked[:top_k]]
