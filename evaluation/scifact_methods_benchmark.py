from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.bm25_variants import BM25VariantRetriever, TitleWeightedBM25Retriever
from retrieval.fusion import distribution_based_score_fusion, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "scifact" / "scifact"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
DEFAULT_TOP_K = 10
SUPPORTED_METHODS = {"bm25", "bm25l", "bm25plus", "title_weighted_bm25", "dense", "splade"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected qrels header in {path}")
        for row in reader:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    gains = np.asarray([relevant.get(doc_id, 0) for doc_id in ranked_ids[:k]], dtype=np.float64)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum((np.power(2.0, gains) - 1.0) / discounts))
    ideal = np.asarray(sorted(relevant.values(), reverse=True)[:k], dtype=np.float64)
    idiscounts = np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum((np.power(2.0, ideal) - 1.0) / idiscounts))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    relevant_ids = {doc_id for doc_id, score in relevant.items() if score > 0}
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def mrr_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> dict[str, float | int]:
    query_ids = [qid for qid in qrels if qid in run]
    if not query_ids:
        raise ValueError("No overlap between retrieved query IDs and relevance judgments")
    return {
        f"nDCG@{k}": float(np.mean([ndcg_at_k(run[qid], qrels[qid], k) for qid in query_ids])),
        f"Recall@{k}": float(np.mean([recall_at_k(run[qid], qrels[qid], k) for qid in query_ids])),
        f"MRR@{k}": float(np.mean([mrr_at_k(run[qid], qrels[qid], k) for qid in query_ids])),
        "evaluated_queries": len(query_ids),
    }


def fingerprint(doc_ids: list[str], documents: list[str]) -> str:
    digest = hashlib.sha256()
    for doc_id, document in zip(doc_ids, documents, strict=True):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class DenseQwenRetriever:
    """Qwen3 dense retrieval with a fingerprinted local vector cache."""

    name = "dense"
    model_id = "majentik/Qwen3-Embedding-0.6B-MLX-8bit"

    def __init__(self, cache_dir: Path, batch_size: int = 8) -> None:
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.embedder: Any | None = None
        self.doc_ids: list[str] = []
        self.vectors: np.ndarray | None = None
        self.cache_hit = False

    def fit(self, doc_ids: list[str], documents: list[str]) -> float:
        from embedder.mlx_qwen3 import Qwen3MLXEmbedder

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.doc_ids = list(doc_ids)
        self.embedder = Qwen3MLXEmbedder(batch_size=self.batch_size)
        self.embedder.load_model()
        doc_fingerprint = fingerprint(doc_ids, documents)
        vector_path = self.cache_dir / "scifact_qwen3_mlx_vectors.npy"
        meta_path = self.cache_dir / "scifact_qwen3_mlx_vectors.json"
        expected = {
            "model_id": self.model_id,
            "fingerprint": doc_fingerprint,
            "document_count": len(doc_ids),
            "embedding_dimension": 1024,
        }
        if vector_path.exists() and meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                vectors = np.load(vector_path, mmap_mode="r")
                if metadata == expected and vectors.shape == (len(doc_ids), 1024):
                    self.vectors = vectors
                    self.cache_hit = True
                    print(f"Loaded Qwen3 vector cache: {vector_path}")
                    return 0.0
            except (OSError, ValueError, json.JSONDecodeError):
                print("Qwen3 vector cache is invalid; rebuilding it.")

        print(f"Embedding {len(documents):,} SciFact documents with Qwen3 (one-time cache build)...")
        started = time.perf_counter()
        vectors = self.embedder.embed_documents(documents)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(doc_ids), 1024):
            raise ValueError(f"Unexpected Qwen3 embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Qwen3 embeddings contain NaN or infinity")
        temp_vectors = vector_path.with_suffix(".npy.tmp")
        with temp_vectors.open("wb") as handle:
            np.save(handle, vectors)
        temp_vectors.replace(vector_path)
        temp_meta = meta_path.with_suffix(".json.tmp")
        temp_meta.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        temp_meta.replace(meta_path)
        self.vectors = np.load(vector_path, mmap_mode="r")
        self.cache_hit = False
        return time.perf_counter() - started

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        if self.embedder is None or self.vectors is None:
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        qvec = self.embedder.embed_query(query)
        scores = np.asarray(self.vectors @ qvec, dtype=np.float32)
        top_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in top_indices]


def parse_methods(value: str) -> list[str]:
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not methods:
        raise ValueError("Provide at least one method")
    invalid = sorted(set(methods) - SUPPORTED_METHODS)
    if invalid:
        raise ValueError(f"Unknown methods: {', '.join(invalid)}. Supported: {', '.join(sorted(SUPPORTED_METHODS))}")
    # Preserve user order while removing duplicates.
    return list(dict.fromkeys(methods))


def build_fusion_pairs(methods: list[str]) -> list[tuple[str, str]]:
    available = set(methods)
    candidates = [
        ("bm25", "dense"),
        ("title_weighted_bm25", "dense"),
        ("bm25l", "dense"),
        ("bm25plus", "dense"),
        ("bm25", "splade"),
        ("splade", "dense"),
    ]
    return [pair for pair in candidates if set(pair).issubset(available)]


def load_previous_results(
    path: Path, *, docs: int, queries: int, top_k: int, candidate_k: int, split: str
) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
        if (
            old.get("dataset") == "BEIR SciFact"
            and old.get("documents") == docs
            and old.get("queries_evaluated") == queries
            and old.get("top_k") == top_k
            and old.get("candidate_k") == candidate_k
            and old.get("qrels_split") == split
        ):
            return old
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SciFact retrieval methods and score fusion.")
    parser.add_argument(
        "--methods", default="bm25,bm25l,bm25plus,title_weighted_bm25,dense",
        help="Comma-separated methods: bm25,bm25l,bm25plus,title_weighted_bm25,dense,splade",
    )
    parser.add_argument("--fusion", choices=("none", "rrf", "weighted_rrf", "dbsf", "all"), default="all")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Cutoff used for final evaluation")
    parser.add_argument("--candidate-k", type=int, default=100, help="Candidates retained per retriever before fusion")
    parser.add_argument("--title-weight", type=float, default=0.35, help="Normalized title field weight (0-1)")
    parser.add_argument("--dense-weight", type=float, default=1.5, help="Dense source weight for weighted RRF; other sources weight 1.0")
    parser.add_argument("--batch-size", type=int, default=8, help="Qwen3/SPLADE batch size during indexing")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="SPLADE device if selected")
    parser.add_argument("--split", choices=("test", "train"), default="test", help="Use train for exploratory tuning, test for final evaluation")
    args = parser.parse_args()

    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("Require --top-k >= 1 and --candidate-k >= --top-k")
    if not 0.0 <= args.title_weight <= 1.0:
        parser.error("--title-weight must be between 0 and 1")
    if args.dense_weight < 0:
        parser.error("--dense-weight must be non-negative")
    methods = parse_methods(args.methods)

    qrels_path = DATA / "qrels" / f"{args.split}.tsv"
    for required in (DATA / "corpus.jsonl", DATA / "queries.jsonl", qrels_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing {required}. Expected SciFact under dataset/scifact/scifact/.")

    docs = read_jsonl(DATA / "corpus.jsonl")
    all_queries = {str(row["_id"]): str(row["text"]) for row in read_jsonl(DATA / "queries.jsonl")}
    qrels = load_qrels(qrels_path)
    queries = {qid: query for qid, query in all_queries.items() if qid in qrels}
    if not queries:
        raise ValueError(f"No queries in queries.jsonl overlap with {qrels_path}")

    doc_ids = [str(row["_id"]) for row in docs]
    titles = [str(row.get("title", "")) for row in docs]
    bodies = [str(row.get("text", "")) for row in docs]
    full_texts = [f"{title.strip()}\n{body.strip()}".strip() for title, body in zip(titles, bodies, strict=True)]
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: BEIR SciFact | split={args.split}")
    print(f"Documents: {len(docs):,} | queries with qrels: {len(queries):,} | qrels pairs: {sum(map(len, qrels.values())):,}")
    print(f"Methods: {', '.join(methods)} | candidate_k={args.candidate_k} | evaluation top_k={args.top_k}")
    if args.split == "test":
        print("Reminder: use --split train for tuning weights; reserve test scores for final comparison.")

    retriever_objects: dict[str, Any] = {}
    rankings: dict[str, dict[str, list[tuple[str, float]]]] = {}
    method_reports: dict[str, Any] = {}
    per_query_ms: dict[str, dict[str, float]] = {}

    for method in methods:
        print(f"\n=== Indexing {method} ===")
        started = time.perf_counter()
        cache_hit: bool | None = None

        if method in {"bm25", "bm25l", "bm25plus"}:
            variant = {"bm25": "okapi", "bm25l": "bm25l", "bm25plus": "bm25plus"}[method]
            retriever = BM25VariantRetriever(variant=variant)  # type: ignore[arg-type]
            retriever.fit(doc_ids, full_texts)
            cache_hit = False
        elif method == "title_weighted_bm25":
            retriever = TitleWeightedBM25Retriever(title_weight=args.title_weight)
            retriever.fit(doc_ids, titles, bodies)
            cache_hit = False
        elif method == "dense":
            retriever = DenseQwenRetriever(CACHE, batch_size=args.batch_size)
            retriever.fit(doc_ids, full_texts)
            cache_hit = retriever.cache_hit
        else:  # optional existing SPLADE implementation
            from retrieval.splade import SPLADERetriever

            retriever = SPLADERetriever(CACHE, batch_size=args.batch_size, device=args.device)
            retriever.fit(doc_ids, full_texts)
            cache_hit = retriever.cache_hit

        setup_seconds = time.perf_counter() - started
        retriever_objects[method] = retriever
        rankings[method] = {}
        latencies_ms: list[float] = []

        # Warm-up is not included in reported per-query latency.
        first_query = next(iter(queries.values()))
        retriever.search(first_query, top_k=args.candidate_k)
        for index, (qid, query) in enumerate(queries.items(), start=1):
            tick = time.perf_counter()
            results = retriever.search(query, top_k=args.candidate_k)
            elapsed_ms = (time.perf_counter() - tick) * 1000.0
            rankings[method][qid] = results
            latencies_ms.append(elapsed_ms)
            if index % 100 == 0:
                print(f"{method}: processed {index}/{len(queries)} queries")

        per_query_ms[method] = {qid: ms for qid, ms in zip(queries.keys(), latencies_ms, strict=True)}
        run_ids = {qid: [doc_id for doc_id, _ in ranking] for qid, ranking in rankings[method].items()}
        method_reports[method] = {
            "model": getattr(retriever, "model_id", method),
            "metrics": evaluate(run_ids, qrels, args.top_k),
            "setup_seconds": setup_seconds,
            "cache_hit": cache_hit,
            "mean_query_latency_ms": float(np.mean(latencies_ms)),
            "p95_query_latency_ms": float(np.percentile(latencies_ms, 95)),
            "configuration": ({"title_weight": args.title_weight} if method == "title_weighted_bm25" else {}),
        }

    if args.fusion == "none":
        fusion_modes: list[str] = []
    elif args.fusion == "all":
        fusion_modes = ["rrf", "weighted_rrf", "dbsf"]
    else:
        fusion_modes = [args.fusion]

    fusion_reports: dict[str, Any] = {}
    fusion_pairs = build_fusion_pairs(methods) if fusion_modes else []
    if fusion_modes and not fusion_pairs:
        print("\nNo compatible fusion pair selected. Include `dense` and/or `splade` to fuse independent retrievers.")

    for left, right in fusion_pairs:
        pair_label = f"{left}+{right}"
        for fusion_mode in fusion_modes:
            key = f"{left}__{right}__{fusion_mode}"
            print(f"\n=== Fusion {pair_label}: {fusion_mode} ===")
            fused_run: dict[str, list[str]] = {}
            fusion_ms: list[float] = []
            end_to_end_ms: list[float] = []
            for qid in queries:
                source_rankings = {left: rankings[left][qid], right: rankings[right][qid]}
                tick = time.perf_counter()
                if fusion_mode == "rrf":
                    fused = reciprocal_rank_fusion(source_rankings, rrf_k=60, top_k=args.candidate_k)
                elif fusion_mode == "weighted_rrf":
                    weights = {left: 1.0, right: 1.0}
                    if right == "dense":
                        weights[right] = args.dense_weight
                    elif left == "dense":
                        weights[left] = args.dense_weight
                    fused = reciprocal_rank_fusion(source_rankings, weights=weights, rrf_k=60, top_k=args.candidate_k)
                else:
                    fused = distribution_based_score_fusion(source_rankings, top_k=args.candidate_k)
                merge_ms = (time.perf_counter() - tick) * 1000.0
                fusion_ms.append(merge_ms)
                base_ms = per_query_ms[left][qid] + per_query_ms[right][qid]
                end_to_end_ms.append(base_ms + merge_ms)
                fused_run[qid] = [doc_id for doc_id, _ in fused]

            config: dict[str, Any] = {"sources": [left, right], "rrf_k": 60, "candidate_k": args.candidate_k}
            if fusion_mode == "weighted_rrf":
                config["weights"] = {
                    left: args.dense_weight if left == "dense" else 1.0,
                    right: args.dense_weight if right == "dense" else 1.0,
                }
            fusion_reports[key] = {
                "fusion": fusion_mode,
                "metrics": evaluate(fused_run, qrels, args.top_k),
                "configuration": config,
                "mean_fusion_latency_ms": float(np.mean(fusion_ms)),
                "mean_end_to_end_query_latency_ms": float(np.mean(end_to_end_ms)),
            }

    output_path = RESULTS / "scifact_methods_benchmark.json"
    previous = load_previous_results(
        output_path,
        docs=len(docs),
        queries=len(queries),
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        split=args.split,
    )
    merged_methods = previous.get("methods", {})
    merged_fusions = previous.get("fusion_runs", {})
    old_method_reports = dict(merged_methods)

    # Invalidate old fusion metrics only if an input retriever's configuration
    # actually changed. This lets a lexical-only run with --fusion none preserve
    # valid hybrid results already saved from an earlier run.
    changed_methods = {
        name for name, report in method_reports.items()
        if name not in old_method_reports
        or old_method_reports[name].get("model") != report.get("model")
        or old_method_reports[name].get("configuration", {}) != report.get("configuration", {})
    }
    for key, record in list(merged_fusions.items()):
        sources = set(record.get("configuration", {}).get("sources", []))
        if sources.intersection(changed_methods):
            del merged_fusions[key]
    merged_methods.update(method_reports)
    merged_fusions.update(fusion_reports)

    output = {
        "dataset": "BEIR SciFact",
        "qrels_split": args.split,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "documents": len(docs),
        "queries_evaluated": len(queries),
        "methods": merged_methods,
        "fusion_runs": merged_fusions,
        "notes": [
            "Results are merged by method; rerunning one method does not erase unrelated method results.",
            "Use the training split for weight tuning and the test split for final comparison.",
            "Title-weighted BM25 independently min-max-normalizes title and abstract scores per query; default title weight is 0.35.",
            "Weighted RRF uses weight 1.0 for the lexical/SPLADE source and --dense-weight for dense Qwen3 when present.",
            "DBSF approximates distribution-based score fusion over each retriever's top candidate_k results, not full-corpus score distributions.",
            "Dense query latency includes Qwen3 query embedding; fusion end-to-end latency adds the measured component query latencies and fusion overhead.",
        ],
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nSummary:")
    for name, report in method_reports.items():
        print(f"{name:24s} nDCG@{args.top_k}={report['metrics'][f'nDCG@{args.top_k}']:.4f}  "
              f"Recall@{args.top_k}={report['metrics'][f'Recall@{args.top_k}']:.4f}  "
              f"MRR@{args.top_k}={report['metrics'][f'MRR@{args.top_k}']:.4f}  "
              f"latency={report['mean_query_latency_ms']:.1f} ms")
    for name, report in fusion_reports.items():
        print(f"{name:44s} nDCG@{args.top_k}={report['metrics'][f'nDCG@{args.top_k}']:.4f}  "
              f"Recall@{args.top_k}={report['metrics'][f'Recall@{args.top_k}']:.4f}  "
              f"MRR@{args.top_k}={report['metrics'][f'MRR@{args.top_k}']:.4f}")
    print(f"\nSaved combined results to {output_path}")


if __name__ == "__main__":
    main()
