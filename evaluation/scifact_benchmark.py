from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "scifact" / "scifact"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
TOP_K = 10


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
            qid = row["query-id"]
            doc_id = row["corpus-id"]
            qrels.setdefault(qid, {})[doc_id] = int(row["score"])
    return qrels


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int = TOP_K) -> float:
    gains = np.asarray([relevant.get(doc_id, 0) for doc_id in ranked_ids[:k]], dtype=np.float64)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum((np.power(2.0, gains) - 1.0) / discounts))
    ideal_gains = np.asarray(sorted(relevant.values(), reverse=True)[:k], dtype=np.float64)
    ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = float(np.sum((np.power(2.0, ideal_gains) - 1.0) / ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int = TOP_K) -> float:
    relevant_ids = {doc_id for doc_id, score in relevant.items() if score > 0}
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def mrr_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int = TOP_K) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float | int]:
    query_ids = [qid for qid in qrels if qid in run]
    if not query_ids:
        raise ValueError("No overlap between retrieved query IDs and test qrels.")
    return {
        "nDCG@10": float(np.mean([ndcg_at_k(run[qid], qrels[qid]) for qid in query_ids])),
        "Recall@10": float(np.mean([recall_at_k(run[qid], qrels[qid]) for qid in query_ids])),
        "MRR@10": float(np.mean([mrr_at_k(run[qid], qrels[qid]) for qid in query_ids])),
        "evaluated_queries": len(query_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BM25 and SPLADE-Mini on BEIR SciFact.")
    parser.add_argument("--retriever", choices=("bm25", "splade", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=8, help="SPLADE document-encoding batch size")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="SPLADE inference device; CPU is the compatibility-first default")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    for required in (DATA / "corpus.jsonl", DATA / "queries.jsonl", DATA / "qrels" / "test.tsv"):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Expected SciFact under dataset/scifact/scifact/ from your RAG root."
            )

    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    docs = read_jsonl(DATA / "corpus.jsonl")
    query_texts = {row["_id"]: row["text"] for row in read_jsonl(DATA / "queries.jsonl")}
    qrels = load_qrels(DATA / "qrels" / "test.tsv")
    queries = {qid: text for qid, text in query_texts.items() if qid in qrels}

    doc_ids = [str(row["_id"]) for row in docs]
    # Include titles, which are useful evidence-bearing terms in SciFact.
    doc_texts = [
        f"{row.get('title', '').strip()}\n{row.get('text', '').strip()}".strip()
        for row in docs
    ]
    if not queries:
        raise ValueError("No SciFact test queries matched qrels/test.tsv")

    print(f"Documents: {len(docs):,}")
    print(f"Queries with test qrels: {len(queries):,}")
    print(f"Test query-document relevance pairs: {sum(len(v) for v in qrels.values()):,}")

    chosen = ["bm25", "splade"] if args.retriever == "both" else [args.retriever]
    reports: dict[str, Any] = {}

    for name in chosen:
        print(f"\n=== {name.upper()} ===")
        if name == "bm25":
            retriever = BM25Retriever()
            fit_start = time.perf_counter()
            retriever.fit(doc_ids, doc_texts)
            setup_seconds = time.perf_counter() - fit_start
            cache_hit = False
        else:
            # Load the neural-retrieval dependencies only when SPLADE is selected,
            # so BM25-only runs remain lightweight.
            from retrieval.splade import SPLADERetriever

            retriever = SPLADERetriever(CACHE, batch_size=args.batch_size, device=args.device)
            fit_start = time.perf_counter()
            retriever.fit(doc_ids, doc_texts)
            setup_seconds = time.perf_counter() - fit_start
            cache_hit = retriever.cache_hit

        # Warm-up is outside measured query latencies (also ensures lazy setup happens first).
        first_query = next(iter(queries.values()))
        retriever.search(first_query, top_k=args.top_k)

        run: dict[str, list[str]] = {}
        query_latencies: list[float] = []
        started = time.perf_counter()
        for index, (qid, query) in enumerate(queries.items(), start=1):
            query_start = time.perf_counter()
            results = retriever.search(query, top_k=args.top_k)
            query_latencies.append(time.perf_counter() - query_start)
            run[qid] = [doc_id for doc_id, _ in results]
            if index % 50 == 0:
                print(f"Processed {index}/{len(queries)} queries")
        evaluation_seconds = time.perf_counter() - started

        reports[name] = {
            "model": retriever.model_id,
            "metrics": evaluate(run, qrels),
            "setup_seconds_including_model_load": setup_seconds,
            "cache_hit": cache_hit,
            "mean_query_latency_ms_after_warmup": 1000 * float(np.mean(query_latencies)),
            "p95_query_latency_ms_after_warmup": 1000 * float(np.percentile(query_latencies, 95)),
            "total_evaluation_seconds": evaluation_seconds,
        }

    # Merge with previous results so running one retriever does not erase another.
    out_path = RESULTS / "scifact_sparse_baseline.json"
    saved_retrievers = {}

    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
            same_dataset = (
                previous.get("dataset") == "BEIR SciFact"
                and previous.get("documents") == len(docs)
                and previous.get("queries") == len(queries)
            )
            if same_dataset:
                saved_retrievers = previous.get("retrievers", {})
        except (OSError, json.JSONDecodeError):
            print("Previous results could not be read; starting a new report.")

    # Update only the retrievers evaluated in this run.
    saved_retrievers.update(reports)

    output = {
        "dataset": "BEIR SciFact",
        "top_k": TOP_K,
        "documents": len(docs),
        "queries": len(queries),
        "retrievers": saved_retrievers,
        "note": (
            "Local benchmark. Scores depend on tokenization and document formatting; "
            "use the same protocol for every retriever."
        ),
    }

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nCombined results:")
    print(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
