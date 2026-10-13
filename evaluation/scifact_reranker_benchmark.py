from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.bm25_variants import BM25VariantRetriever
from retrieval.fusion import reciprocal_rank_fusion
from evaluation.scifact_methods_benchmark import (
    CACHE,
    DATA,
    RESULTS,
    DenseQwenRetriever,
    evaluate,
    load_qrels,
    read_jsonl,
    recall_at_k,
)

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a cross-encoder reranker on top of BM25Plus + Qwen3 weighted-RRF "
            "candidates for BEIR SciFact. Writes to a separate result file."
        )
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--top-k", type=int, default=10, help="Evaluation cutoff")
    parser.add_argument("--candidate-k", type=int, default=200,
                        help="Candidate depth from each first-stage retriever and fusion")
    parser.add_argument("--dense-weight", type=float, default=7.0,
                        help="Dense weight for weighted RRF; BM25Plus weight remains 1.0")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--rerank-k", type=int, default=None,
                        help="Number of fused candidates to score; defaults to candidate-k")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Cross-encoder inference batch size")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum query-document token length for the cross-encoder")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="Inference device; start with cpu for predictable memory use")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Optional query limit for a quick smoke test; do not use for final metrics")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: results/scifact_reranker_benchmark.json)")
    args = parser.parse_args()

    rerank_k = args.candidate_k if args.rerank_k is None else args.rerank_k
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be >= --top-k")
    if not 1 <= rerank_k <= args.candidate_k:
        parser.error("--rerank-k must be between 1 and --candidate-k")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_length < 16:
        parser.error("--max-length must be at least 16")
    if args.dense_weight < 0:
        parser.error("--dense-weight must be non-negative")
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be at least 1")

    qrels_path = DATA / "qrels" / f"{args.split}.tsv"
    for required in (DATA / "corpus.jsonl", DATA / "queries.jsonl", qrels_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Expected SciFact under dataset/scifact/scifact/."
            )

    docs = read_jsonl(DATA / "corpus.jsonl")
    all_queries = {str(row["_id"]): str(row["text"]) for row in read_jsonl(DATA / "queries.jsonl")}
    qrels = load_qrels(qrels_path)
    queries = {qid: query for qid, query in all_queries.items() if qid in qrels}
    if args.max_queries is not None:
        queries = dict(list(queries.items())[:args.max_queries])
        qrels = {qid: qrels[qid] for qid in queries}
    if not queries:
        raise ValueError(f"No queries in queries.jsonl overlap with {qrels_path}")

    doc_ids = [str(row["_id"]) for row in docs]
    titles = [str(row.get("title", "")) for row in docs]
    bodies = [str(row.get("text", "")) for row in docs]
    full_texts = [
        f"{title.strip()}\n{body.strip()}".strip()
        for title, body in zip(titles, bodies, strict=True)
    ]
    id_to_text = dict(zip(doc_ids, full_texts, strict=True))

    print(f"Dataset: BEIR SciFact | split={args.split}")
    print(
        f"Documents: {len(docs):,} | queries evaluated: {len(queries):,} | "
        f"qrels pairs in evaluated queries: {sum(map(len, qrels.values())):,}"
    )
    print(
        f"First stage: BM25Plus + Qwen3 weighted RRF | candidate_k={args.candidate_k} | "
        f"dense_weight={args.dense_weight:g}"
    )
    print(
        f"Cross-encoder: {args.reranker_model} | rerank_k={rerank_k} | "
        f"batch_size={args.batch_size} | max_length={args.max_length} | device={args.device}"
    )
    if args.max_queries is not None:
        print("WARNING: --max-queries is a smoke test only; do not report its metrics as benchmark results.")
    if args.split == "test":
        print("Reminder: choose settings on train; test scores are exploratory after repeated test comparisons.")

    # First-stage BM25Plus retrieval.
    print("\n=== Indexing BM25Plus ===")
    bm25 = BM25VariantRetriever(variant="bm25plus")
    started = time.perf_counter()
    bm25.fit(doc_ids, full_texts)
    bm25_setup_s = time.perf_counter() - started

    # First-stage Qwen3 retrieval. This reuses the existing vector cache when valid.
    print("\n=== Loading Qwen3 dense retriever ===")
    dense = DenseQwenRetriever(CACHE, batch_size=8)
    started = time.perf_counter()
    dense.fit(doc_ids, full_texts)
    dense_setup_s = time.perf_counter() - started
    print(f"Qwen3 vector cache hit: {dense.cache_hit}")

    query_ids = list(queries)
    candidate_run: dict[str, list[str]] = {}
    source_latency_ms: dict[str, list[float]] = {"bm25plus": [], "dense": []}
    fusion_latency_ms: list[float] = []

    print("\n=== Retrieving hybrid candidates ===")
    first_query = next(iter(queries.values()))
    bm25.search(first_query, top_k=args.candidate_k)
    dense.search(first_query, top_k=args.candidate_k)

    for i, (qid, query) in enumerate(queries.items(), start=1):
        tick = time.perf_counter()
        bm25_results = bm25.search(query, top_k=args.candidate_k)
        source_latency_ms["bm25plus"].append((time.perf_counter() - tick) * 1000.0)

        tick = time.perf_counter()
        dense_results = dense.search(query, top_k=args.candidate_k)
        source_latency_ms["dense"].append((time.perf_counter() - tick) * 1000.0)

        tick = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"bm25plus": bm25_results, "dense": dense_results},
            weights={"bm25plus": 1.0, "dense": args.dense_weight},
            rrf_k=60,
            top_k=args.candidate_k,
        )
        fusion_latency_ms.append((time.perf_counter() - tick) * 1000.0)
        candidate_run[qid] = [doc_id for doc_id, _ in fused]
        if i % 50 == 0 or i == len(queries):
            print(f"Hybrid retrieval: processed {i}/{len(queries)} queries")

    baseline_metrics = evaluate(candidate_run, qrels, args.top_k)
    candidate_recall = float(np.mean([
        recall_at_k(candidate_run[qid], qrels[qid], args.candidate_k)
        for qid in query_ids
    ]))

    # Load the cross-encoder only after candidate generation/cache loading has finished.
    print("\n=== Loading cross-encoder ===")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install it in your .env with: "
            "python -m pip install sentence-transformers"
        ) from exc

    started = time.perf_counter()
    reranker = CrossEncoder(
        args.reranker_model,
        max_length=args.max_length,
        device=args.device,
    )
    reranker_setup_s = time.perf_counter() - started

    reranked_run: dict[str, list[str]] = {}
    rerank_ms_by_query: list[float] = []
    print("\n=== Reranking hybrid candidates ===")
    for i, qid in enumerate(query_ids, start=1):
        query = queries[qid]
        base_ids = candidate_run[qid]
        to_score = base_ids[:rerank_k]
        pairs = [(query, id_to_text[doc_id]) for doc_id in to_score]
        tick = time.perf_counter()
        scores = reranker.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        score_values = np.asarray(scores).reshape(-1)
        if len(score_values) != len(to_score):
            raise RuntimeError(
                f"Cross-encoder returned {len(score_values)} scores for {len(to_score)} candidates"
            )
        elapsed_ms = (time.perf_counter() - tick) * 1000.0
        rerank_ms_by_query.append(elapsed_ms)

        reranked_ids = [doc_id for _, doc_id in sorted(
            zip(score_values.tolist(), to_score, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )]
        # If rerank_k is smaller than candidate_k, retain unscored candidates after the scored prefix.
        reranked_run[qid] = reranked_ids + base_ids[rerank_k:]
        if i % 25 == 0 or i == len(query_ids):
            print(f"Cross-encoder: processed {i}/{len(query_ids)} queries")

    reranked_metrics = evaluate(reranked_run, qrels, args.top_k)
    reranked_candidate_recall = float(np.mean([
        recall_at_k(reranked_run[qid], qrels[qid], args.candidate_k)
        for qid in query_ids
    ]))

    first_stage_ms = [
        source_latency_ms["bm25plus"][i]
        + source_latency_ms["dense"][i]
        + fusion_latency_ms[i]
        for i in range(len(query_ids))
    ]
    end_to_end_ms = [first_stage_ms[i] + rerank_ms_by_query[i] for i in range(len(query_ids))]

    output = {
        "dataset": "BEIR SciFact",
        "qrels_split": args.split,
        "documents": len(docs),
        "queries_evaluated": len(queries),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "candidate_source": {
            "retrievers": ["bm25plus", "dense"],
            "fusion": "weighted_rrf",
            "rrf_k": 60,
            "weights": {"bm25plus": 1.0, "dense": args.dense_weight},
        },
        "reranker": {
            "model": args.reranker_model,
            "rerank_k": rerank_k,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": args.device,
            "setup_seconds_including_model_load": reranker_setup_s,
        },
        "candidate_generation": {
            "metrics_at_top_k": baseline_metrics,
            f"Recall@{args.candidate_k}": candidate_recall,
            "bm25plus_setup_seconds": bm25_setup_s,
            "dense_setup_seconds": dense_setup_s,
            "dense_cache_hit": dense.cache_hit,
        },
        "reranked": {
            "metrics_at_top_k": reranked_metrics,
            f"Recall@{args.candidate_k}": reranked_candidate_recall,
        },
        "latency": {
            "mean_bm25plus_query_ms": float(np.mean(source_latency_ms["bm25plus"])),
            "p95_bm25plus_query_ms": percentile(source_latency_ms["bm25plus"], 95),
            "mean_dense_query_ms": float(np.mean(source_latency_ms["dense"])),
            "p95_dense_query_ms": percentile(source_latency_ms["dense"], 95),
            "mean_fusion_ms": float(np.mean(fusion_latency_ms)),
            "mean_reranker_query_ms": float(np.mean(rerank_ms_by_query)),
            "p95_reranker_query_ms": percentile(rerank_ms_by_query, 95),
            "mean_end_to_end_query_ms": float(np.mean(end_to_end_ms)),
            "p95_end_to_end_query_ms": percentile(end_to_end_ms, 95),
        },
        "notes": [
            "Reranking changes the order of retrieved candidates; it cannot recover relevant documents absent from the candidate pool.",
            "Candidate generation uses BM25Plus + Qwen3 weighted RRF; dense_weight and candidate_k are explicit in the output.",
            "If rerank_k is lower than candidate_k, unscored candidates retain their original order after the reranked prefix.",
            "Use the training split for model/configuration choices. Test scores are exploratory after repeated test-set comparisons.",
        ],
    }

    output_path = args.output or (RESULTS / "scifact_reranker_benchmark.json")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nSummary")
    print("Candidate-generation metrics:")
    for key, value in baseline_metrics.items():
        if key == "evaluated_queries":
            continue
        print(f"  {key}: {value:.5f}")
    print(f"  Recall@{args.candidate_k}: {candidate_recall:.5f}")
    print("After cross-encoder reranking:")
    for key, value in reranked_metrics.items():
        if key == "evaluated_queries":
            continue
        print(f"  {key}: {value:.5f}")
    print(f"  Recall@{args.candidate_k}: {reranked_candidate_recall:.5f}")
    print(f"Mean reranker query latency: {np.mean(rerank_ms_by_query):.1f} ms")
    print(f"Mean end-to-end query latency: {np.mean(end_to_end_ms):.1f} ms")
    print(f"Dense cache hit: {dense.cache_hit}")
    print(f"Saved separate result file to: {output_path}")


if __name__ == "__main__":
    main()
