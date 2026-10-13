# Local RAG Retrieval Experiments

A local retrieval-augmented generation (RAG) project focused on finding and ranking evidence-bearing scientific papers. Experiments use **BEIR SciFact** and run on an Apple M3 MacBook Air (16 GB unified memory).

## Current status

- **Best observed test result:** BM25Plus + Qwen3 dense retrieval with DBSF fusion (`nDCG@10 = 0.7230`).
- **Current reranker candidate:** `ncbi/MedCPT-Cross-Encoder` improved all three top-10 metrics on a 100-query training sample. The full training evaluation is pending.
- **Dense embeddings are cached**, so reruns should avoid rebuilding the Qwen3 document vectors. Check `dense_cache_hit` in the output.

## Pipeline

1. **BM25Plus** retrieves documents using lexical matches.
2. **Qwen3 embeddings + vector search** retrieve semantically similar documents. Qwen3 creates 1,024-dimensional vectors; vector search compares a query vector with the cached document vectors.
3. **Fusion** combines the ranked lists. Tested methods include RRF, weighted RRF and DBSF.
4. **Optional MedCPT cross-encoder** scores query-document pairs and reranks the candidates. It changes their order but cannot retrieve documents absent from the candidate pool.

The benchmark reports nDCG@10, Recall@10 and MRR@10. Higher is better. Recall@200 measures whether relevant documents made it into the candidate pool; latency measures the cost of retrieval and reranking.

## Dataset

Local files: `dataset/scifact/scifact/`

- `corpus.jsonl`: 5,183 scientific documents
- `queries.jsonl`: query text
- `qrels/train.tsv`: training relevance judgments (809 evaluated queries)
- `qrels/test.tsv`: test relevance judgments (300 evaluated queries)

Use the **training split** to choose models and parameters. The test split has already been used for exploratory comparisons, so its results should not be described as an untouched final evaluation.

## Key results

### Exploratory test results

| Method | nDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| BM25Okapi | 0.6519 | 0.7740 | 0.6186 |
| BM25Plus | 0.6646 | 0.7816 | 0.6339 |
| Qwen3 dense | 0.7028 | 0.8349 | 0.6659 |
| BM25Plus + Qwen3, DBSF (`candidate_k=100`) | **0.7230** | 0.8436 | **0.6901** |
| BM25Plus + Qwen3, weighted RRF (dense weight 7, `candidate_k=200`) | 0.7178 | 0.8432 | 0.6827 |

### Tuning and reranking

- Weighted RRF dense weight **7.0** scored best among tested weights on training data.
- Candidate depths 50, 100 and 200 performed very similarly; 200 slightly favored nDCG@10 and Recall@10, while 100 was marginally better on MRR@10 and cheaper.
- The general MiniLM cross-encoder reduced ranking quality on an initial 20-query sample, so it was not advanced to a full run.
- MedCPT on a 100-query training sample improved nDCG@10 from **0.7501 to 0.7747**, Recall@10 from **0.8350 to 0.8700**, and MRR@10 from **0.7304 to 0.7458**. Mean reranking time was about 4.2 seconds/query on Apple's MPS GPU. This is promising but needs confirmation on all 809 training queries.

Small smoke tests are sensitive to query selection; compare metrics from the same sample and use the full training split before drawing conclusions.

## Running experiments

Activate the project's environment first. These commands assume the repository root is the current directory.

### Compare retrieval methods

```bash
python -m evaluation.scifact_methods_benchmark \
  --methods bm25plus,dense \
  --fusion weighted_rrf \
  --split train \
  --dense-weight 7 \
  --candidate-k 200
```

### Full MedCPT training evaluation

```bash
python -m evaluation.scifact_reranker_benchmark \
  --split train \
  --candidate-k 200 \
  --rerank-k 100 \
  --dense-weight 7 \
  --reranker-model ncbi/MedCPT-Cross-Encoder \
  --batch-size 8 \
  --max-length 512 \
  --device mps \
  --output results/tuning/medcpt_reranker_train_full.json
```

The full MedCPT run is estimated to take about an hour based on the 100-query sample. Connect the Mac to power and monitor memory pressure. Verify that the Qwen3 dense cache is reused.

## Next steps

1. Complete the full MedCPT training evaluation.
2. If its gains persist, evaluate the fixed configuration on SciFact test and record the result as exploratory.
3. Compare a scientific-domain embedding model (e.g. SPECTER2) and query expansion against the same baseline.
