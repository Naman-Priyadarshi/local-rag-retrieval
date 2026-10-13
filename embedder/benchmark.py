import json

from mlx_qwen3 import Qwen3MLXEmbedder


def main():
    # Temporary examples to validate the pipeline.
    # Replace these with representative chunks from your dataset.
    sample_texts = [
        "Hybrid retrieval combines keyword search with dense vector search.",
        "BM25 ranks documents using lexical term statistics.",
        "A reranker scores query-document pairs for relevance.",
        "Embeddings represent semantic meaning as numerical vectors.",
        "Recall at k measures how many relevant documents appear in top k.",
        "nDCG rewards relevant results appearing near the top of a ranking.",
        "A language model generates an answer from retrieved context.",
        "Chunk overlap can preserve context across document boundaries.",
        "Vector search commonly uses cosine similarity or inner products.",
        "Evaluation requires representative queries and relevance labels.",
    ]

    embedder = Qwen3MLXEmbedder(batch_size=8)

    print(f"Loading: {embedder.model_id}")
    embedder.load_model()
    vectors = embedder.embed_documents(sample_texts)
    query_vector = embedder.embed_query(
        "How can I evaluate the quality of document retrieval?"
    )

    print("\nSanity checks")
    print("Document embedding shape:", vectors.shape)
    print("Query embedding shape:", query_vector.shape)
    print("All finite:", bool(__import__("numpy").isfinite(vectors).all()))
    print("First vector norm:", float(__import__("numpy").linalg.norm(vectors[0])))

    scores = vectors @ query_vector
    ranked = scores.argsort()[::-1]

    print("\nTop results")
    for index in ranked[:5]:
        print(f"{scores[index]:.4f} | {sample_texts[index]}")

    print("\nBenchmark")
    result = embedder.benchmark(sample_texts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
