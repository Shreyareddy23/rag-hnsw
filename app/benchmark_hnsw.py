import time

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "rag_documents"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_data():
    client = QdrantClient(path="qdrant_data")

    # Get all vectors from Qdrant
    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_vectors=True,
        with_payload=True,
    )

    points = result[0]

    vectors = np.array(
        [point.vector for point in points]
    )

    return client, points, vectors


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def exact_search(query_vector, vectors, top_k=5):

    similarities = []

    for idx, vector in enumerate(vectors):

        score = cosine_similarity(
            query_vector,
            vector
        )

        similarities.append((idx, score))

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_k]


def hnsw_search(client, query_vector, top_k=5):

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
    ).points

    return [
        (result.id, result.score)
        for result in results
    ]


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    client, points, vectors = load_data()

    print(f"Total vectors: {len(vectors)}")
    print(f"Vector dimension: {vectors.shape[1]}")

    queries = [
        "What is HNSW?",
        "How does HNSW search work?",
        "What is approximate nearest neighbor search?",
        "What is retrieval augmented generation?",
        "How are HNSW layers constructed?"
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        query_vector = model.encode(query)

        # -----------------------------
        # EXACT SEARCH
        # -----------------------------

        start = time.perf_counter()

        exact_results = exact_search(
            query_vector,
            vectors,
            top_k=5
        )

        exact_time = (
            time.perf_counter() - start
        ) * 1000

        # -----------------------------
        # HNSW SEARCH
        # -----------------------------

        start = time.perf_counter()

        hnsw_results = hnsw_search(
            client,
            query_vector,
            top_k=5
        )

        hnsw_time = (
            time.perf_counter() - start
        ) * 1000

        # -----------------------------
        # RECALL
        # -----------------------------

        exact_ids = {
            points[idx].id
            for idx, score in exact_results
        }

        hnsw_ids = {
            idx
            for idx, score in hnsw_results
        }

        intersection = (
            exact_ids & hnsw_ids
        )

        recall = (
            len(intersection) / 5
        )

        # -----------------------------
        # RESULTS
        # -----------------------------

        print(
            f"\nExact search: {exact_time:.4f} ms"
        )

        print(
            f"HNSW search:  {hnsw_time:.4f} ms"
        )

        print(
            f"Recall@5:     {recall * 100:.2f}%"
        )

        print(
            f"\nExact IDs: {exact_ids}"
        )

        print(
            f"HNSW IDs:  {hnsw_ids}"
        )


if __name__ == "__main__":
    main()