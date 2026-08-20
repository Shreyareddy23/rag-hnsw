from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "rag_documents"
MODEL_NAME = "all-MiniLM-L6-v2"


def search(query, top_k=5):

    # Connect to our local Qdrant database
    client = QdrantClient(path="qdrant_data")

    # Load the same embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Convert query into vector
    query_vector = model.encode(query).tolist()

    # Search Qdrant using HNSW
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return results


if __name__ == "__main__":

    query = input("Enter your question: ")

    results = search(query)

    print("\nRetrieved documents:\n")

    for i, result in enumerate(results, 1):

        print(f"--- Result {i} ---")
        print(f"Score: {result.score}")
        print(f"Source: {result.payload['source']}")
        print(result.payload["text"][:1000])
        print()