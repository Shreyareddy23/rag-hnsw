from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
)

from embeddings import create_embeddings


COLLECTION_NAME = "rag_documents"


def create_vector_store():

    # Local Qdrant database
    client = QdrantClient(path="qdrant_data")

    chunks, embeddings = create_embeddings()

    # Create collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
    )

    points = []

    for idx, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        points.append(
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print("Vector store created!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors stored: {len(points)}")

    return client


if __name__ == "__main__":
    create_vector_store()