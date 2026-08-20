from sentence_transformers import SentenceTransformer
from ingestion import load_pdfs, chunk_text


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings():
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    documents = load_pdfs()

    all_chunks = []

    for document in documents:
        chunks = chunk_text(document["text"])

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": document["source"]
            })

    print(f"Total chunks: {len(all_chunks)}")

    texts = [chunk["text"] for chunk in all_chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    print("Embeddings created!")
    print("Number of embeddings:", len(embeddings))
    print("Embedding dimension:", len(embeddings[0]))

    return all_chunks, embeddings


if __name__ == "__main__":
    create_embeddings()