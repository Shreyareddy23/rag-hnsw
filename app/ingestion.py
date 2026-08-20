from pathlib import Path
from pypdf import PdfReader


DOCUMENTS_DIR = Path("data/documents")


def load_pdfs():
    documents = []

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        documents.append({
            "source": pdf_path.name,
            "text": text
        })

    return documents


# def chunk_text(text, chunk_size=500, overlap=50):
#     chunks = []

#     start = 0

#     while start < len(text):
#         end = start + chunk_size

#         chunk = text[start:end].strip()

#         if chunk:
#             chunks.append(chunk)

#         start += chunk_size - overlap

#     return chunks
def chunk_text(text, chunk_size=800, overlap=100):
    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If adding the paragraph keeps us under the limit
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n"

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Keep some overlap from previous chunk
            overlap_text = current_chunk[-overlap:]

            current_chunk = overlap_text + "\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


if __name__ == "__main__":
    documents = load_pdfs()

    print(f"Loaded {len(documents)} PDFs")

    for document in documents:
        chunks = chunk_text(document["text"])

        print(
            f"{document['source']}: "
            f"{len(document['text'])} characters → "
            f"{len(chunks)} chunks"
        )

        print("\nFirst chunk:")
        print(chunks[0][:500])
        print("-" * 60)