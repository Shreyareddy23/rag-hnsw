import os

from dotenv import load_dotenv
from groq import Groq

from retriever import search


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


client = Groq(api_key=API_KEY)


def generate_answer(query, top_k=5):

    # 1. Retrieve relevant chunks using HNSW
    results = search(query, top_k=top_k)

    # 2. Build context from retrieved chunks
    context_parts = []

    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}: {result.payload['source']}]\n"
            f"{result.payload['text']}"
        )

    context = "\n\n".join(context_parts)

    # 3. Send retrieved context to Groq
    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information is not available in the documents.

Mention the source document in your answer.

Context:
----------------
{context}
----------------

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You answer questions using retrieved document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return answer, results


if __name__ == "__main__":

    query = input("Enter your question: ")

    answer, results = generate_answer(query)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES RETRIEVED")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result.payload['source']} "
            f"(score={result.score:.4f})"
        )