from retrieval.loader import load_documents
from retrieval.faiss_store import build_faiss_index, search
from llm.llm_service import generate_answer

print("🔄 Loading documents...")
docs = load_documents("data")

print("🔄 Building FAISS index...")
build_faiss_index(docs)

print("✅ RAG system ready")


def rag_answer(query: str) -> str:
    results = search(query, top_k=3)

    if not results:
        return "Sorry, no relevant information found."

    context = "\n\n".join([doc["text"] for doc in results])

    prompt = f"""
You are a career guidance assistant.
Answer ONLY using the context below.
If not found in context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    return generate_answer(prompt)
