# retrieval/faiss_store.py

import faiss
import numpy as np
from typing import List, Dict
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.llm_service import get_embedding   # adjust name if needed

# Global variables
faiss_index = None
documents_store = []


def build_faiss_index(documents: List[Dict]):
    """
    Build FAISS index from documents
    """
    global faiss_index, documents_store

    documents_store = documents

    embeddings = []
    for doc in documents:
        vector = get_embedding(doc["text"])
        embeddings.append(vector)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    print(f"✅ FAISS index built with {faiss_index.ntotal} vectors")


def search(query: str, top_k: int = 3) -> List[Dict]:
    """
    Search similar documents
    """
    if faiss_index is None:
        raise ValueError("FAISS index not built")

    query_vector = np.array(
        [get_embedding(query)]
    ).astype("float32")

    distances, indices = faiss_index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(documents_store):
            results.append(documents_store[idx])

    return results


if __name__ == "__main__":
    from loader import load_documents

    docs = load_documents()
    build_faiss_index(docs)

    res = search("How to become a data scientist?")
    print(res)
