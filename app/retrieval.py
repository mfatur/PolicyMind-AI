import logging
from pathlib import Path

import chromadb

from app.embedding import create_embeddings
from langfuse import observe

DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
COLLECTION_NAME = "policy_documents"
logger = logging.getLogger(__name__)

client = chromadb.PersistentClient(path=str(DB_DIR))

def get_collection():
    return client.get_collection(name=COLLECTION_NAME)

@observe(name="retrieve-documents", as_type="retriever")
def retrieve_documents(question, top_k=3, search_query=None):
    collection = get_collection()
    # Ambil kandidat lebih banyak sebelum reranking.
    candidate_k = min(max(top_k * 10, top_k), collection.count())

    retrieval_query = search_query or question
    query_embedding = create_embeddings([retrieval_query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []

    for i, content in enumerate(results["documents"][0]):
        candidates.append({
            "content": content,
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    logger.info(
        "ChromaDB mengembalikan %d kandidat; memilih top_k=%d.",
        len(candidates),
        min(top_k, len(candidates)),
    )
    if not candidates:
        return []

    # ChromaDB sudah mengurutkan kandidat berdasarkan jarak embedding.
    # Ambil top_k kandidat teratas tanpa menjalankan reranker.
    return candidates[:top_k]


if __name__ == "__main__":
    question = "Berapa maksimal hari WFH dalam seminggu?"
    results = retrieve_documents(question, top_k=3)

    print("Pertanyaan:", question)

    for i, result in enumerate(results, start=1):
        print(f"\n--- Hasil {i} ---")
        print("Sumber:", result["source"])
        print("Distance:", round(result["distance"], 4))
        print("Isi:", result["content"])
