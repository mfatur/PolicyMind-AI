from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder

from app.embedding import create_embeddings


DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
COLLECTION_NAME = "policy_documents"

client = chromadb.PersistentClient(path=str(DB_DIR))

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER_MODEL)


def get_collection():
    return client.get_collection(name=COLLECTION_NAME)


def retrieve_documents(question, top_k=3, search_query=None):
    collection = get_collection()
    # Ambil kandidat lebih banyak sebelum reranking.
    candidate_k = min(max(top_k * 3, top_k), collection.count())

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
    print("\n=== KANDIDAT CHROMADB ===")
    for i, candidate in enumerate(candidates, start=1):
        print(
            f"{i}. source={candidate['source']} | "
            f"distance={candidate['distance']:.4f}"
        )
        print(candidate["content"][:200].replace("\n", " "))
    if not candidates:
        return []

    pairs = [
        (question, candidate["content"])
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked_documents = sorted(
        candidates,
        key=lambda doc: doc["rerank_score"],
        reverse=True,
    )
    print("\n=== HASIL RERANKING ===")
    for i, doc in enumerate(ranked_documents, start=1):
        print(
            f"{i}. source={doc['source']} | "
            f"rerank_score={doc['rerank_score']:.4f}"
        )
        print(doc["content"][:200].replace("\n", " "))
    return ranked_documents[:top_k]


if __name__ == "__main__":
    question = "Kalau saya sudah membeli kebutuhan kerja, apakah pasti akan diganti?"
    results = retrieve_documents(question, top_k=3)

    print("Pertanyaan:", question)

    for i, result in enumerate(results, start=1):
        print(f"\n--- Hasil {i} ---")
        print("Sumber:", result["source"])
        print("Distance:", round(result["distance"], 4))
        print("Rerank score:", round(result["rerank_score"], 4))
        print("Isi:", result["content"])
