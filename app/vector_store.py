
from pathlib import Path

import chromadb

from app.embedding import embed_documents


DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
COLLECTION_NAME = "policy_documents"


def create_vector_store():
    chunks, embeddings = embed_documents()

    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"{chunk['source']}_{chunk['start_index']}"
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "start_index": chunk["start_index"],
            }
            for chunk in chunks
        ],
    )

    return collection


if __name__ == "__main__":
    collection = create_vector_store()

    print("Vector store berhasil dibuat.")
    print("Jumlah data:", collection.count())
    print("Nama collection:", collection.name)
