
from pathlib import Path

import chromadb

from app.embedding import embed_documents


DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
COLLECTION_NAME = "policy_documents"


def create_vector_store():
    # Siapkan chunk dan embedding sebelum menyentuh database lama.
    chunks, embeddings = embed_documents()

    if not chunks:
        raise ValueError("Tidak ada chunk yang bisa dimasukkan ke vector store.")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Hapus collection lama agar tidak ada chunk kebijakan yang tertinggal.
    existing_collections = client.list_collections()
    existing_names = [
        collection.name if hasattr(collection, "name") else collection
        for collection in existing_collections
    ]

    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    # Buat collection baru khusus dokumen yang sedang aktif.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"{chunk['source']}_{chunk['start_index']}"
        for chunk in chunks
    ]

    collection.add(
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

    print("Vector store berhasil dibangun ulang.")
    print("Jumlah data:", collection.count())
    print("Nama collection:", collection.name)
    print("Sumber dokumen:", sorted({
        metadata["source"]
        for metadata in collection.get(include=["metadatas"])["metadatas"]
    }))
