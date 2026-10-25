import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path

import chromadb


DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
COLLECTION_NAME = "policy_documents"


@contextmanager
def _vector_store_lock():
    lock_path = DB_DIR / ".build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()

            while True:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(
                        lock_file.fileno(),
                        msvcrt.LK_LOCK,
                        1,
                    )
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(0.1)

            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(
                    lock_file.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _build_vector_store(client):
    from app.embedding import embed_documents

    # Siapkan chunk dan embedding sebelum menyentuh database lama.
    chunks, embeddings = embed_documents()

    if not chunks:
        raise ValueError("Tidak ada chunk yang bisa dimasukkan ke vector store.")

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


def create_vector_store():
    """Rebuild the policy collection explicitly."""
    with _vector_store_lock():
        client = chromadb.PersistentClient(path=str(DB_DIR))
        return _build_vector_store(client)


def ensure_vector_store():
    """Ensure the collection and embedding model are ready under one lock."""
    with _vector_store_lock():
        client = chromadb.PersistentClient(path=str(DB_DIR))
        existing_names = {
            collection.name
            if hasattr(collection, "name")
            else collection
            for collection in client.list_collections()
        }

        if COLLECTION_NAME in existing_names:
            collection = client.get_collection(name=COLLECTION_NAME)
            if collection.count() == 0:
                collection = _build_vector_store(client)
        else:
            collection = _build_vector_store(client)

        from app.embedding import get_embedding_model

        get_embedding_model()
        return collection


if __name__ == "__main__":
    collection = ensure_vector_store()

    print("Vector store siap digunakan.")
    print("Jumlah data:", collection.count())
    print("Nama collection:", collection.name)
    print("Sumber dokumen:", sorted({
        metadata["source"]
        for metadata in collection.get(include=["metadatas"])["metadatas"]
    }))
