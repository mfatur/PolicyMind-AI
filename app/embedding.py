
from sentence_transformers import SentenceTransformer

from app.load_documents import load_documents
from app.chunking import chunk_text


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Model dimuat satu kali ketika module diimpor.
model = SentenceTransformer(MODEL_NAME)


def get_embedding_model():
    return model


def create_embeddings(texts):
    return get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
    )


def embed_documents():
    documents = load_documents()
    all_chunks = []

    for document in documents:
        chunks = chunk_text(document["content"])

        for chunk in chunks:
            all_chunks.append({
                "source": document["source"],
                "content": chunk["content"],
                "start_index": chunk["start_index"],
            })

    texts = [chunk["content"] for chunk in all_chunks]
    embeddings = create_embeddings(texts)

    return all_chunks, embeddings


if __name__ == "__main__":
    chunks, embeddings = embed_documents()

    print("Jumlah chunk:", len(chunks))
    print("Bentuk embedding:", embeddings.shape)
    print("Sumber chunk pertama:", chunks[0]["source"])
