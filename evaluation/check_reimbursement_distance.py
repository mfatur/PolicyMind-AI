
from pathlib import Path

import chromadb
import numpy as np

from app.embedding import create_embeddings


db_dir = Path(__file__).resolve().parents[1] / "data" / "chroma_db"

client = chromadb.PersistentClient(path=str(db_dir))
collection = client.get_collection(name="policy_documents")


questions = [
    "Kalau saya sudah membeli kebutuhan kerja, apakah pasti akan diganti?",
    "Bagaimana prosedur reimbursement pembelian kebutuhan kerja?",
    "Apakah biaya barang yang dibeli untuk pekerjaan bisa diganti perusahaan?",
    "Apa syarat agar pembelian kebutuhan kerja dapat di-reimburse?",
]

for question in questions:
    query_embedding = create_embeddings([question])[0]

    results = collection.get(
        where={"source": "kebijakan_reimbursement.txt"},
        include=["embeddings", "documents"],
    )

    distances = []

    for doc_id, embedding, content in zip(
        results["ids"],
        results["embeddings"],
        results["documents"],
    ):
        embedding = np.asarray(embedding)
        distance = 1 - np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        distances.append((distance, doc_id, content))

    distances.sort(key=lambda item: item[0])

    print(f"\nPertanyaan: {question}")
    for distance, doc_id, content in distances:
        print(f"{distance:.4f} | {doc_id} | {content[:100].replace(chr(10), ' ')}")


    print(f"\nID: {doc_id}")
    print(f"Distance: {distance:.4f}")
    print("Isi:", content[:180].replace("\n", " "))
