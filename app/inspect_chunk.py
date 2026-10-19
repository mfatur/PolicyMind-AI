
from pathlib import Path

import chromadb


DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name="policy_documents")

results = collection.get(
    where={"source": "kebijakan_jam_kerja.txt"},
    include=["documents", "metadatas"],
)

for index, (content, metadata) in enumerate(
    zip(results["documents"], results["metadatas"]),
    start=1,
):
    print(f"\n--- Chunk {index} ---")
    print("Start index:", metadata.get("start_index"))
    print("Panjang:", len(content))
    print("Isi:", content)
