
from pathlib import Path

import chromadb

DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
COLLECTION_NAME = "policy_documents"

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

results = collection.get(
    include=["documents", "metadatas"]
)

print(f"Total chunks: {len(results['ids'])}\n")

for i, (chunk_id, content, metadata) in enumerate(
    zip(
        results["ids"],
        results["documents"],
        results["metadatas"],
    ),
    start=1,
):
    print("=" * 80)
    print(f"CHUNK {i}")
    print(f"ID      : {chunk_id}")
    print(f"Sumber  : {metadata['source']}")
    print(f"Index   : {metadata['start_index']}")
    print("-" * 80)
    print(content)
    print()