
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_documents():
    documents = []

    for file_path in DATA_DIR.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "content": content,
        })

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"Jumlah dokumen: {len(docs)}")
    print(f"\nSumber: {docs[0]['source']}")
    print(f"Isi dokumen:\n{docs[0]['content']}")
