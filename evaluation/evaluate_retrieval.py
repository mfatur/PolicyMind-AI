
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from io import StringIO

from app.retrieval import retrieve_documents


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "evaluation" / "data" / "eval_dataset.json"
OUTPUT_PATH = PROJECT_DIR / "evaluation" / "data" / "retrieval_results.json"


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    results = []
    hit_at_3 = 0
    hit_at_5 = 0

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        expected_source = item["source"]

        # Menahan print diagnostik agar terminal tetap rapi.
        with redirect_stdout(StringIO()):
            retrieved = retrieve_documents(question, top_k=5)

        retrieved_sources = [
            doc["source"] for doc in retrieved
        ]

        hit3 = expected_source in retrieved_sources[:3]
        hit5 = expected_source in retrieved_sources[:5]

        hit_at_3 += int(hit3)
        hit_at_5 += int(hit5)

        results.append({
            "id": index,
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "hit_at_3": hit3,
            "hit_at_5": hit5,
        })

        status = "OK" if hit5 else "MISS"
        print(f"[{index:02d}/30] {status} | Hit@3={hit3} | Hit@5={hit5}")

    total = len(dataset)

    summary = {
        "total_questions": total,
        "hit_rate_at_3": hit_at_3 / total if total else 0,
        "hit_rate_at_5": hit_at_5 / total if total else 0,
        "hit_count_at_3": hit_at_3,
        "hit_count_at_5": hit_at_5,
    }

    output = {
        "summary": summary,
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\n=== RINGKASAN RETRIEVAL ===")
    print(f"Jumlah pertanyaan : {total}")
    print(
        f"Hit Rate@3        : {summary['hit_rate_at_3']:.2%} "
        f"({hit_at_3}/{total})"
    )
    print(
        f"Hit Rate@5        : {summary['hit_rate_at_5']:.2%} "
        f"({hit_at_5}/{total})"
    )
    print(f"Hasil disimpan    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
