
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_DIR / "evaluation" / "data" / "retrieval_results.json"


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("Pertanyaan yang sumbernya tidak masuk Top-3:\n")

    for item in data["results"]:
        if not item["hit_at_3"]:
            print(f"ID: {item['id']}")
            print(f"Pertanyaan: {item['question']}")
            print(f"Sumber target: {item['expected_source']}")
            print(f"Hasil retrieval: {item['retrieved_sources']}")
            print("-" * 60)

    print("\nSelesai. Belum ada file atau indeks yang diubah.")


if __name__ == "__main__":
    main()
