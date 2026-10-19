
import json
from pathlib import Path

# Lokasi folder proyek
BASE_DIR = Path(__file__).resolve().parents[1]

# File hasil evaluasi RAGAS yang sudah tersedia
RESULTS_PATH = BASE_DIR / "evaluation" / "data" / "ragas_results_30.json"

def main():
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    dataset_path = BASE_DIR / "evaluation" / "data" / "eval_dataset.json"

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    # Ambil hanya data pertanyaan, bukan key tambahan seperti "_summary"
    questions = [
        (int(question_id), result)
        for question_id, result in results.items()
        if question_id.isdigit()
    ]

    # Urutkan dari Answer Relevancy terendah
    questions.sort(
        key=lambda item: item[1]["Answer Relevancy"]
    )

    print("\n=== 5 PERTANYAAN DENGAN ANSWER RELEVANCY TERENDAH ===\n")

    for question_id, result in questions[:5]:
        question_data = dataset[question_id - 1]

        print(f"\n{'=' * 70}")
        print(f"ID               : {question_id}")
        print(f"Pertanyaan       : {question_data['question']}")
        print(f"Ground truth     : {question_data['ground_truth']}")
        print(f"Answer Relevancy : {result['Answer Relevancy']}")

        print("\nKonteks retrieval:")
        for index, context in enumerate(result.get("contexts", []), start=1):
            print(f"\n[Context {index}]")
            print(context)

        print("\nJawaban:")
        print(result.get("answer", "Tidak tersedia"))

if __name__ == "__main__":
    main()
