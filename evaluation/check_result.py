
import json
from pathlib import Path

results_path = Path(__file__).parent / "data" / "ragas_results.json"

with open(results_path, "r", encoding="utf-8") as file:
    data = json.load(file)

question_ids = [key for key in data if key.isdigit()]
metrics = [
    "Faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
]

all_valid = True

for metric in metrics:
    scores = [
        data[qid][metric]
        for qid in question_ids
        if metric in data[qid]
    ]

    summary = data.get("_summary", {}).get(metric, {})
    average = sum(scores) / len(scores) if scores else None

    valid = (
        average is not None
        and summary.get("completed") == len(scores)
        and summary.get("total") == len(question_ids)
        and abs(summary.get("average", -999) - average) < 1e-9
    )

    print(
        f"{metric}: {'VALID' if valid else 'CHECK'}"
        + (f" | rata-rata={average:.4f}" if average is not None else "")
    )

    if not valid:
        all_valid = False

print("\nHASIL:", "SEMUA VALID" if all_valid else "PERLU DIPERIKSA")
