
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Lokasi file
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ragas_results_30.json"
OUTPUT_DIR = BASE_DIR / "grafik_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Baca hasil evaluasi
with DATA_PATH.open("r", encoding="utf-8") as file:
    results = json.load(file)

# Ambil hanya entri pertanyaan yang memiliki empat skor
metrics = {
    "Faithfulness": [],
    "Answer Relevancy": [],
    "Context Precision": [],
    "Context Recall": [],
}

question_numbers = []


for question_id, result in sorted(
    (
        item for item in results.items()
        if item[0].isdigit()
    ),
    key=lambda item: int(item[0]),
):
    if not all(name in result for name in metrics):
        continue

    question_numbers.append(int(question_id))

    for name in metrics:
        metrics[name].append(result[name])

# Buat grafik
fig, ax = plt.subplots(figsize=(12, 6))

for name, scores in metrics.items():
    ax.plot(
        question_numbers,
        scores,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=name,
    )

ax.set_title("PolicyMind AI — RAGAS Scores per Question")
ax.set_xlabel("Question Number")
ax.set_ylabel("Score")
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(question_numbers)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

# Simpan grafik
output_path = OUTPUT_DIR / "ragas_per_question.png"
fig.savefig(output_path, dpi=180)
plt.show()

print(f"Jumlah pertanyaan yang divisualisasikan: {len(question_numbers)}")
print(f"Grafik disimpan di: {output_path}")
