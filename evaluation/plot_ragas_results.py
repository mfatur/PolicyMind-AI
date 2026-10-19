
import matplotlib.pyplot as plt
from pathlib import Path

metrics = {
    "Faithfulness": 0.8594,
    "Answer Relevancy": 0.8138,
    "Context Precision": 0.8889,
    "Context Recall": 0.9333,
}

names = list(metrics.keys())
scores = [score * 100 for score in metrics.values()]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(names, scores)
ax.set_xlim(0, 100)
ax.set_xlabel("Score (%)")
ax.set_title("PolicyMind AI — RAGAS Evaluation (30 Questions)")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

for bar, score in zip(bars, scores):
    ax.text(
        score + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{score:.2f}%",
        va="center",
    )

fig.tight_layout()
OUTPUT_DIR = Path(__file__).resolve().parent / "grafik_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(OUTPUT_DIR / "ragas_summary.png", dpi=180)
plt.show()
