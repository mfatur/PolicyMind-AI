
import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "data" / "ragas_results.json"
OUTPUT_DIR = BASE_DIR / "grafik_result"
OUTPUT_PATH = OUTPUT_DIR / "ragas_metrics_summary.png"


def main():
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    summary = results["_summary"]

    metrics = [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
    ]

    scores = [summary[name]["average"] for name in metrics]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.barh(metrics, scores)

    ax.set_title("PolicyMind AI - RAGAS Evaluation")
    ax.set_xlabel("Average Score")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik berhasil disimpan: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
