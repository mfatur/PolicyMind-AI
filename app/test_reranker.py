from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path

from sentence_transformers import CrossEncoder
from app.retrieval import retrieve_documents


DATA_PATH = Path("evaluation/data/eval_dataset.json")
QUESTIONS = [1, 13, 16, 28, 30]

data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

model = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

for number in QUESTIONS:
    item = data[number - 1]
    question = item["question"]
    references = item["reference_contexts"]

    # Ambil kandidat satu kali agar kedua metode membandingkan
    # kumpulan dokumen yang sama.
    with redirect_stdout(StringIO()):
        candidates = retrieve_documents(question, top_k=9)

    pairs = [
        (question, candidate["content"])
        for candidate in candidates
    ]
    scores = model.predict(pairs)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    def matches_reference(content):
        content = " ".join(content.split()).casefold()
        return any(
            " ".join(reference.split()).casefold() in content
            for reference in references
        )

    baseline_rank = next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if matches_reference(candidate["content"])
        ),
        None,
    )

    reranked_rank = next(
        (
            rank
            for rank, (candidate, _) in enumerate(reranked, start=1)
            if matches_reference(candidate["content"])
        ),
        None,
    )

    print(f"\n{'=' * 65}")
    print(f"Q{number}: {question}")
    print("Baseline rank:", baseline_rank)
    print("CrossEncoder rank:", reranked_rank)

    print("\nTop 3 CrossEncoder:")
    for rank, (candidate, score) in enumerate(reranked[:3], start=1):
        print(f"{rank}. score={float(score):.4f}")
        print(candidate["content"][:220].replace("\n", " "))