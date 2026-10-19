import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(__file__).resolve().parent / "data" / "eval_dataset.json"

sys.path.insert(0, str(PROJECT_DIR / "app"))

from rag_pipeline import ask_policy

load_dotenv(PROJECT_DIR / ".env")

# LLM evaluator melalui endpoint Groq yang kompatibel dengan OpenAI
client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

judge_llm = llm_factory(
    "openai/gpt-oss-120b",
    client=client,
    adapter="instructor",
    max_tokens=2048,
)

# Embedding lokal untuk AnswerRelevancy
judge_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu",
)

# Empat metrik RAGAS
faithfulness = Faithfulness(llm=judge_llm)
answer_relevancy = AnswerRelevancy(
    llm=judge_llm,
    embeddings=judge_embeddings,
)
context_precision = ContextPrecision(llm=judge_llm)
context_recall = ContextRecall(llm=judge_llm)


async def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    results_path = DATASET_PATH.parent / "ragas_results_30.json"

    # Muat hasil sebelumnya jika ada.
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as file:
            saved_results = json.load(file)
    else:
        saved_results = {}

    metrics = {
        "Faithfulness": faithfulness,
        "Answer Relevancy": answer_relevancy,
        "Context Precision": context_precision,
        "Context Recall": context_recall,
    }

    for index, sample in enumerate(dataset, start=1):
        question_id = str(index)

        print(f"\n{'=' * 50}")
        print(f"Evaluasi {index}/{len(dataset)}")
        print("Pertanyaan:", sample["question"])

        # Ambil hasil evaluasi sebelumnya jika tersedia.
        question_results = saved_results.setdefault(question_id, {})
        if all(
            metric_name in question_results
            for metric_name in metrics
        ):
            print("Semua metrik sudah tersimpan; dilewati.")
            continue

        # Gunakan jawaban dan konteks yang sudah tersimpan.
        if "answer" in question_results and "contexts" in question_results:
            result = {
                "answer": question_results["answer"],
                "contexts": question_results["contexts"],
            }
        else:
            # Buat hasil RAG hanya jika belum tersimpan.
            result = ask_policy(sample["question"])

            question_results["answer"] = result["answer"]
            question_results["contexts"] = result["contexts"]

            with open(results_path, "w", encoding="utf-8") as file:
                json.dump(
                    saved_results,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        print("Jawaban:", result["answer"])
        print("Jumlah konteks:", len(result["contexts"]))

        for metric_name, metric in metrics.items():
            if metric_name in question_results:
                print(
                    f"{metric_name}: {question_results[metric_name]} "
                    "(tersimpan, dilewati)"
                )
                continue

            print(f"\nMenghitung {metric_name}...")

            if metric_name == "Faithfulness":
                score = await metric.ascore(
                    user_input=sample["question"],
                    response=result["answer"],
                    retrieved_contexts=result["contexts"],
                )

            elif metric_name == "Answer Relevancy":
                score = await metric.ascore(
                    user_input=sample["question"],
                    response=result["answer"],
                )

            else:
                score = await metric.ascore(
                    user_input=sample["question"],
                    retrieved_contexts=result["contexts"],
                    reference=sample["ground_truth"],
                )

            question_results[metric_name] = score.value

            # Simpan segera setelah satu metrik berhasil.
            with open(results_path, "w", encoding="utf-8") as file:
                json.dump(
                    saved_results,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"{metric_name}: {score.value}")

    
    print(f"\n{'=' * 50}")
    print("RINGKASAN SKOR")
    print(f"{'=' * 50}")

    summary = {}

    for metric_name in metrics:
        scores = [
            question_results[metric_name]
            for question_id, question_results in saved_results.items()
            if question_id != "_summary"
            and metric_name in question_results
        ]

        average = sum(scores) / len(scores) if scores else None

        summary[metric_name] = {
            "average": average,
            "completed": len(scores),
            "total": len(dataset),
        }

        if average is not None:
            print(
                f"{metric_name}: {average:.4f} "
                f"({len(scores)}/{len(dataset)} pertanyaan)"
            )
        else:
            print(f"{metric_name}: belum ada skor")

    saved_results["_summary"] = summary

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(
            saved_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nHasil evaluasi tersimpan di: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())