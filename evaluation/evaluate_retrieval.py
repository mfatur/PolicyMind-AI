
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "evaluation" / "data" / "eval_dataset.json"
OUTPUT_PATH = PROJECT_DIR / "evaluation" / "data" / "retrieval_results.json"
TOP_K = 5
RESULT_SCHEMA_VERSION = 2


class DatasetValidationError(ValueError):
    pass


def validate_dataset(dataset):
    if not isinstance(dataset, list) or not dataset:
        raise DatasetValidationError("Dataset harus berupa list yang tidak kosong.")

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise DatasetValidationError(f"Dataset Q{index} harus berupa object.")
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        references = item.get("reference_contexts")
        if not isinstance(question, str) or not question.strip():
            raise DatasetValidationError(f"Dataset Q{index} kehilangan question.")
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise DatasetValidationError(f"Dataset Q{index} kehilangan ground_truth.")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(value, str) or not value.strip() for value in references)
        ):
            raise DatasetValidationError(
                f"Dataset Q{index} memerlukan reference_contexts berupa list string."
            )
    return dataset


def load_dataset(path=None):
    dataset_path = Path(path) if path is not None else DATASET_PATH
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetValidationError(f"Dataset tidak ditemukan: {dataset_path}") from error
    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            f"JSON dataset rusak pada baris {error.lineno}, kolom {error.colno}."
        ) from error
    return validate_dataset(dataset)


def normalize_context(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def matching_reference_ids(content, reference_contexts):
    normalized_content = normalize_context(content)
    return [
        index
        for index, reference in enumerate(reference_contexts, start=1)
        if normalize_context(reference) in normalized_content
    ]


def score_retrieved_contexts(reference_contexts, retrieved, indexed_documents, top_k=TOP_K):
    reference_ranks = {}
    retrieved_records = []

    for rank, candidate in enumerate(retrieved[:top_k], start=1):
        matched_ids = matching_reference_ids(
            candidate["content"],
            reference_contexts,
        )
        for reference_id in matched_ids:
            reference_ranks.setdefault(reference_id, rank)
        retrieved_records.append({
            "rank": rank,
            "distance": candidate.get("distance"),
            "source": candidate.get("source"),
            "start_index": candidate.get("start_index"),
            "matched_reference_ids": matched_ids,
            "content": candidate["content"],
        })

    collection_reference_ids = sorted({
        reference_id
        for document in indexed_documents
        for reference_id in matching_reference_ids(document, reference_contexts)
    })
    first_rank = min(reference_ranks.values()) if reference_ranks else None

    if first_rank is not None:
        evidence_status = "reference_context_found_in_retrieval"
    elif collection_reference_ids:
        evidence_status = "reference_context_in_index_but_not_top_k"
    else:
        evidence_status = "no_exact_reference_context_match_in_index"

    return {
        "retrieved_contexts": retrieved_records,
        "reference_context_ranks": reference_ranks,
        "reference_contexts_in_index": collection_reference_ids,
        "evidence_status": evidence_status,
        "hit_at_1": first_rank is not None and first_rank <= 1,
        "hit_at_3": first_rank is not None and first_rank <= 3,
        "hit_at_5": first_rank is not None and first_rank <= 5,
        "mrr_at_5": 1 / first_rank if first_rank is not None else 0.0,
    }


def evaluate_dataset(dataset, retrieve, indexed_documents, top_k=TOP_K):
    validate_dataset(dataset)
    results = []

    for index, item in enumerate(dataset, start=1):
        retrieved = retrieve(item["question"], top_k=top_k)
        scores = score_retrieved_contexts(
            item["reference_contexts"],
            retrieved,
            indexed_documents,
            top_k=top_k,
        )
        results.append({
            "id": index,
            "question": item["question"],
            "reference_contexts": item["reference_contexts"],
            **scores,
        })

    total = len(results)
    summary = {
        "total_questions": total,
        "top_k": top_k,
        "metric_definition": (
            "Hit@k is true when any normalized reference_context is a substring "
            "of a retrieved chunk at rank <= k. MRR@k is the reciprocal rank of "
            "the first such match, or zero when none is retrieved."
        ),
        "hit_count_at_1": sum(item["hit_at_1"] for item in results),
        "hit_count_at_3": sum(item["hit_at_3"] for item in results),
        "hit_count_at_5": sum(item["hit_at_5"] for item in results),
        "hit_rate_at_1": sum(item["hit_at_1"] for item in results) / total,
        "hit_rate_at_3": sum(item["hit_at_3"] for item in results) / total,
        "hit_rate_at_5": sum(item["hit_at_5"] for item in results) / total,
        "mrr_at_5": sum(item["mrr_at_5"] for item in results) / total,
        "no_exact_reference_match_in_index": sum(
            item["evidence_status"] == "no_exact_reference_context_match_in_index"
            for item in results
        ),
        "interpretation_note": (
            "No exact reference match in the index is not proof that the source "
            "document contains no answer; reference_contexts may be paraphrases."
        ),
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "retrieval_source": "current ChromaDB policy_documents collection",
        "summary": summary,
        "results": results,
    }


def atomic_write_json(data, path=None):
    output_path = Path(path) if path is not None else OUTPUT_PATH
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Direktori output tidak ditemukan: {output_path.parent}")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def ensure_output_replacement_allowed(output_path, replace_existing=False):
    if output_path.exists() and not replace_existing:
        raise FileExistsError(
            f"{output_path} sudah berisi hasil sebelumnya (termasuk format legacy). "
            "File tidak ditimpa. Setelah menyetujui penggantian, jalankan dengan "
            "--replace-existing-results."
        )


def retrieve_from_collection(question, collection, embedding_fn, top_k=TOP_K):
    count = collection.count()
    if count == 0:
        return []
    candidate_k = min(max(top_k * 10, top_k), count)
    query_embedding = embedding_fn([question])
    response = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]
    return [
        {
            "content": content,
            "source": metadata.get("source"),
            "start_index": metadata.get("start_index"),
            "distance": distance,
        }
        for content, metadata, distance in zip(documents, metadatas, distances)
    ][:top_k]


def run_evaluation(dataset, top_k=TOP_K, results_path=None, replace_existing=False):
    output_path = Path(results_path) if results_path is not None else OUTPUT_PATH
    ensure_output_replacement_allowed(output_path, replace_existing)
    validate_dataset(dataset)

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from app.embedding import create_embeddings
    from app.retrieval import get_collection

    collection = get_collection()
    all_documents = collection.get(include=["documents"])["documents"]

    def retrieve(question, top_k=TOP_K):
        return retrieve_from_collection(
            question,
            collection,
            create_embeddings,
            top_k=top_k,
        )

    output = evaluate_dataset(dataset, retrieve, all_documents, top_k=top_k)
    atomic_write_json(output, output_path)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluasi retrieval terhadap reference_contexts.")
    parser.add_argument(
        "--replace-existing-results",
        action="store_true",
        help="Izinkan mengganti retrieval_results.json yang sudah ada.",
    )
    args = parser.parse_args(argv)
    dataset = load_dataset()
    output = run_evaluation(
        dataset,
        top_k=TOP_K,
        replace_existing=args.replace_existing_results,
    )
    summary = output["summary"]
    print(f"Pertanyaan: {summary['total_questions']}")
    print(
        f"Hit@1/3/5: {summary['hit_count_at_1']}/"
        f"{summary['hit_count_at_3']}/{summary['hit_count_at_5']}"
    )
    print(f"MRR@5: {summary['mrr_at_5']:.4f}")
    print(f"Hasil tersimpan: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
