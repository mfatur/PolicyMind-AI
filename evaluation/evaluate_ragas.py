
import asyncio
import importlib.metadata
import inspect
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(__file__).resolve().parent / "data" / "eval_dataset.json"
RESULTS_PATH = DATASET_PATH.parent / "ragas_results.json"

METRIC_NAMES = (
    "Faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
)
RESULT_FIELDS = {
    "answer",
    "contexts",
    "sources",
    "question",
    "evaluation_status",
    "errors",
    *METRIC_NAMES,
}
SUMMARY_FIELDS = {"average", "completed", "failed", "pending", "total"}
RESULT_STATUSES = {
    "pending",
    "in_progress",
    "partial_failure",
    "failed",
    "completed",
}
MAX_METRIC_ATTEMPTS = 2
MAX_RATE_LIMIT_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 3
RATE_LIMIT_RETRY_BUFFER_SECONDS = 1
MAX_RETRY_WAIT_SECONDS = 120
POST_METRIC_DELAY_SECONDS = 1
EXPECTED_RAGAS_VERSION = "0.4.3"
EXPECTED_OPENAI_VERSION = "2.54.0"


class DatasetValidationError(ValueError):
    pass


class ResultValidationError(ValueError):
    pass


class RetryExhausted(Exception):
    def __init__(self, cause, attempts):
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts


def validate_dataset(dataset):
    if not isinstance(dataset, list) or not dataset:
        raise DatasetValidationError("Dataset harus berupa list yang tidak kosong.")

    for index, item in enumerate(dataset, start=1):
        prefix = f"Dataset Q{index}"
        if not isinstance(item, Mapping):
            raise DatasetValidationError(f"{prefix} harus berupa object JSON.")

        for field in ("question", "ground_truth"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DatasetValidationError(
                    f"{prefix} memerlukan {field} berupa string non-kosong."
                )

        references = item.get("reference_contexts")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(value, str) or not value.strip() for value in references)
        ):
            raise DatasetValidationError(
                f"{prefix} memerlukan reference_contexts berupa list string non-kosong."
            )

        source = item.get("source")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise DatasetValidationError(
                f"{prefix}.source harus berupa string non-kosong jika tersedia."
            )

    return dataset


def load_dataset(path=None):
    dataset_path = Path(path) if path is not None else DATASET_PATH
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetValidationError(
            f"Dataset tidak ditemukan: {dataset_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            f"JSON dataset rusak pada baris {error.lineno}, kolom {error.colno}."
        ) from error
    return validate_dataset(dataset)


def is_valid_score(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def validate_summary(summary):
    if not isinstance(summary, Mapping):
        raise ResultValidationError("_summary harus berupa object JSON.")

    for metric_name, values in summary.items():
        if metric_name not in METRIC_NAMES or not isinstance(values, Mapping):
            raise ResultValidationError("Schema _summary tidak dikenal.")
        if set(values) - SUMMARY_FIELDS:
            raise ResultValidationError("Field _summary tidak dikenal.")
        if "average" not in values:
            raise ResultValidationError("_summary kehilangan field average.")
        average = values["average"]
        if average is not None and not is_valid_score(average):
            raise ResultValidationError("Nilai average pada _summary tidak valid.")
        for field in ("completed", "total"):
            value = values.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ResultValidationError(
                    f"Nilai {field} pada _summary tidak valid."
                )
        for field in ("failed", "pending"):
            if field in values:
                value = values[field]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ResultValidationError(
                        f"Nilai {field} pada _summary tidak valid."
                    )


def validate_results(results, dataset_count):
    if not isinstance(results, dict):
        raise ResultValidationError("Root hasil harus berupa object JSON.")

    for question_id, item in results.items():
        if question_id == "_summary":
            validate_summary(item)
            continue

        if (
            not isinstance(question_id, str)
            or not question_id.isdecimal()
            or str(int(question_id)) != question_id
            or not 1 <= int(question_id) <= dataset_count
        ):
            raise ResultValidationError(
                f"ID hasil tidak dikenal untuk dataset ini: {question_id!r}."
            )
        if not isinstance(item, dict):
            raise ResultValidationError(f"Hasil Q{question_id} harus berupa object.")
        unknown_fields = set(item) - RESULT_FIELDS
        if unknown_fields:
            raise ResultValidationError(
                f"Field hasil Q{question_id} tidak dikenal: "
                f"{', '.join(sorted(unknown_fields))}."
            )

        if "question" in item and not isinstance(item["question"], str):
            raise ResultValidationError(f"question Q{question_id} harus string.")
        if "answer" in item and (
            not isinstance(item["answer"], str) or not item["answer"].strip()
        ):
            raise ResultValidationError(f"answer Q{question_id} tidak valid.")
        if "contexts" in item and (
            not isinstance(item["contexts"], list)
            or any(not isinstance(value, str) for value in item["contexts"])
        ):
            raise ResultValidationError(f"contexts Q{question_id} tidak valid.")
        if "sources" in item and (
            not isinstance(item["sources"], list)
            or any(not isinstance(value, str) for value in item["sources"])
        ):
            raise ResultValidationError(f"sources Q{question_id} tidak valid.")
        if "evaluation_status" in item and item["evaluation_status"] not in RESULT_STATUSES:
            raise ResultValidationError(
                f"evaluation_status Q{question_id} tidak dikenal."
            )

        errors = item.get("errors", {})
        if not isinstance(errors, dict):
            raise ResultValidationError(f"errors Q{question_id} harus berupa object.")
        for name, detail in errors.items():
            if name not in (*METRIC_NAMES, "retrieval") or not isinstance(detail, dict):
                raise ResultValidationError(f"Error Q{question_id} tidak dikenal.")
            if set(detail) != {"type", "message", "attempts"}:
                raise ResultValidationError(
                    f"Schema errors Q{question_id}/{name} tidak dikenal."
                )
            if (
                not isinstance(detail["type"], str)
                or not isinstance(detail["message"], str)
                or not isinstance(detail["attempts"], int)
                or isinstance(detail["attempts"], bool)
                or detail["attempts"] < 1
            ):
                raise ResultValidationError(
                    f"Detail errors Q{question_id}/{name} tidak valid."
                )

        for metric_name in METRIC_NAMES:
            if metric_name in item and not is_valid_score(item[metric_name]):
                raise ResultValidationError(
                    f"Skor Q{question_id}/{metric_name} harus numerik, finite, "
                    "dan berada di antara 0 dan 1."
                )
            if metric_name in item and metric_name in errors:
                raise ResultValidationError(
                    f"Q{question_id}/{metric_name} memiliki skor sekaligus error."
                )

    return results


def load_saved_results(path=None, dataset_count=None):
    results_path = Path(path) if path is not None else RESULTS_PATH
    if not results_path.exists():
        return {}
    if results_path.stat().st_size == 0:
        raise ResultValidationError(
            f"File hasil kosong dan tidak akan ditimpa: {results_path}"
        )

    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResultValidationError(
            f"JSON hasil rusak pada baris {error.lineno}, kolom {error.colno}; "
            "file tidak ditimpa."
        ) from error

    if dataset_count is None:
        raise ValueError("dataset_count wajib diberikan untuk validasi hasil.")
    if not isinstance(results, dict) or not any(
        question_id != "_summary" for question_id in results
    ):
        raise ResultValidationError(
            f"File hasil tidak berisi record pertanyaan dan tidak akan ditimpa: "
            f"{results_path}"
        )
    return validate_results(results, dataset_count)


def entry_status(item):
    completed = sum(name in item and is_valid_score(item[name]) for name in METRIC_NAMES)
    errors = item.get("errors", {})
    if completed == len(METRIC_NAMES):
        return "completed"
    if errors:
        return "partial_failure" if completed else "failed"
    if completed or "answer" in item or "contexts" in item:
        return "in_progress"
    return "pending"


def build_summary(results, dataset_count):
    question_results = [
        item
        for question_id, item in results.items()
        if question_id != "_summary" and isinstance(item, Mapping)
    ]
    summary = {}

    for metric_name in METRIC_NAMES:
        scores = [
            float(item[metric_name])
            for item in question_results
            if metric_name in item and is_valid_score(item[metric_name])
        ]
        failed = sum(
            metric_name in item.get("errors", {})
            or ("retrieval" in item.get("errors", {}) and metric_name not in item)
            for item in question_results
            if metric_name not in item
        )
        completed = len(scores)
        summary[metric_name] = {
            "average": sum(scores) / completed if completed else None,
            "completed": completed,
            "failed": failed,
            "pending": max(dataset_count - completed - failed, 0),
            "total": dataset_count,
        }
    return summary


def save_results(results, dataset_count, path=None):
    results_path = Path(path) if path is not None else RESULTS_PATH
    validate_results(results, dataset_count)
    for question_id, item in results.items():
        if question_id != "_summary":
            item["evaluation_status"] = entry_status(item)
    results["_summary"] = build_summary(results, dataset_count)

    if not results_path.parent.is_dir():
        raise FileNotFoundError(
            f"Direktori hasil tidak ditemukan: {results_path.parent}"
        )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=results_path.parent,
            prefix=f".{results_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(results, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, results_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def safe_error_details(error, api_key=None, attempts=1):
    message = str(error)
    if api_key:
        message = message.replace(api_key, "[REDACTED_API_KEY]")
    message = re.sub(
        r"\b(?:gsk_|sk-)[A-Za-z0-9_-]{8,}\b",
        "[REDACTED_API_KEY]",
        message,
    )
    return {
        "type": type(error).__name__,
        "message": message[:500],
        "attempts": attempts,
    }


def error_status_code(error):
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status


def is_rate_limit_error(error):
    message = str(error).lower()
    return error_status_code(error) == 429 or any(
        marker in message
        for marker in (
            "rate limit",
            "rate_limit_exceeded",
            "tokens per day",
            "tokens per minute",
            "quota exceeded",
        )
    )


def parse_retry_after(value, now=None):
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value).strip())
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        seconds = (retry_at - current_time).total_seconds()

    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def retry_after_header(error):
    owners = (
        error,
        getattr(error, "response", None),
        getattr(error, "http_response", None),
    )
    for owner in owners:
        headers = getattr(owner, "headers", None)
        if isinstance(headers, Mapping):
            for name, value in headers.items():
                if str(name).lower() == "retry-after":
                    return value
    return None


def rate_limit_retry_delay(error):
    delay = parse_retry_after(retry_after_header(error))
    if delay is None:
        match = re.search(
            r"\b(?:try again|retry)\s+in\s+(\d+(?:\.\d+)?)\s*"
            r"(?:seconds?|secs?|s)\b",
            str(error),
            flags=re.IGNORECASE,
        )
        if match:
            delay = parse_retry_after(match.group(1))

    if delay is None:
        return None
    return delay + RATE_LIMIT_RETRY_BUFFER_SECONDS


def is_retryable_error(error):
    message = str(error).lower()
    error_type = type(error).__name__.lower()
    if isinstance(error, (ValueError, TypeError, KeyError, AssertionError)):
        return False
    if "validationerror" in error_type:
        return False
    if error_status_code(error) in {400, 401, 403, 404, 422}:
        return False
    if "tokens per day" in message or "quota exceeded" in message:
        return False
    if is_rate_limit_error(error):
        return True
    if error_status_code(error) in {408, 425, 429, 500, 502, 503, 504}:
        return True
    return isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError)) or any(
        marker in message
        for marker in (
            "temporarily unavailable",
            "connection reset",
            "timed out",
            "rate limit",
            "rate_limit_exceeded",
            "tokens per minute",
        )
    )


async def call_with_retries(call):
    attempt = 1
    while True:
        try:
            return await call(), attempt
        except Exception as error:
            if not is_retryable_error(error):
                raise RetryExhausted(error, attempt) from error

            if is_rate_limit_error(error):
                if attempt >= MAX_RATE_LIMIT_ATTEMPTS:
                    raise RetryExhausted(error, attempt) from error
                delay = rate_limit_retry_delay(error)
                if "tokens per minute" in str(error).lower():
                    delay = max(delay or 60, 60)
                elif delay is None:
                    delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                if delay > MAX_RETRY_WAIT_SECONDS:
                    raise RetryExhausted(error, attempt) from error
            else:
                if attempt >= MAX_METRIC_ATTEMPTS:
                    raise RetryExhausted(error, attempt) from error
                delay = RETRY_DELAY_SECONDS * attempt

            await asyncio.sleep(delay)
            attempt += 1


async def call_policy_with_retries(ask_policy, question):
    async def invoke():
        if inspect.iscoroutinefunction(ask_policy):
            return await ask_policy(question)
        result = await asyncio.to_thread(ask_policy, question)
        if inspect.isawaitable(result):
            return await result
        return result

    return await call_with_retries(invoke)


async def calculate_metric(metric_name, metric, sample, result):
    if metric_name == "Faithfulness":
        return await metric.ascore(
            user_input=sample["question"],
            response=result["answer"],
            retrieved_contexts=result["contexts"],
        )
    if metric_name == "Answer Relevancy":
        return await metric.ascore(
            user_input=sample["question"],
            response=result["answer"],
        )
    if metric_name in {"Context Precision", "Context Recall"}:
        return await metric.ascore(
            user_input=sample["question"],
            retrieved_contexts=result["contexts"],
            reference=sample["ground_truth"],
        )
    raise ValueError(f"Metrik tidak dikenal: {metric_name}")


def metric_value(score):
    value = getattr(score, "value", score)
    if not is_valid_score(value):
        raise ValueError("Evaluator mengembalikan skor non-numerik atau di luar 0..1.")
    return float(value)


def normalize_rag_result(result):
    if not isinstance(result, Mapping):
        raise ValueError("ask_policy harus mengembalikan object hasil.")
    answer = result.get("answer")
    contexts = result.get("contexts")
    sources = result.get("sources", [])
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Jawaban retrieval harus berupa string non-kosong.")
    if not isinstance(contexts, list) or any(
        not isinstance(context, str) for context in contexts
    ):
        raise ValueError("contexts retrieval harus berupa list string.")
    if not isinstance(sources, list) or any(
        not isinstance(source, str) for source in sources
    ):
        raise ValueError("sources retrieval harus berupa list string.")
    return {
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
    }


def record_error(item, name, error, api_key=None, attempts=1):
    item.setdefault("errors", {})[name] = safe_error_details(
        error,
        api_key=api_key,
        attempts=attempts,
    )


def clear_error(item, name):
    errors = item.get("errors")
    if not errors:
        return
    errors.pop(name, None)
    if not errors:
        item.pop("errors", None)


async def evaluate_dataset(
    dataset,
    saved_results,
    metrics,
    ask_policy,
    results_path=None,
    api_key=None,
):
    validate_dataset(dataset)
    validate_results(saved_results, len(dataset))

    for index, sample in enumerate(dataset, start=1):
        question_id = str(index)
        question_results = saved_results.setdefault(question_id, {})
        if all(
            metric_name in question_results
            and is_valid_score(question_results[metric_name])
            for metric_name in METRIC_NAMES
        ):
            print(f"Q{index}: empat metrik valid sudah tersimpan; dilewati.")
            continue

        print(f"\nEvaluasi Q{index}: {sample['question']}")
        if (
            isinstance(question_results.get("answer"), str)
            and question_results["answer"].strip()
            and isinstance(question_results.get("contexts"), list)
        ):
            result = {
                "answer": question_results["answer"],
                "contexts": question_results["contexts"],
            }
            print("Menggunakan answer dan contexts tersimpan.")
        else:
            try:
                raw_result, retrieval_attempts = await call_policy_with_retries(
                    ask_policy,
                    sample["question"],
                )
                result = normalize_rag_result(raw_result)
            except RetryExhausted as failure:
                error = failure.cause
                record_error(
                    question_results,
                    "retrieval",
                    error,
                    api_key,
                    attempts=failure.attempts,
                )
                save_results(saved_results, len(dataset), results_path)
                print(
                    f"Gagal Q{index}/retrieval: {type(error).__name__}: "
                    f"{safe_error_details(error, api_key, failure.attempts)['message']}"
                )
                if is_rate_limit_error(error):
                    print("Rate limit pada retrieval; evaluasi dihentikan.")
                    return saved_results
                continue
            except Exception as error:
                record_error(question_results, "retrieval", error, api_key)
                save_results(saved_results, len(dataset), results_path)
                print(
                    f"Gagal Q{index}/retrieval: {type(error).__name__}: "
                    f"{safe_error_details(error, api_key)['message']}"
                )
                if is_rate_limit_error(error):
                    print("Rate limit pada retrieval; evaluasi dihentikan.")
                    return saved_results
                continue

            question_results.update(result)
            question_results["question"] = sample["question"]
            clear_error(question_results, "retrieval")
            save_results(saved_results, len(dataset), results_path)

        for metric_name in METRIC_NAMES:
            if metric_name in question_results and is_valid_score(
                question_results[metric_name]
            ):
                print(f"{metric_name}: skor valid tersimpan; dilewati.")
                continue

            try:
                score, attempts = await call_with_retries(
                    lambda: calculate_metric(
                        metric_name,
                        metrics[metric_name],
                        sample,
                        result,
                    )
                )
                value = metric_value(score)
            except RetryExhausted as failure:
                error = failure.cause
                record_error(
                    question_results,
                    metric_name,
                    error,
                    api_key,
                    attempts=failure.attempts,
                )
                save_results(saved_results, len(dataset), results_path)
                message = safe_error_details(error, api_key)["message"]
                print(
                    f"Gagal Q{index}/{metric_name} setelah "
                    f"{failure.attempts} percobaan: {type(error).__name__}: {message}"
                )
                if is_rate_limit_error(error):
                    print("Rate limit terdeteksi; progres disimpan, evaluasi dihentikan.")
                    return saved_results
                continue
            except Exception as error:
                record_error(question_results, metric_name, error, api_key)
                save_results(saved_results, len(dataset), results_path)
                message = safe_error_details(error, api_key)["message"]
                print(
                    f"Gagal Q{index}/{metric_name}: "
                    f"{type(error).__name__}: {message}"
                )
                if is_rate_limit_error(error):
                    print("Rate limit terdeteksi; progres disimpan, evaluasi dihentikan.")
                    return saved_results
                continue

            question_results[metric_name] = value
            clear_error(question_results, metric_name)
            question_results["question"] = sample["question"]
            save_results(saved_results, len(dataset), results_path)
            print(f"{metric_name}: {value}")
            await asyncio.sleep(POST_METRIC_DELAY_SECONDS)

    save_results(saved_results, len(dataset), results_path)
    return saved_results


def require_runtime_config():
    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY belum dikonfigurasi; evaluasi dihentikan sebelum inisialisasi model."
        )

    for package, expected in (
        ("ragas", EXPECTED_RAGAS_VERSION),
        ("openai", EXPECTED_OPENAI_VERSION),
    ):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"Dependency {package} belum terpasang; periksa requirements.txt."
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"Versi {package} tidak sesuai: perlu {expected}, ditemukan {actual}."
            )
    return api_key


def build_evaluator(api_key):
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        max_retries=0,
        timeout=60,
    )
    judge_llm = llm_factory(
        "openai/gpt-oss-120b",
        client=client,
        adapter="instructor",
        max_tokens=2048,
        max_retries=0,
    )
    judge_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
    )
    metrics = {
        "Faithfulness": Faithfulness(llm=judge_llm),
        "Answer Relevancy": AnswerRelevancy(
            llm=judge_llm,
            embeddings=judge_embeddings,
        ),
        "Context Precision": ContextPrecision(llm=judge_llm),
        "Context Recall": ContextRecall(llm=judge_llm),
    }
    return metrics, client


def all_metrics_complete(results, dataset_count):
    return all(
        all(
            metric_name in results.get(str(index), {})
            and is_valid_score(results[str(index)][metric_name])
            for metric_name in METRIC_NAMES
        )
        for index in range(1, dataset_count + 1)
    )


async def main():
    dataset = load_dataset()
    saved_results = load_saved_results(RESULTS_PATH, len(dataset))

    if all_metrics_complete(saved_results, len(dataset)):
        save_results(saved_results, len(dataset), RESULTS_PATH)
        print("Semua pertanyaan telah memiliki empat skor valid; tidak ada panggilan API.")
        return

    api_key = require_runtime_config()
    metrics, client = build_evaluator(api_key)
    try:
        if str(PROJECT_DIR) not in sys.path:
            sys.path.insert(0, str(PROJECT_DIR))
        from app.rag_pipeline import ask_policy

        await evaluate_dataset(
            dataset,
            saved_results,
            metrics,
            ask_policy,
            RESULTS_PATH,
            api_key,
        )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
