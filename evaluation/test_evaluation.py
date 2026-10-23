import asyncio
import json
from pathlib import Path

import pytest

from evaluation import evaluate_ragas, evaluate_retrieval


METRIC_NAMES = evaluate_ragas.METRIC_NAMES


def make_sample(question="Question?", reference="Answer context"):
    return {
        "question": question,
        "ground_truth": "Expected answer.",
        "reference_contexts": [reference],
        "source": "policy.pdf",
    }


def make_complete_result(value=0.5):
    return {
        "question": "Question?",
        "answer": "Answer.",
        "contexts": ["Answer context"],
        "sources": ["policy.pdf"],
        **{metric: value for metric in METRIC_NAMES},
    }


class FakeMetric:
    def __init__(self, value=0.25, error=None):
        self.value = value
        self.error = error
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.value


def fake_metrics(error_metric=None, error=None):
    return {
        name: FakeMetric(error=error if name == error_metric else None)
        for name in METRIC_NAMES
    }


def test_build_evaluator_caps_output_and_disables_nested_retries(monkeypatch):
    import openai
    import ragas.embeddings
    import ragas.llms
    import ragas.metrics.collections

    client_options = {}
    llm_options = {}

    class FakeClient:
        def __init__(self, **kwargs):
            client_options.update(kwargs)

        async def close(self):
            return None

    class FakeEmbeddings:
        def __init__(self, **_kwargs):
            pass

    class FakeMetricInstance:
        def __init__(self, **_kwargs):
            pass

    def fake_llm_factory(model, **kwargs):
        llm_options["model"] = model
        llm_options.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(ragas.embeddings, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(ragas.llms, "llm_factory", fake_llm_factory)
    for name in (
        "Faithfulness",
        "AnswerRelevancy",
        "ContextPrecision",
        "ContextRecall",
    ):
        monkeypatch.setattr(
            ragas.metrics.collections,
            name,
            FakeMetricInstance,
        )

    metrics, _client = evaluate_ragas.build_evaluator("test-only-key")

    assert set(metrics) == set(METRIC_NAMES)
    assert client_options["max_retries"] == 0
    assert llm_options["model"] == "openai/gpt-oss-120b"
    assert llm_options["max_tokens"] == 2048
    assert llm_options["max_retries"] == 0


def test_dataset_path_is_independent_of_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset = evaluate_ragas.load_dataset()

    assert len(dataset) == 30
    assert dataset[0]["question"]


def test_invalid_dataset_fails_before_evaluator_initialization(tmp_path, monkeypatch):
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(json.dumps([{"question": "Missing fields"}]))
    monkeypatch.setattr(evaluate_ragas, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(evaluate_ragas, "RESULTS_PATH", tmp_path / "results.json")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(evaluate_ragas, "load_dotenv", lambda *_args, **_kwargs: None)

    def unexpected_evaluator(_api_key):
        pytest.fail("Evaluator tidak boleh dibuat untuk dataset invalid.")

    monkeypatch.setattr(evaluate_ragas, "build_evaluator", unexpected_evaluator)

    with pytest.raises(evaluate_ragas.DatasetValidationError):
        asyncio.run(evaluate_ragas.main())


def test_missing_api_key_fails_before_evaluator_initialization(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([make_sample()]))
    monkeypatch.setattr(evaluate_ragas, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(evaluate_ragas, "RESULTS_PATH", tmp_path / "results.json")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(evaluate_ragas, "load_dotenv", lambda *_args, **_kwargs: None)

    def unexpected_evaluator(_api_key):
        pytest.fail("Evaluator tidak boleh dibuat tanpa API key.")

    monkeypatch.setattr(evaluate_ragas, "build_evaluator", unexpected_evaluator)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        asyncio.run(evaluate_ragas.main())


def test_empty_corrupt_and_unknown_result_files_are_not_overwritten(tmp_path):
    for content in (
        "",
        "{broken",
        "{}",
        json.dumps({"_summary": {}}),
        json.dumps({"legacy": {"score": 1}}),
    ):
        path = tmp_path / "results.json"
        path.write_text(content, encoding="utf-8")
        original = path.read_bytes()

        with pytest.raises(evaluate_ragas.ResultValidationError):
            evaluate_ragas.load_saved_results(path, dataset_count=1)

        assert path.read_bytes() == original


def test_atomic_ragas_save_preserves_existing_file_if_serialization_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "ragas_results.json"
    original = b"previous-valid-content"
    path.write_bytes(original)

    def fail_dump(*_args, **_kwargs):
        raise TypeError("serialization failed")

    monkeypatch.setattr(evaluate_ragas.json, "dump", fail_dump)

    with pytest.raises(TypeError, match="serialization failed"):
        evaluate_ragas.save_results({"1": {}}, dataset_count=1, path=path)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".ragas_results.json.*.tmp")) == []


def test_resume_skips_complete_questions_and_calculates_only_missing_metrics(
    tmp_path, monkeypatch
):
    dataset = [make_sample(f"Q{index}") for index in range(1, 4)]
    saved = {
        "1": make_complete_result(0.8),
        "2": {
            "question": "Q2",
            "answer": "Cached answer",
            "contexts": ["Cached context"],
            "Faithfulness": 0.7,
        },
    }
    metrics = fake_metrics()
    asked = []

    def ask_policy(question):
        asked.append(question)
        return {
            "answer": "Fresh answer",
            "contexts": ["Fresh context"],
            "sources": ["policy.pdf"],
        }

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(evaluate_ragas.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(evaluate_ragas, "POST_METRIC_DELAY_SECONDS", 0)
    path = tmp_path / "resume.json"

    result = asyncio.run(
        evaluate_ragas.evaluate_dataset(
            dataset,
            saved,
            metrics,
            ask_policy,
            results_path=path,
        )
    )

    assert asked == ["Q3"]
    assert len(metrics["Faithfulness"].calls) == 1
    assert all(
        len(metrics[name].calls) == 2
        for name in METRIC_NAMES
        if name != "Faithfulness"
    )
    assert result["1"]["Faithfulness"] == 0.8
    assert result["2"]["Faithfulness"] == 0.7
    assert result["2"]["Answer Relevancy"] == 0.25
    assert result["3"]["sources"] == ["policy.pdf"]
    assert result["_summary"]["Faithfulness"]["completed"] == 3
    assert json.loads(path.read_text(encoding="utf-8"))["2"]["Faithfulness"] == 0.7


def test_interleaved_summary_skips_q1_to_q10_and_resumes_q11(tmp_path, monkeypatch):
    dataset = [make_sample(f"Q{index}") for index in range(1, 12)]
    saved = {
        str(index): make_complete_result(0.8)
        for index in range(1, 8)
    }
    saved["_summary"] = evaluate_ragas.build_summary(saved, len(dataset))
    saved.update({
        str(index): make_complete_result(0.7)
        for index in range(8, 11)
    })
    saved["11"] = {
        "question": "Q11",
        "answer": "Saved answer",
        "contexts": ["Saved context"],
        "sources": ["policy.pdf"],
        "evaluation_status": "failed",
        "errors": {
            "Faithfulness": {
                "type": "InstructorRetryException",
                "message": "HTTP 429 rate limit",
                "attempts": 4,
            }
        },
    }
    metrics = fake_metrics()
    asked = []

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(evaluate_ragas.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(evaluate_ragas, "POST_METRIC_DELAY_SECONDS", 0)
    path = tmp_path / "interleaved-summary.json"

    result = asyncio.run(
        evaluate_ragas.evaluate_dataset(
            dataset,
            saved,
            metrics,
            lambda question: asked.append(question)
            or pytest.fail("Q11 answer/contexts tersimpan harus dipakai."),
            results_path=path,
        )
    )

    assert asked == []
    assert all(len(metrics[name].calls) == 1 for name in METRIC_NAMES)
    assert all(
        result[str(index)][name] == 0.8
        for index in range(1, 8)
        for name in METRIC_NAMES
    )
    assert all(
        result[str(index)][name] == 0.7
        for index in range(8, 11)
        for name in METRIC_NAMES
    )
    assert result["11"]["answer"] == "Saved answer"
    assert result["11"]["contexts"] == ["Saved context"]
    assert result["11"]["Faithfulness"] == 0.25
    assert "errors" not in result["11"]
    assert result["11"]["evaluation_status"] == "completed"
    assert result["_summary"]["Faithfulness"]["failed"] == 0
    assert list(json.loads(path.read_text(encoding="utf-8"))).index("_summary") == 7


def test_summary_counts_only_valid_scores_and_distinguishes_failure():
    summary = evaluate_ragas.build_summary(
        {
            "1": {"Faithfulness": 0.4},
            "2": {
                "errors": {
                    "Faithfulness": {
                        "type": "RateLimitError",
                        "message": "limited",
                        "attempts": 2,
                    }
                }
            },
        },
        dataset_count=3,
    )

    assert summary["Faithfulness"] == {
        "average": 0.4,
        "completed": 1,
        "failed": 1,
        "pending": 1,
        "total": 3,
    }
    assert summary["Context Recall"]["average"] is None
    assert summary["Context Recall"]["completed"] == 0


def test_summary_ignores_non_numeric_and_non_finite_values():
    summary = evaluate_ragas.build_summary(
        {
            "1": {"Faithfulness": 0.4},
            "2": {"Faithfulness": "0.9"},
            "3": {"Faithfulness": float("nan")},
            "4": {"Faithfulness": True},
        },
        dataset_count=4,
    )

    assert summary["Faithfulness"]["average"] == 0.4
    assert summary["Faithfulness"]["completed"] == 1


def test_sync_retrieval_retries_temporary_error_once():
    attempts = []

    class TemporaryError(Exception):
        status_code = 503

    def ask_policy(_question):
        attempts.append(1)
        if len(attempts) == 1:
            raise TemporaryError("temporarily unavailable")
        return {"answer": "answer", "contexts": ["context"]}

    async def no_sleep(_delay):
        return None

    async def run():
        return await evaluate_ragas.call_policy_with_retries(ask_policy, "Q?")

    original_sleep = evaluate_ragas.asyncio.sleep
    evaluate_ragas.asyncio.sleep = no_sleep
    try:
        result, count = asyncio.run(run())
    finally:
        evaluate_ragas.asyncio.sleep = original_sleep

    assert result["answer"] == "answer"
    assert count == 2
    assert len(attempts) == 2


def test_retry_after_header_takes_precedence_over_error_message():
    error = RuntimeError("Please try again in 3s")
    error.response = type(
        "Response",
        (),
        {"headers": {"Retry-After": "6.945"}},
    )()

    assert evaluate_ragas.rate_limit_retry_delay(error) == pytest.approx(7.945)


def test_rate_limit_message_delay_includes_buffer():
    error = RuntimeError("Please try again in 6.945s")

    assert evaluate_ragas.rate_limit_retry_delay(error) == pytest.approx(7.945)


def test_rate_limit_retry_count_is_bounded_and_uses_server_delay(monkeypatch):
    attempts = []
    delays = []

    class RateLimitError(Exception):
        status_code = 429

    async def no_sleep(delay):
        delays.append(delay)

    async def failing_call():
        attempts.append(1)
        raise RateLimitError("Please try again in 6.945s")

    monkeypatch.setattr(evaluate_ragas.asyncio, "sleep", no_sleep)

    with pytest.raises(evaluate_ragas.RetryExhausted) as raised:
        asyncio.run(evaluate_ragas.call_with_retries(failing_call))

    assert raised.value.attempts == evaluate_ragas.MAX_RATE_LIMIT_ATTEMPTS == 4
    assert len(attempts) == 4
    assert delays == pytest.approx([7.945] * 3)


def test_validation_and_permanent_errors_are_not_retried(monkeypatch):
    attempts = []

    async def no_sleep(_delay):
        pytest.fail("Error permanen tidak boleh memicu retry.")

    async def invalid_call():
        attempts.append(1)
        raise ValueError("invalid metric input; 429 is only text")

    monkeypatch.setattr(evaluate_ragas.asyncio, "sleep", no_sleep)

    with pytest.raises(evaluate_ragas.RetryExhausted) as raised:
        asyncio.run(evaluate_ragas.call_with_retries(invalid_call))

    assert raised.value.attempts == 1
    assert len(attempts) == 1


def test_api_key_is_redacted_from_persistable_error():
    api_key = "gsk_test_secret_value"
    detail = evaluate_ragas.safe_error_details(
        RuntimeError(f"request failed using {api_key}"),
        api_key=api_key,
    )

    assert api_key not in detail["message"]
    assert "[REDACTED_API_KEY]" in detail["message"]


def test_rate_limit_keeps_prior_progress_and_records_partial_failure(
    tmp_path, monkeypatch
):
    class RateLimitError(Exception):
        status_code = 429

    dataset = [make_sample("Q1"), make_sample("Q2")]
    saved = {
        "1": make_complete_result(0.9),
        "2": {
            "question": "Q2",
            "answer": "Cached answer",
            "contexts": ["Cached context"],
            "Faithfulness": 0.6,
        },
    }
    metrics = fake_metrics("Answer Relevancy", RateLimitError("429 rate limit"))
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(evaluate_ragas.asyncio, "sleep", no_sleep)
    path = tmp_path / "rate_limit.json"

    result = asyncio.run(
        evaluate_ragas.evaluate_dataset(
            dataset,
            saved,
            metrics,
            lambda _question: pytest.fail("Cached contexts should be reused."),
            results_path=path,
        )
    )

    assert len(metrics["Answer Relevancy"].calls) == evaluate_ragas.MAX_RATE_LIMIT_ATTEMPTS
    assert result["1"]["Faithfulness"] == 0.9
    assert result["2"]["Faithfulness"] == 0.6
    assert "Answer Relevancy" in result["2"]["errors"]
    assert result["2"]["errors"]["Answer Relevancy"]["attempts"] == 4
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["1"]["Context Recall"] == 0.9
    assert "Answer Relevancy" not in persisted["2"]
    assert persisted["_summary"]["Faithfulness"]["completed"] == 2


def test_retrieval_metrics_use_reference_contexts_not_source_labels():
    references = ["Annual leave is 12 days per year."]
    retrieved = [
        {"content": "Unrelated text", "source": "same.pdf", "distance": 0.1},
        {
            "content": "Policy: Annual leave is 12 days per year.",
            "source": "same.pdf",
            "distance": 0.2,
        },
    ]
    scored = evaluate_retrieval.score_retrieved_contexts(
        references,
        retrieved,
        [item["content"] for item in retrieved],
        top_k=5,
    )

    assert scored["hit_at_1"] is False
    assert scored["hit_at_3"] is True
    assert scored["hit_at_5"] is True
    assert scored["mrr_at_5"] == 0.5
    assert scored["evidence_status"] == "reference_context_found_in_retrieval"


def test_no_reference_match_is_not_interpreted_as_source_absence():
    scored = evaluate_retrieval.score_retrieved_contexts(
        ["Reference paraphrase not present verbatim"],
        [{"content": "Different text", "source": "same.pdf"}],
        ["Different text"],
    )

    assert scored["evidence_status"] == "no_exact_reference_context_match_in_index"
    assert scored["hit_at_5"] is False
    assert scored["mrr_at_5"] == 0.0


def test_retrieval_results_are_written_atomically(tmp_path, monkeypatch):
    path = tmp_path / "retrieval_results.json"
    original = b"previous-results"
    path.write_bytes(original)

    def fail_dump(*_args, **_kwargs):
        raise TypeError("serialization failed")

    monkeypatch.setattr(evaluate_retrieval.json, "dump", fail_dump)

    with pytest.raises(TypeError, match="serialization failed"):
        evaluate_retrieval.atomic_write_json({"schema_version": 2}, path)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".retrieval_results.json.*.tmp")) == []


def test_legacy_retrieval_results_require_explicit_replacement(tmp_path):
    path = tmp_path / "retrieval_results.json"
    path.write_text(
        json.dumps({"summary": {"hit_rate_at_3": 1.0}, "results": []}),
        encoding="utf-8",
    )
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="--replace-existing-results"):
        evaluate_retrieval.ensure_output_replacement_allowed(path)

    assert path.read_bytes() == original
    evaluate_retrieval.ensure_output_replacement_allowed(path, replace_existing=True)
