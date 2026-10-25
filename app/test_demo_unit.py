import json
from io import BytesIO
from urllib.error import URLError

import demo


def test_ask_policy_posts_to_backend_and_formats_sources(monkeypatch):
    response = BytesIO(
        json.dumps({
            "answer": "Cuti tahunan diberikan sesuai kebijakan.",
            "sources": ["cuti.pdf"],
        }).encode("utf-8")
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return response

    monkeypatch.setattr(demo, "urlopen", fake_urlopen)

    answer = demo.ask_policy("  Berapa cuti tahunan?  ")

    request, timeout = calls[0]
    assert request.full_url == f"{demo.API_URL}/ask"
    assert request.method == "POST"
    assert json.loads(request.data) == {"question": "Berapa cuti tahunan?"}
    assert timeout == 60
    assert "Cuti tahunan diberikan sesuai kebijakan." in answer
    assert "`cuti.pdf`" in answer


def test_ask_policy_handles_empty_question_without_backend_call(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Backend should not be called for empty questions.")

    monkeypatch.setattr(demo, "urlopen", fail_if_called)

    assert demo.ask_policy("  ") == "Silakan masukkan pertanyaan terlebih dahulu."


def test_ask_policy_reports_unavailable_backend(monkeypatch):
    def fail_to_connect(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(demo, "urlopen", fail_to_connect)

    assert demo.ask_policy("Pertanyaan uji") == (
        "Backend tidak dapat dihubungi. Pastikan API sedang berjalan."
    )
