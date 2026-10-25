from fastapi.testclient import TestClient

from main import app


FRONTEND_ORIGIN = "https://mfatur-policymind-ai-demo.static.hf.space"


def test_cors_preflight_allows_frontend_post_json():
    client = TestClient(app)

    response = client.options(
        "/ask",
        headers={
            "Origin": FRONTEND_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == FRONTEND_ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in response.headers


def test_cors_preflight_rejects_other_origins():
    client = TestClient(app)

    response = client.options(
        "/ask",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
