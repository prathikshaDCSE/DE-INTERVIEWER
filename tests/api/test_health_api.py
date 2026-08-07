"""
tests/api/test_health_api.py

Integration tests for health and runtime configuration endpoints.
"""

from fastapi.testclient import TestClient


def test_get_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] in ("healthy", "degraded")
    assert "services" in payload["data"]
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers


def test_get_config_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["question_count"] > 0
    assert "gemini_model" in payload["data"]
