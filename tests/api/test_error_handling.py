"""
tests/api/test_error_handling.py

Tests verifying HTTP status code mapping and standardized error payload schemas.
"""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from exceptions.gemini_exceptions import GeminiTimeoutError
from app.dependencies import get_container


def test_missing_required_fields_returns_422(api_client: TestClient) -> None:
    response = api_client.post("/api/v1/interviews/start", json={})
    assert response.status_code == 422


def test_invalid_session_id_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/interviews/invalid_session_9999/question")
    assert response.status_code in (400, 404)
    payload = response.json()
    assert payload["success"] is False


def test_unknown_route_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/unknown_route_that_does_not_exist")
    assert response.status_code == 404


def test_unsupported_method_returns_405(api_client: TestClient) -> None:
    response = api_client.put("/api/v1/interviews/start", json={})
    assert response.status_code == 405


def test_gemini_timeout_returns_504(api_client: TestClient) -> None:
    container = get_container()
    container.gemini_service.generate_json = MagicMock(
        side_effect=GeminiTimeoutError("Request timed out", timeout_seconds=5)
    )

    start_resp = api_client.post(
        "/api/v1/interviews/start",
        json={"candidate_name": "Eve", "experience": 3.0},
    )
    session_id = start_resp.json()["data"]["session_id"]

    response = api_client.post(
        f"/api/v1/interviews/{session_id}/answer",
        json={"answer": "Some answer text"},
    )
    assert response.status_code == 504
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "SERVICE_TIMEOUT"
