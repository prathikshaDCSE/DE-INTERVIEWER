"""
tests/api/test_report_api.py

Tests for report generation and retrieval API endpoints.
"""

from fastapi.testclient import TestClient


def test_complete_interview_and_get_report_api(api_client: TestClient) -> None:
    start_resp = api_client.post(
        "/api/v1/interviews/start",
        json={"candidate_name": "Dave", "experience": 3.0},
    )
    session_id = start_resp.json()["data"]["session_id"]

    for _ in range(15):
        ans_resp = api_client.post(
            f"/api/v1/interviews/{session_id}/answer",
            json={"answer": "Detailed answer explaining technical concept."},
        )
        if ans_resp.json()["data"]["is_completed"]:
            break

    complete_resp = api_client.post(f"/api/v1/interviews/{session_id}/complete")
    assert complete_resp.status_code == 200
    complete_payload = complete_resp.json()
    assert complete_payload["success"] is True
    report_data = complete_payload["data"]
    report_id = report_data["report_id"]

    fetch_resp = api_client.get(f"/api/v1/reports/{report_id}")
    assert fetch_resp.status_code == 200
    fetch_payload = fetch_resp.json()
    assert fetch_payload["success"] is True
    assert fetch_payload["data"]["report_id"] == report_id


def test_get_nonexistent_report_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/reports/nonexistent_report_id_12345")
    assert response.status_code == 404
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] in ("SESSION_NOT_FOUND", "QUESTION_NOT_FOUND", "NOT_FOUND", "404")
