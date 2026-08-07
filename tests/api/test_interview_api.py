"""
tests/api/test_interview_api.py

Unit and integration tests for interview API lifecycle endpoints.
"""

from fastapi.testclient import TestClient


def test_start_interview_api(api_client: TestClient) -> None:
    request_body = {
        "candidate_name": "Jane Smith",
        "experience": 4.5,
        "candidate_level": "Senior",
        "target_role": "Data Engineer",
    }
    response = api_client.post("/api/v1/interviews/start", json=request_body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert "session_id" in data
    assert "question_id" in data
    assert len(data["question"]) > 0


def test_get_current_question_api(api_client: TestClient) -> None:
    start_resp = api_client.post(
        "/api/v1/interviews/start",
        json={"candidate_name": "Bob", "experience": 2.0},
    )
    session_id = start_resp.json()["data"]["session_id"]

    response = api_client.get(f"/api/v1/interviews/{session_id}/question")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["session_id"] == session_id


def test_submit_answer_api(api_client: TestClient) -> None:
    start_resp = api_client.post(
        "/api/v1/interviews/start",
        json={"candidate_name": "Alice", "experience": 3.0},
    )
    session_id = start_resp.json()["data"]["session_id"]

    answer_body = {"answer": "Partitioning divides data by key values to prune scans."}
    response = api_client.post(
        f"/api/v1/interviews/{session_id}/answer", json=answer_body
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "next_action" in payload["data"]
    assert "evaluation" in payload["data"]


def test_get_interview_progress_api(api_client: TestClient) -> None:
    start_resp = api_client.post(
        "/api/v1/interviews/start",
        json={"candidate_name": "Charlie", "experience": 5.0},
    )
    session_id = start_resp.json()["data"]["session_id"]

    response = api_client.get(f"/api/v1/interviews/{session_id}/progress")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["completed_questions"] == 0
    assert payload["data"]["remaining_questions"] > 0
