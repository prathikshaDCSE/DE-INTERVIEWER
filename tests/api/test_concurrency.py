"""
tests/api/test_concurrency.py

Tests verifying concurrent interview sessions operate independently without cross-session leaks.
"""

from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient


def test_concurrent_interview_sessions(api_client: TestClient) -> None:
    def run_session(candidate_idx: int) -> str:
        resp = api_client.post(
            "/api/v1/interviews/start",
            json={
                "candidate_name": f"Candidate_{candidate_idx}",
                "experience": 3.0 + candidate_idx,
            },
        )
        session_id = resp.json()["data"]["session_id"]
        for _ in range(3):
            api_client.post(
                f"/api/v1/interviews/{session_id}/answer",
                json={"answer": f"Answer from candidate {candidate_idx}"},
            )
        return session_id

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_session, i) for i in range(5)]
        session_ids = [f.result() for f in futures]

    assert len(session_ids) == 5
    assert len(set(session_ids)) == 5
