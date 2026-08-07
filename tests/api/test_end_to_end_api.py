"""
tests/api/test_end_to_end_api.py

Full HTTP end-to-end integration test walking through complete candidate interview lifecycle.
"""

from fastapi.testclient import TestClient


def test_full_http_interview_end_to_end(api_client: TestClient) -> None:
    # 1. Health Check
    health_resp = api_client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["success"] is True

    # 2. Start Interview
    start_req = {
        "candidate_name": "Full Lifecycle Candidate",
        "experience": 4.0,
        "candidate_level": "Senior",
        "target_role": "Data Engineer",
    }
    start_resp = api_client.post("/api/v1/interviews/start", json=start_req)
    assert start_resp.status_code == 200
    session_id = start_resp.json()["data"]["session_id"]
    assert len(session_id) > 0

    # 3. Answer Questions
    turns = 0
    is_complete = False
    while not is_complete and turns < 15:
        turns += 1
        ans_resp = api_client.post(
            f"/api/v1/interviews/{session_id}/answer",
            json={"answer": "High quality answer covering architecture, indexing, and partitioning."},
        )
        assert ans_resp.status_code == 200
        data = ans_resp.json()["data"]
        is_complete = data["is_completed"]

    # 4. Check Progress
    prog_resp = api_client.get(f"/api/v1/interviews/{session_id}/progress")
    assert prog_resp.status_code == 200
    assert prog_resp.json()["data"]["completed_questions"] > 0

    # 5. Complete Interview & Generate Report
    comp_resp = api_client.post(f"/api/v1/interviews/{session_id}/complete")
    assert comp_resp.status_code == 200
    report_data = comp_resp.json()["data"]
    report_id = report_data["report_id"]
    assert report_data["decision"] in ("SELECT", "HOLD", "REJECT")

    # 6. Retrieve Report
    rep_resp = api_client.get(f"/api/v1/reports/{report_id}")
    assert rep_resp.status_code == 200
    assert rep_resp.json()["data"]["report_id"] == report_id
