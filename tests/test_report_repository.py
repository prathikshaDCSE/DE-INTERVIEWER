from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repository.report_repository import (
    InvalidReportError,
    ReportNotFoundError,
    ReportRecord,
    ReportRepository,
)


def _build_repository(mock_repository: MagicMock) -> ReportRepository:
    mock_repository.project = "test-project"
    mock_repository.dataset = "test_dataset"
    return ReportRepository(
        bigquery_repository=mock_repository,
        logger=logging.getLogger("report-repository-test"),
        config_path=Path("config"),
        valid_report_statuses={"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "DELETED"},
    )


def _report_row(
    *,
    report_id: str = "rep-001",
    session_id: str = "session-001",
    candidate_id: str = "candidate-001",
    candidate_name: str = "Jane Doe",
    candidate_email: str = "jane.doe@example.com",
    interviewer: str = "Reviewer A",
    interview_stage: str = "S1",
    overall_score: float = 4.5,
    overall_rating: str | None = "Strong",
    recommendation: str = "SELECT",
    strengths: str | None = "SQL, design",
    weaknesses: str | None = "Monitoring depth",
    feedback: str | None = "Solid interview",
    question_count: int = 10,
    questions_answered: int = 9,
    average_response_time: float | None = 38.2,
    interview_duration: float = 26.5,
    report_status: str = "COMPLETED",
    created_at: str = "2026-07-27T10:00:00+00:00",
    updated_at: str = "2026-07-27T10:15:00+00:00",
    created_by: str = "system",
    updated_by: str = "system",
) -> dict[str, object]:
    return {
        "report_id": report_id,
        "session_id": session_id,
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "candidate_email": candidate_email,
        "interviewer": interviewer,
        "interview_stage": interview_stage,
        "overall_score": overall_score,
        "overall_rating": overall_rating,
        "recommendation": recommendation,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "feedback": feedback,
        "question_count": question_count,
        "questions_answered": questions_answered,
        "average_response_time": average_response_time,
        "interview_duration": interview_duration,
        "report_status": report_status,
        "created_at": created_at,
        "updated_at": updated_at,
        "created_by": created_by,
        "updated_by": updated_by,
    }


def test_create_report_inserts_and_returns_record() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    payload = _report_row()

    mock_repository.fetch_one.side_effect = [
        {"total_count": 0},
        {"total_count": 0},
        _report_row(),
    ]

    created = repository.create_report(payload)

    assert isinstance(created, ReportRecord)
    assert created.report_id == "rep-001"
    assert mock_repository.insert.called


def test_get_report_returns_record() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = _report_row()

    record = repository.get_report("rep-001")

    assert record.report_id == "rep-001"
    assert record.candidate_email == "jane.doe@example.com"


def test_update_report_updates_and_returns_record() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.side_effect = [
        _report_row(),
        _report_row(overall_score=4.8, updated_by="admin"),
    ]
    mock_repository.update.return_value = 1

    updated = repository.update_report(
        "rep-001",
        updates={"overall_score": 4.8, "recommendation": "SELECT"},
        updated_by="admin",
    )

    assert updated.overall_score == 4.8
    assert mock_repository.update.called


def test_delete_report_soft_deletes_report() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.side_effect = [
        _report_row(),
        _report_row(report_status="DELETED", updated_by="admin"),
    ]
    mock_repository.update.return_value = 1

    deleted = repository.delete_report("rep-001", updated_by="admin")

    assert deleted.report_status == "DELETED"


def test_report_exists_returns_boolean() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 1}

    assert repository.report_exists(report_id="rep-001") is True


def test_list_reports_returns_paginated_records() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 1}
    mock_repository.fetch_all.return_value = [_report_row()]

    result = repository.list_reports(
        status="COMPLETED",
        stage="S1",
        page=1,
        page_size=10,
    )

    assert result["total"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0].report_id == "rep-001"


def test_search_reports_returns_matching_records() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 1}
    mock_repository.fetch_all.return_value = [_report_row(candidate_name="Jane Doe")]

    result = repository.search_reports("jane", page=1, page_size=10)

    assert result["total"] == 1
    assert result["items"][0].candidate_name == "Jane Doe"


def test_get_statistics_returns_aggregates() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {
        "total_reports": 3,
        "completed_reports": 2,
        "failed_reports": 1,
        "average_score": 4.2,
        "average_duration": 24.5,
        "average_questions": 9.0,
    }
    mock_repository.fetch_all.side_effect = [
        [
            {"grouping_key": "SELECT", "total_count": 2},
            {"grouping_key": "REJECT", "total_count": 1},
        ],
        [
            {"grouping_key": "S1", "total_count": 2},
            {"grouping_key": "S2", "total_count": 1},
        ],
        [
            {"created_day": "2026-07-26", "total_count": 1},
            {"created_day": "2026-07-27", "total_count": 2},
        ],
    ]

    result = repository.get_statistics()

    assert result["total_reports"] == 3
    assert result["completed_reports"] == 2
    assert result["recommendation_distribution"]["SELECT"] == 2
    assert result["reports_per_stage"]["S1"] == 2
    assert result["reports_per_day"]["2026-07-27"] == 2


def test_create_report_validates_score_range() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)

    with pytest.raises(InvalidReportError):
        repository.create_report(
            {
                **_report_row(overall_score=6.0),
            }
        )


def test_get_report_raises_when_missing() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = None

    with pytest.raises(ReportNotFoundError):
        repository.get_report("missing")
