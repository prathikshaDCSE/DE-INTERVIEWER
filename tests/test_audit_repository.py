from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from repository.audit_repository import (
    AuditRecord,
    AuditRecordNotFoundError,
    AuditRepository,
    InvalidAuditRecordError,
)


def _build_repository(mock_repository: MagicMock) -> AuditRepository:
    mock_repository.project = "test-project"
    mock_repository.dataset = "test_dataset"
    return AuditRepository(
        bigquery_repository=mock_repository,
        logger=logging.getLogger("audit-repository-test"),
        valid_severities={"INFO", "WARNING", "ERROR", "SECURITY"},
        valid_statuses={"SUCCESS", "FAILURE"},
    )


def _audit_row(
    *,
    audit_id: str = "audit-001",
    user_id: str | None = "user-001",
    action: str = "LOGIN",
    entity: str | None = "candidate_sessions",
    entity_id: str | None = "session-001",
    description: str | None = "User logged in",
    ip_address: str | None = "127.0.0.1",
    status: str = "SUCCESS",
    severity: str = "INFO",
    timestamp: str = "2026-07-27T10:00:00+00:00",
) -> dict[str, object]:
    return {
        "audit_id": audit_id,
        "user_id": user_id,
        "action": action,
        "entity": entity,
        "entity_id": entity_id,
        "description": description,
        "ip_address": ip_address,
        "status": status,
        "severity": severity,
        "timestamp": timestamp,
    }


def test_log_event_inserts_and_returns_record() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    payload = _audit_row()
    mock_repository.fetch_one.return_value = _audit_row()

    created = repository.log_event(payload)

    assert isinstance(created, AuditRecord)
    assert created.audit_id == "audit-001"
    assert mock_repository.insert.called


def test_get_audit_returns_record() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = _audit_row()

    record = repository.get_audit("audit-001")

    assert record.action == "LOGIN"
    assert record.user_id == "user-001"


def test_get_user_audits_returns_paginated_records() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 1}
    mock_repository.fetch_all.return_value = [_audit_row(user_id="user-001")]

    result = repository.get_user_audits(user_id="user-001", page=1, page_size=10)

    assert result["total"] == 1
    assert result["items"][0].user_id == "user-001"


def test_get_candidate_audits_returns_paginated_records() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 1}
    mock_repository.fetch_all.return_value = [_audit_row(entity="candidate", entity_id="cand-001")]

    result = repository.get_candidate_audits(
        candidate_id="cand-001",
        page=1,
        page_size=10,
    )

    assert result["total"] == 1
    assert result["items"][0].entity_id == "cand-001"


def test_search_audits_returns_matching_records() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 1}
    mock_repository.fetch_all.return_value = [_audit_row(description="User logged in")]

    result = repository.search_audits("logged", page=1, page_size=10)

    assert result["total"] == 1
    assert result["items"][0].description == "User logged in"


def test_list_audits_returns_paginated_records() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 1}
    mock_repository.fetch_all.return_value = [_audit_row(severity="ERROR", status="FAILURE")]

    result = repository.list_audits(
        severity="ERROR",
        status="FAILURE",
        page=1,
        page_size=10,
    )

    assert result["total"] == 1
    assert result["items"][0].severity == "ERROR"


def test_get_statistics_returns_aggregates() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {
        "total_events": 4,
        "success_events": 3,
        "failure_events": 1,
    }
    mock_repository.fetch_all.side_effect = [
        [{"grouping_key": "candidate_sessions", "total_count": 2}],
        [{"grouping_key": "INFO", "total_count": 3}],
        [{"grouping_key": "user-001", "total_count": 2}],
        [{"event_day": "2026-07-27", "total_count": 4}],
        [{"grouping_key": "LOGIN", "total_count": 2}],
    ]

    result = repository.get_statistics()

    assert result["total_events"] == 4
    assert result["success_rate"] == 75.0
    assert result["failure_rate"] == 25.0
    assert result["events_by_category"]["candidate_sessions"] == 2
    assert result["most_common_actions"]["LOGIN"] == 2


def test_count_audits_returns_integer() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 7}

    assert repository.count_audits(status="SUCCESS") == 7


def test_audit_exists_returns_boolean() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = {"total_count": 1}

    assert repository.audit_exists("audit-001") is True


def test_get_audit_raises_when_missing() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)
    mock_repository.fetch_one.return_value = None

    with pytest.raises(AuditRecordNotFoundError):
        repository.get_audit("missing")


def test_log_event_validates_status() -> None:
    mock_repository = MagicMock()
    repository = _build_repository(mock_repository)

    with pytest.raises(InvalidAuditRecordError):
        repository.log_event({**_audit_row(status="PENDING")})
