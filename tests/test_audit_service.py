from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from repository.audit_repository import (
    AuditRecord,
    AuditRecordNotFoundError,
    AuditRepositoryError,
    InvalidAuditRecordError,
)
from services.audit_service import (
    AuditExportError,
    AuditNotFoundError,
    AuditOperationError,
    AuditService,
    AuditServiceError,
    AuditValidationError,
)


# ============================================================
# Fixtures / Helpers
# ============================================================


def make_service() -> tuple[AuditService, MagicMock]:
    mock_repo = MagicMock()
    mock_repo.REQUIRED_FIELDS = frozenset({"action", "status", "severity"})

    service = AuditService(
        audit_repository=mock_repo,
        logger=logging.getLogger("audit-service-test"),
    )
    return service, mock_repo


def make_record(
    *,
    audit_id: str = "audit-1",
    user_id: str | None = "user-1",
    action: str = "LOGIN",
    entity: str | None = "candidate",
    entity_id: str | None = "cand-1",
    description: str | None = "User logged in",
    ip_address: str | None = "127.0.0.1",
    status: str = "SUCCESS",
    severity: str = "INFO",
    timestamp: datetime | None = None,
) -> AuditRecord:
    return AuditRecord(
        audit_id=audit_id,
        user_id=user_id,
        action=action,
        entity=entity,
        entity_id=entity_id,
        description=description,
        ip_address=ip_address,
        status=status,
        severity=severity,
        timestamp=timestamp or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def make_page(items: list[AuditRecord], total: int | None = None) -> dict:
    return {
        "items": items,
        "page": 1,
        "page_size": max(len(items), 1),
        "total": total if total is not None else len(items),
        "total_pages": 1,
        "has_next": False,
        "has_previous": False,
    }


VALID_PAYLOAD = {
    "action": "LOGIN",
    "status": "SUCCESS",
    "severity": "INFO",
    "user_id": "user-1",
    "entity": "candidate",
    "entity_id": "cand-1",
    "description": "User logged in",
}


# ============================================================
# log_event
# ============================================================


def test_log_event_success_delegates_and_returns_record():
    service, mock_repo = make_service()
    expected_record = make_record()
    mock_repo.log_event.return_value = expected_record

    record = service.log_event(VALID_PAYLOAD)

    assert record is expected_record
    mock_repo.log_event.assert_called_once_with(VALID_PAYLOAD)


def test_log_event_rejects_non_mapping_payload():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.log_event(["not", "a", "mapping"])  # type: ignore[arg-type]

    mock_repo.log_event.assert_not_called()


@pytest.mark.parametrize("missing_field", ["action", "status", "severity"])
def test_log_event_rejects_missing_required_field(missing_field):
    service, mock_repo = make_service()
    payload = dict(VALID_PAYLOAD)
    payload.pop(missing_field)

    with pytest.raises(AuditValidationError):
        service.log_event(payload)

    mock_repo.log_event.assert_not_called()


def test_log_event_rejects_blank_required_field():
    service, mock_repo = make_service()
    payload = dict(VALID_PAYLOAD)
    payload["action"] = "   "

    with pytest.raises(AuditValidationError):
        service.log_event(payload)

    mock_repo.log_event.assert_not_called()


def test_log_event_translates_invalid_audit_record_error():
    service, mock_repo = make_service()
    mock_repo.log_event.side_effect = InvalidAuditRecordError("bad severity")

    with pytest.raises(AuditValidationError):
        service.log_event(VALID_PAYLOAD)


def test_log_event_translates_generic_repository_error():
    service, mock_repo = make_service()
    mock_repo.log_event.side_effect = AuditRepositoryError("insert failed")

    with pytest.raises(AuditOperationError):
        service.log_event(VALID_PAYLOAD)


# ============================================================
# bulk_log_events
# ============================================================


def test_bulk_log_events_rejects_empty_sequence():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.bulk_log_events([])

    mock_repo.log_event.assert_not_called()


def test_bulk_log_events_success_calls_log_event_per_payload():
    service, mock_repo = make_service()
    record_1 = make_record(audit_id="a1")
    record_2 = make_record(audit_id="a2")
    mock_repo.log_event.side_effect = [record_1, record_2]

    results = service.bulk_log_events([VALID_PAYLOAD, VALID_PAYLOAD])

    assert results == [record_1, record_2]
    assert mock_repo.log_event.call_count == 2


def test_bulk_log_events_stop_on_error_true_raises_on_first_failure():
    service, mock_repo = make_service()
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload.pop("action")

    with pytest.raises(AuditValidationError):
        service.bulk_log_events([VALID_PAYLOAD, bad_payload], stop_on_error=True)


def test_bulk_log_events_stop_on_error_false_skips_failures():
    service, mock_repo = make_service()
    good_record = make_record(audit_id="good")
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload.pop("action")

    mock_repo.log_event.return_value = good_record

    results = service.bulk_log_events(
        [bad_payload, VALID_PAYLOAD],
        stop_on_error=False,
    )

    assert results == [good_record]
    mock_repo.log_event.assert_called_once_with(VALID_PAYLOAD)


def test_bulk_log_events_stop_on_error_false_skips_repository_failures():
    service, mock_repo = make_service()
    mock_repo.log_event.side_effect = AuditRepositoryError("boom")

    results = service.bulk_log_events(
        [VALID_PAYLOAD, VALID_PAYLOAD],
        stop_on_error=False,
    )

    assert results == []


# ============================================================
# get_event / get_events / event_exists
# ============================================================


def test_get_event_success():
    service, mock_repo = make_service()
    expected_record = make_record(audit_id="audit-9")
    mock_repo.get_audit.return_value = expected_record

    record = service.get_event("audit-9")

    assert record is expected_record
    mock_repo.get_audit.assert_called_once_with("audit-9")


def test_get_event_rejects_blank_id():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_event("   ")

    mock_repo.get_audit.assert_not_called()


def test_get_event_translates_not_found_error():
    service, mock_repo = make_service()
    mock_repo.get_audit.side_effect = AuditRecordNotFoundError("missing")

    with pytest.raises(AuditNotFoundError):
        service.get_event("missing-id")


def test_get_event_translates_generic_repository_error():
    service, mock_repo = make_service()
    mock_repo.get_audit.side_effect = AuditRepositoryError("query failed")

    with pytest.raises(AuditOperationError):
        service.get_event("audit-1")


def test_get_events_success():
    service, mock_repo = make_service()
    records = [make_record(audit_id="a1"), make_record(audit_id="a2")]
    mock_repo.get_audits.return_value = records

    result = service.get_events(["a1", "a2"])

    assert result == records
    mock_repo.get_audits.assert_called_once_with(["a1", "a2"])


def test_get_events_rejects_empty_sequence():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_events([])

    mock_repo.get_audits.assert_not_called()


def test_get_events_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.get_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.get_events(["a1"])


def test_event_exists_true_and_false():
    service, mock_repo = make_service()
    mock_repo.audit_exists.return_value = True
    assert service.event_exists("audit-1") is True

    mock_repo.audit_exists.return_value = False
    assert service.event_exists("audit-1") is False


def test_event_exists_rejects_blank_id():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.event_exists("")

    mock_repo.audit_exists.assert_not_called()


def test_event_exists_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.audit_exists.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.event_exists("audit-1")


# ============================================================
# search_events / get_user_history
# ============================================================


def test_search_events_success():
    service, mock_repo = make_service()
    page = make_page([make_record()])
    mock_repo.search_audits.return_value = page

    result = service.search_events("logged", page=1, page_size=10)

    assert result is page
    mock_repo.search_audits.assert_called_once_with("logged", page=1, page_size=10)


def test_search_events_rejects_blank_term():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.search_events("   ")

    mock_repo.search_audits.assert_not_called()


@pytest.mark.parametrize("page,page_size", [(0, 10), (1, 0), (1, 501)])
def test_search_events_rejects_invalid_pagination(page, page_size):
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.search_events("term", page=page, page_size=page_size)

    mock_repo.search_audits.assert_not_called()


def test_search_events_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.search_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.search_events("term")


def test_get_user_history_success():
    service, mock_repo = make_service()
    page = make_page([make_record(user_id="user-1")])
    mock_repo.get_user_audits.return_value = page

    result = service.get_user_history("user-1", page=1, page_size=25)

    assert result is page
    mock_repo.get_user_audits.assert_called_once_with(
        user_id="user-1", page=1, page_size=25
    )


def test_get_user_history_rejects_blank_user_id():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_user_history("")

    mock_repo.get_user_audits.assert_not_called()


def test_get_user_history_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.get_user_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.get_user_history("user-1")


# ============================================================
# get_action_history
# ============================================================


def test_get_action_history_filters_exact_matches_case_insensitively():
    service, mock_repo = make_service()
    matching_1 = make_record(audit_id="a1", action="LOGIN")
    matching_2 = make_record(audit_id="a2", action="login")
    non_matching = make_record(audit_id="a3", action="LOGIN_ATTEMPT")
    mock_repo.search_audits.return_value = make_page(
        [matching_1, matching_2, non_matching],
        total=10,
    )

    result = service.get_action_history("LOGIN", page=1, page_size=50)

    assert result["items"] == [matching_1, matching_2]
    assert result["matched_count"] == 2
    assert result["total"] == 10
    mock_repo.search_audits.assert_called_once_with("LOGIN", page=1, page_size=50)


def test_get_action_history_no_matches_returns_empty_items():
    service, mock_repo = make_service()
    mock_repo.search_audits.return_value = make_page(
        [make_record(action="OTHER")],
        total=1,
    )

    result = service.get_action_history("LOGIN")

    assert result["items"] == []
    assert result["matched_count"] == 0


def test_get_action_history_rejects_blank_action():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_action_history("   ")

    mock_repo.search_audits.assert_not_called()


def test_get_action_history_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.search_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.get_action_history("LOGIN")


# ============================================================
# get_entity_history
# ============================================================


@pytest.mark.parametrize("entity_alias", ["candidate"])
def test_get_entity_history_candidate_dispatches_to_candidate_audits(entity_alias):
    service, mock_repo = make_service()
    records = [make_record(entity="candidate", entity_id="cand-1")]
    mock_repo.get_candidate_audits.return_value = make_page(records)

    result = service.get_entity_history(entity_alias, "cand-1", limit=100)

    assert result == records
    mock_repo.get_candidate_audits.assert_called_once_with(
        candidate_id="cand-1", page=1, page_size=100
    )


@pytest.mark.parametrize(
    "entity_alias", ["session", "candidate_sessions", "interview_session"]
)
def test_get_entity_history_session_dispatches_to_session_audits(entity_alias):
    service, mock_repo = make_service()
    records = [make_record(entity="candidate_sessions", entity_id="sess-1")]
    mock_repo.get_session_audits.return_value = records

    result = service.get_entity_history(entity_alias, "sess-1")

    assert result == records
    mock_repo.get_session_audits.assert_called_once_with("sess-1")


@pytest.mark.parametrize("entity_alias", ["report", "final_reports", "final_report"])
def test_get_entity_history_report_dispatches_to_report_audits(entity_alias):
    service, mock_repo = make_service()
    records = [make_record(entity="final_reports", entity_id="report-1")]
    mock_repo.get_report_audits.return_value = records

    result = service.get_entity_history(entity_alias, "report-1")

    assert result == records
    mock_repo.get_report_audits.assert_called_once_with("report-1")


def test_get_entity_history_unknown_entity_falls_back_to_list_audits():
    service, mock_repo = make_service()
    records = [make_record(entity="custom_entity", entity_id="custom-1")]
    mock_repo.list_audits.return_value = make_page(records)

    result = service.get_entity_history("custom_entity", "custom-1", limit=50)

    assert result == records
    mock_repo.list_audits.assert_called_once_with(
        entity="custom_entity", entity_id="custom-1", page=1, page_size=50
    )


def test_get_entity_history_rejects_blank_entity():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_entity_history("", "cand-1")


def test_get_entity_history_rejects_blank_entity_id():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_entity_history("candidate", "")


@pytest.mark.parametrize("limit", [0, 1001])
def test_get_entity_history_rejects_out_of_range_limit(limit):
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_entity_history("candidate", "cand-1", limit=limit)


def test_get_entity_history_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.get_session_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.get_entity_history("session", "sess-1")


# ============================================================
# get_recent_events
# ============================================================


def test_get_recent_events_success():
    service, mock_repo = make_service()
    records = [make_record(audit_id="a1"), make_record(audit_id="a2")]
    mock_repo.list_audits.return_value = make_page(records)

    result = service.get_recent_events(limit=10)

    assert result == records
    mock_repo.list_audits.assert_called_once_with(page=1, page_size=10)


def test_get_recent_events_default_limit():
    service, mock_repo = make_service()
    mock_repo.list_audits.return_value = make_page([])

    service.get_recent_events()

    mock_repo.list_audits.assert_called_once_with(page=1, page_size=10)


@pytest.mark.parametrize("limit", [0, 501, -1])
def test_get_recent_events_rejects_out_of_range_limit(limit):
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_recent_events(limit=limit)

    mock_repo.list_audits.assert_not_called()


def test_get_recent_events_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.list_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.get_recent_events()


# ============================================================
# get_events_page
# ============================================================


def test_get_events_page_success_passes_filters_through():
    service, mock_repo = make_service()
    page = make_page([make_record()])
    mock_repo.list_audits.return_value = page

    result = service.get_events_page(
        page=2,
        page_size=25,
        severity="ERROR",
        status="FAILURE",
        user_id="user-1",
        entity="candidate",
        entity_id="cand-1",
    )

    assert result is page
    mock_repo.list_audits.assert_called_once_with(
        page=2,
        page_size=25,
        created_from=None,
        created_to=None,
        severity="ERROR",
        category=None,
        status="FAILURE",
        user_id="user-1",
        candidate_id=None,
        entity="candidate",
        entity_id="cand-1",
    )


def test_get_events_page_rejects_invalid_date_order():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.get_events_page(
            created_from=date(2026, 6, 1),
            created_to=date(2026, 1, 1),
        )

    mock_repo.list_audits.assert_not_called()


def test_get_events_page_skips_date_validation_for_string_bounds():
    service, mock_repo = make_service()
    mock_repo.list_audits.return_value = make_page([])

    service.get_events_page(
        created_from="2026-06-01",
        created_to="2026-01-01",
    )

    mock_repo.list_audits.assert_called_once()


def test_get_events_page_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.list_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.get_events_page()


# ============================================================
# count_events
# ============================================================


def test_count_events_success():
    service, mock_repo = make_service()
    mock_repo.count_audits.return_value = 42

    result = service.count_events(status="SUCCESS")

    assert result == 42
    mock_repo.count_audits.assert_called_once_with(
        created_from=None,
        created_to=None,
        severity=None,
        category=None,
        status="SUCCESS",
        user_id=None,
        candidate_id=None,
        entity_id=None,
    )


def test_count_events_rejects_invalid_date_order():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.count_events(
            created_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
            created_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    mock_repo.count_audits.assert_not_called()


def test_count_events_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.count_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.count_events()


# ============================================================
# get_statistics
# ============================================================


def test_get_statistics_success():
    service, mock_repo = make_service()
    stats = {"total_events": 5}
    mock_repo.get_statistics.return_value = stats

    result = service.get_statistics()

    assert result is stats


def test_get_statistics_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.get_statistics.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.get_statistics()


# ============================================================
# generate_timeline
# ============================================================


def test_generate_timeline_orders_chronologically():
    service, mock_repo = make_service()
    newest = make_record(
        audit_id="newest",
        action="C",
        timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )
    middle = make_record(
        audit_id="middle",
        action="B",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    oldest = make_record(
        audit_id="oldest",
        action="A",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    # Repository returns most-recent-first, as documented.
    mock_repo.list_audits.return_value = make_page([newest, middle, oldest])

    timeline = service.generate_timeline(page_size=100)

    assert [entry["action"] for entry in timeline] == ["A", "B", "C"]
    assert timeline[0]["timestamp"] == oldest.timestamp.isoformat()


def test_generate_timeline_entry_contains_expected_fields():
    service, mock_repo = make_service()
    record = make_record()
    mock_repo.list_audits.return_value = make_page([record])

    timeline = service.generate_timeline()

    entry = timeline[0]
    assert entry["action"] == record.action
    assert entry["description"] == record.description
    assert entry["user_id"] == record.user_id
    assert entry["entity"] == record.entity
    assert entry["entity_id"] == record.entity_id
    assert entry["status"] == record.status
    assert entry["severity"] == record.severity


def test_generate_timeline_rejects_invalid_date_order():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.generate_timeline(
            created_from=date(2026, 6, 1),
            created_to=date(2026, 1, 1),
        )

    mock_repo.list_audits.assert_not_called()


@pytest.mark.parametrize("page_size", [0, 1001])
def test_generate_timeline_rejects_invalid_page_size(page_size):
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.generate_timeline(page_size=page_size)

    mock_repo.list_audits.assert_not_called()


def test_generate_timeline_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.list_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.generate_timeline()


# ============================================================
# export_events
# ============================================================


def test_export_events_success_structure():
    service, mock_repo = make_service()
    record = make_record()
    mock_repo.list_audits.return_value = make_page([record], total=1)

    export_payload = service.export_events(
        severity="INFO",
        status="SUCCESS",
        page_size=100,
    )

    assert export_payload["total_matching"] == 1
    assert export_payload["exported_count"] == 1
    assert export_payload["events"] == [record.to_dict()]
    assert export_payload["filters"]["severity"] == "INFO"
    assert export_payload["filters"]["status"] == "SUCCESS"
    assert "generated_at" in export_payload


def test_export_events_serializes_date_filters_as_iso_strings():
    service, mock_repo = make_service()
    mock_repo.list_audits.return_value = make_page([])

    export_payload = service.export_events(
        created_from=date(2026, 1, 1),
        created_to=date(2026, 1, 31),
    )

    assert export_payload["filters"]["created_from"] == date(2026, 1, 1).isoformat()
    assert export_payload["filters"]["created_to"] == date(2026, 1, 31).isoformat()


def test_export_events_rejects_page_size_over_max():
    service, mock_repo = make_service()

    with pytest.raises(AuditExportError):
        service.export_events(page_size=5001)

    mock_repo.list_audits.assert_not_called()


def test_export_events_rejects_invalid_date_order():
    service, mock_repo = make_service()

    with pytest.raises(AuditValidationError):
        service.export_events(
            created_from=date(2026, 6, 1),
            created_to=date(2026, 1, 1),
        )

    mock_repo.list_audits.assert_not_called()


def test_export_events_translates_repository_error():
    service, mock_repo = make_service()
    mock_repo.list_audits.side_effect = AuditRepositoryError("boom")

    with pytest.raises(AuditOperationError):
        service.export_events()


# ============================================================
# delete_old_events
# ============================================================


def test_delete_old_events_always_raises_operation_error():
    service, mock_repo = make_service()

    with pytest.raises(AuditOperationError):
        service.delete_old_events(retention_days=90)


# ============================================================
# Repository exception translation (direct)
# ============================================================


def test_translate_repository_exception_not_found():
    service, mock_repo = make_service()
    translated = service._translate_repository_exception(
        AuditRecordNotFoundError("missing")
    )
    assert isinstance(translated, AuditNotFoundError)


def test_translate_repository_exception_invalid_record():
    service, mock_repo = make_service()
    translated = service._translate_repository_exception(
        InvalidAuditRecordError("bad")
    )
    assert isinstance(translated, AuditValidationError)


def test_translate_repository_exception_generic_repository_error():
    service, mock_repo = make_service()
    translated = service._translate_repository_exception(
        AuditRepositoryError("boom")
    )
    assert isinstance(translated, AuditOperationError)


def test_translate_repository_exception_unknown_error_falls_back_to_base():
    service, mock_repo = make_service()
    translated = service._translate_repository_exception(ValueError("weird"))
    assert isinstance(translated, AuditServiceError)
    assert not isinstance(translated, (AuditValidationError, AuditOperationError))


# ============================================================
# Additional edge cases
# ============================================================


def test_bulk_log_events_single_payload():
    service, mock_repo = make_service()
    record = make_record(audit_id="solo")
    mock_repo.log_event.return_value = record

    results = service.bulk_log_events([VALID_PAYLOAD])

    assert results == [record]
    mock_repo.log_event.assert_called_once_with(VALID_PAYLOAD)


def test_bulk_log_events_succeeds_after_earlier_failures_stop_on_error_false():
    service, mock_repo = make_service()
    bad_payload_1 = dict(VALID_PAYLOAD)
    bad_payload_1.pop("action")
    bad_payload_2 = dict(VALID_PAYLOAD)
    bad_payload_2.pop("status")
    good_record = make_record(audit_id="good-after-failures")

    mock_repo.log_event.return_value = good_record

    results = service.bulk_log_events(
        [bad_payload_1, bad_payload_2, VALID_PAYLOAD],
        stop_on_error=False,
    )

    assert results == [good_record]
    mock_repo.log_event.assert_called_once_with(VALID_PAYLOAD)


def test_bulk_log_events_preserves_input_order():
    service, mock_repo = make_service()
    record_1 = make_record(audit_id="order-1")
    record_2 = make_record(audit_id="order-2")
    record_3 = make_record(audit_id="order-3")
    mock_repo.log_event.side_effect = [record_1, record_2, record_3]

    payload_1 = dict(VALID_PAYLOAD, action="FIRST")
    payload_2 = dict(VALID_PAYLOAD, action="SECOND")
    payload_3 = dict(VALID_PAYLOAD, action="THIRD")

    results = service.bulk_log_events([payload_1, payload_2, payload_3])

    assert results == [record_1, record_2, record_3]
    called_payloads = [call.args[0] for call in mock_repo.log_event.call_args_list]
    assert called_payloads == [payload_1, payload_2, payload_3]


def test_get_events_passes_through_duplicate_ids():
    service, mock_repo = make_service()
    duplicate_record = make_record(audit_id="a1")
    records = [duplicate_record, duplicate_record, make_record(audit_id="a2")]
    mock_repo.get_audits.return_value = records

    result = service.get_events(["a1", "a1", "a2"])

    assert result == records
    mock_repo.get_audits.assert_called_once_with(["a1", "a1", "a2"])


@pytest.mark.parametrize(
    "page_size",
    [AuditService.DEFAULT_PAGE_SIZE, AuditService.MAX_PAGE_SIZE],
)
def test_search_events_accepts_boundary_page_sizes(page_size):
    service, mock_repo = make_service()
    mock_repo.search_audits.return_value = make_page([])

    service.search_events("term", page=1, page_size=page_size)

    mock_repo.search_audits.assert_called_once_with(
        "term", page=1, page_size=page_size
    )


def test_get_recent_events_accepts_max_limit_boundary():
    service, mock_repo = make_service()
    mock_repo.list_audits.return_value = make_page([])

    service.get_recent_events(limit=AuditService.MAX_RECENT_LIMIT)

    mock_repo.list_audits.assert_called_once_with(
        page=1, page_size=AuditService.MAX_RECENT_LIMIT
    )


def test_generate_timeline_empty_result():
    service, mock_repo = make_service()
    mock_repo.list_audits.return_value = make_page([])

    timeline = service.generate_timeline()

    assert timeline == []


def test_generate_timeline_single_record():
    service, mock_repo = make_service()
    record = make_record(audit_id="only-one")
    mock_repo.list_audits.return_value = make_page([record])

    timeline = service.generate_timeline()

    assert len(timeline) == 1
    assert timeline[0]["action"] == record.action


def test_generate_timeline_duplicate_timestamps_preserve_reversed_order():
    service, mock_repo = make_service()
    same_timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    record_b = make_record(audit_id="b", action="B", timestamp=same_timestamp)
    record_a = make_record(audit_id="a", action="A", timestamp=same_timestamp)
    # Repository returns most-recent-first; with equal timestamps the
    # relative order is whatever the repository provided.
    mock_repo.list_audits.return_value = make_page([record_b, record_a])

    timeline = service.generate_timeline()

    assert [entry["action"] for entry in timeline] == ["A", "B"]
    assert timeline[0]["timestamp"] == timeline[1]["timestamp"]


def test_export_events_zero_records():
    service, mock_repo = make_service()
    mock_repo.list_audits.return_value = make_page([], total=0)

    export_payload = service.export_events()

    assert export_payload["events"] == []
    assert export_payload["exported_count"] == 0
    assert export_payload["total_matching"] == 0


def test_count_events_returns_zero():
    service, mock_repo = make_service()
    mock_repo.count_audits.return_value = 0

    result = service.count_events()

    assert result == 0