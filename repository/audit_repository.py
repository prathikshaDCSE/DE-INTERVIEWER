from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from math import ceil
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from repository.bigquery_repository import (
    BigQueryRepository,
    BigQueryRepositoryError,
    InsertError,
    QueryExecutionError,
)

EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
AUDIT_TABLE = "audit_logs"


class AuditRepositoryError(Exception):
    """Raised when the audit repository encounters a general failure."""


class AuditRecordNotFoundError(AuditRepositoryError):
    """Raised when a requested audit record cannot be found."""


class InvalidAuditRecordError(AuditRepositoryError):
    """Raised when audit record input fails validation."""


@dataclass(frozen=True)
class AuditRecord:
    """Immutable representation of a persisted audit event."""

    audit_id: str
    user_id: str | None
    action: str
    entity: str | None
    entity_id: str | None
    description: str | None
    ip_address: str | None
    status: str
    severity: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation of the audit record."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class AuditRepository:
    """Repository responsible for audit log persistence in BigQuery."""

    DEFAULT_SEVERITIES = frozenset({"INFO", "WARNING", "ERROR", "SECURITY"})
    DEFAULT_STATUSES = frozenset({"SUCCESS", "FAILURE"})
    REQUIRED_FIELDS = frozenset({"action", "status", "severity"})

    def __init__(
        self,
        bigquery_repository: BigQueryRepository,
        logger: logging.Logger | None = None,
        valid_severities: Iterable[str] | None = None,
        valid_statuses: Iterable[str] | None = None,
    ) -> None:
        """
        Initialize the repository.

        Args:
            bigquery_repository: Shared BigQuery repository dependency.
            logger: Optional logger instance.
            valid_severities: Optional supported severity values.
            valid_statuses: Optional supported status values.
        """
        self._repository = bigquery_repository
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._valid_severities = {
            self._normalize_required_string(value, "severity").upper()
            for value in (valid_severities or self.DEFAULT_SEVERITIES)
        }
        self._valid_statuses = {
            self._normalize_required_string(value, "status").upper()
            for value in (valid_statuses or self.DEFAULT_STATUSES)
        }

    def log_event(self, payload: Mapping[str, Any]) -> AuditRecord:
        """
        Persist a new audit event.

        Args:
            payload: Audit event payload.

        Returns:
            The created audit record.
        """
        if not isinstance(payload, Mapping):
            raise InvalidAuditRecordError("payload must be a mapping of audit attributes")

        with self._lock:
            prepared_payload = self._prepare_create_payload(payload)
            try:
                self._repository.insert(AUDIT_TABLE, prepared_payload)
            except (InsertError, BigQueryRepositoryError) as exc:
                self._log_error("log_event_failed", exc, audit_id=prepared_payload["audit_id"])
                raise AuditRepositoryError("Failed to log audit event") from exc

            record = self.get_audit(prepared_payload["audit_id"])
            self.logger.info(
                "Audit event created",
                extra={
                    "event": "audit_created",
                    "audit_id": record.audit_id,
                    "action": record.action,
                    "severity": record.severity,
                },
            )
            return record

    def get_audit(self, audit_id: str) -> AuditRecord:
        """
        Retrieve an audit event by identifier.

        Args:
            audit_id: Unique audit identifier.

        Returns:
            The matching audit record.
        """
        audit_id_clean = self._normalize_required_string(audit_id, "audit_id")
        sql = (
            "SELECT audit_id, user_id, action, entity, entity_id, description, "
            "ip_address, status, severity, timestamp "
            f"FROM {self._table_sql()} "
            "WHERE audit_id = @audit_id "
            "LIMIT 1"
        )
        try:
            row = self._repository.fetch_one(sql, parameters={"audit_id": audit_id_clean})
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("get_audit_failed", exc, audit_id=audit_id_clean)
            raise AuditRepositoryError("Failed to fetch audit record") from exc

        if row is None:
            raise AuditRecordNotFoundError(f"Audit record not found: {audit_id_clean}")
        self.logger.info(
            "Audit event retrieved",
            extra={"event": "audit_read", "audit_id": audit_id_clean},
        )
        return self._row_to_record(row)

    def get_audits(self, audit_ids: Sequence[str]) -> list[AuditRecord]:
        """
        Retrieve multiple audit events by identifier.

        Args:
            audit_ids: Collection of audit identifiers.

        Returns:
            Matching audit records.
        """
        normalized_ids = [
            self._normalize_required_string(audit_id, "audit_id")
            for audit_id in audit_ids
        ]
        if not normalized_ids:
            return []

        sql = (
            "SELECT audit_id, user_id, action, entity, entity_id, description, "
            "ip_address, status, severity, timestamp "
            f"FROM {self._table_sql()} "
            "WHERE audit_id IN UNNEST(@audit_ids) "
            "ORDER BY timestamp DESC, audit_id ASC"
        )
        try:
            rows = self._repository.fetch_all(sql, parameters={"audit_ids": normalized_ids})
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("get_audits_failed", exc)
            raise AuditRepositoryError("Failed to fetch audit records") from exc
        return [self._row_to_record(row) for row in rows]

    def get_user_audits(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        List audit events for a user.

        Args:
            user_id: User identifier.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and matching audit records.
        """
        return self.list_audits(user_id=user_id, page=page, page_size=page_size)

    def get_candidate_audits(
        self,
        candidate_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        List audit events for a candidate entity identifier.

        Args:
            candidate_id: Candidate identifier stored as entity_id.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and matching audit records.
        """
        return self.list_audits(
            entity="candidate",
            entity_id=candidate_id,
            page=page,
            page_size=page_size,
        )

    def get_session_audits(self, session_id: str) -> list[AuditRecord]:
        """
        Retrieve audit events for an interview session entity identifier.

        Args:
            session_id: Interview session identifier.

        Returns:
            Matching audit records.
        """
        return self._get_by_entity("candidate_sessions", session_id)

    def get_report_audits(self, report_id: str) -> list[AuditRecord]:
        """
        Retrieve audit events for a final report entity identifier.

        Args:
            report_id: Interview report identifier.

        Returns:
            Matching audit records.
        """
        return self._get_by_entity("final_reports", report_id)

    def search_audits(
        self,
        search_term: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        Search audit events by action, description, entity, entity id, or email-like text.

        Args:
            search_term: Free-text search input.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and matching audit records.
        """
        normalized_search = self._normalize_required_string(search_term, "search_term")
        normalized_page, normalized_page_size, offset = self._validate_pagination(
            page,
            page_size,
        )
        search_pattern = f"%{normalized_search.lower()}%"
        predicate = (
            "LOWER(action) LIKE @search_pattern "
            "OR LOWER(IFNULL(description, '')) LIKE @search_pattern "
            "OR LOWER(IFNULL(entity, '')) LIKE @search_pattern "
            "OR LOWER(IFNULL(entity_id, '')) LIKE @search_pattern"
        )
        count_sql = (
            "SELECT COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            f"WHERE {predicate}"
        )
        sql = (
            "SELECT audit_id, user_id, action, entity, entity_id, description, "
            "ip_address, status, severity, timestamp "
            f"FROM {self._table_sql()} "
            f"WHERE {predicate} "
            "ORDER BY timestamp DESC, audit_id ASC "
            "LIMIT @limit OFFSET @offset"
        )
        count_parameters = {"search_pattern": search_pattern}
        list_parameters = {
            "search_pattern": search_pattern,
            "limit": normalized_page_size,
            "offset": offset,
        }

        try:
            total_row = self._repository.fetch_one(count_sql, parameters=count_parameters)
            rows = self._repository.fetch_all(sql, parameters=list_parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("search_audits_failed", exc, search_term=normalized_search)
            raise AuditRepositoryError("Failed to search audit records") from exc

        total = int(total_row["total_count"]) if total_row else 0
        return self._build_paginated_response(
            rows=rows,
            page=normalized_page,
            page_size=normalized_page_size,
            total=total,
        )

    def list_audits(
        self,
        page: int = 1,
        page_size: int = 50,
        created_from: date | datetime | str | None = None,
        created_to: date | datetime | str | None = None,
        severity: str | None = None,
        category: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        candidate_id: str | None = None,
        entity: str | None = None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        """
        List audit records using filters and offset-based pagination.

        Args:
            page: Page number starting at 1.
            page_size: Number of records per page.
            created_from: Optional inclusive timestamp lower bound.
            created_to: Optional inclusive timestamp upper bound.
            severity: Optional severity filter.
            category: Optional alias for entity filter.
            status: Optional status filter.
            user_id: Optional user identifier filter.
            candidate_id: Optional candidate entity identifier filter.
            entity: Optional entity type filter.
            entity_id: Optional entity identifier filter.

        Returns:
            Pagination metadata and matching audit records.
        """
        normalized_page, normalized_page_size, offset = self._validate_pagination(
            page,
            page_size,
        )
        resolved_entity = entity or category
        resolved_entity_id = entity_id
        if candidate_id is not None:
            resolved_entity = "candidate"
            resolved_entity_id = candidate_id

        where_clause, parameters = self._build_filter_clause(
            created_from=created_from,
            created_to=created_to,
            severity=severity,
            status=status,
            user_id=user_id,
            entity=resolved_entity,
            entity_id=resolved_entity_id,
        )
        total = self.count_audits(
            created_from=created_from,
            created_to=created_to,
            severity=severity,
            category=resolved_entity,
            status=status,
            user_id=user_id,
            candidate_id=candidate_id,
            entity_id=resolved_entity_id,
        )
        sql = (
            "SELECT audit_id, user_id, action, entity, entity_id, description, "
            "ip_address, status, severity, timestamp "
            f"FROM {self._table_sql()} "
            f"{where_clause} "
            "ORDER BY timestamp DESC, audit_id ASC "
            "LIMIT @limit OFFSET @offset"
        )
        parameters.update({"limit": normalized_page_size, "offset": offset})

        try:
            rows = self._repository.fetch_all(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("list_audits_failed", exc)
            raise AuditRepositoryError("Failed to list audit records") from exc

        return self._build_paginated_response(
            rows=rows,
            page=normalized_page,
            page_size=normalized_page_size,
            total=total,
        )

    def count_audits(
        self,
        created_from: date | datetime | str | None = None,
        created_to: date | datetime | str | None = None,
        severity: str | None = None,
        category: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        candidate_id: str | None = None,
        entity_id: str | None = None,
    ) -> int:
        """
        Count audit records using the same filters as list_audits.

        Args:
            created_from: Optional inclusive timestamp lower bound.
            created_to: Optional inclusive timestamp upper bound.
            severity: Optional severity filter.
            category: Optional entity type filter.
            status: Optional status filter.
            user_id: Optional user identifier filter.
            candidate_id: Optional candidate entity identifier filter.
            entity_id: Optional explicit entity identifier filter.

        Returns:
            Matching audit record count.
        """
        resolved_entity = category
        resolved_entity_id = entity_id
        if candidate_id is not None:
            resolved_entity = "candidate"
            resolved_entity_id = candidate_id

        where_clause, parameters = self._build_filter_clause(
            created_from=created_from,
            created_to=created_to,
            severity=severity,
            status=status,
            user_id=user_id,
            entity=resolved_entity,
            entity_id=resolved_entity_id,
        )
        sql = (
            "SELECT COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            f"{where_clause}"
        )
        try:
            row = self._repository.fetch_one(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("count_audits_failed", exc)
            raise AuditRepositoryError("Failed to count audit records") from exc
        return int(row["total_count"]) if row else 0

    def audit_exists(self, audit_id: str) -> bool:
        """
        Check whether an audit record exists.

        Args:
            audit_id: Unique audit identifier.

        Returns:
            True if the audit record exists, else False.
        """
        audit_id_clean = self._normalize_required_string(audit_id, "audit_id")
        sql = (
            "SELECT COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            "WHERE audit_id = @audit_id"
        )
        try:
            row = self._repository.fetch_one(sql, parameters={"audit_id": audit_id_clean})
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("audit_exists_failed", exc, audit_id=audit_id_clean)
            raise AuditRepositoryError("Failed to check audit existence") from exc
        return bool(row and int(row["total_count"]) > 0)

    def get_statistics(self) -> dict[str, Any]:
        """
        Return aggregated audit statistics.

        Returns:
            Summary counts and distributions for audit events.
        """
        base_sql = (
            "SELECT COUNT(1) AS total_events, "
            "COUNTIF(status = 'SUCCESS') AS success_events, "
            "COUNTIF(status = 'FAILURE') AS failure_events "
            f"FROM {self._table_sql()}"
        )
        try:
            base_row = self._repository.fetch_one(base_sql) or {}
            events_by_category = self._group_statistics("entity")
            events_by_severity = self._group_statistics("severity")
            events_by_user = self._group_statistics("user_id")
            events_per_day = self._group_by_day()
            most_common_actions = self._group_statistics("action")
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("get_statistics_failed", exc)
            raise AuditRepositoryError("Failed to compute audit statistics") from exc

        total_events = int(base_row.get("total_events", 0) or 0)
        success_events = int(base_row.get("success_events", 0) or 0)
        failure_events = int(base_row.get("failure_events", 0) or 0)
        success_rate = round((success_events / total_events) * 100, 2) if total_events else 0.0
        failure_rate = round((failure_events / total_events) * 100, 2) if total_events else 0.0

        statistics = {
            "total_events": total_events,
            "events_by_category": events_by_category,
            "events_by_severity": events_by_severity,
            "events_by_user": events_by_user,
            "events_per_day": events_per_day,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "most_common_actions": most_common_actions,
        }
        self.logger.info(
            "Audit statistics generated",
            extra={"event": "audit_statistics_generated", "total_events": total_events},
        )
        return statistics

    def _prepare_create_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        missing_fields = [
            field
            for field in sorted(self.REQUIRED_FIELDS)
            if payload.get(field) is None or str(payload.get(field)).strip() == ""
        ]
        if missing_fields:
            raise InvalidAuditRecordError(
                f"Missing required audit fields: {', '.join(missing_fields)}"
            )

        timestamp_value = self._normalize_datetime(payload.get("timestamp", self._utcnow()))
        prepared_payload = {
            "audit_id": self._normalize_optional_string(payload.get("audit_id")) or str(uuid4()),
            "user_id": self._normalize_optional_string(payload.get("user_id")),
            "action": self._normalize_required_string(payload["action"], "action"),
            "entity": self._normalize_optional_string(payload.get("entity")),
            "entity_id": self._normalize_optional_string(payload.get("entity_id")),
            "description": self._normalize_optional_string(payload.get("description")),
            "ip_address": self._normalize_optional_string(payload.get("ip_address")),
            "status": self._normalize_status(payload["status"]),
            "severity": self._normalize_severity(payload["severity"]),
            "timestamp": timestamp_value.isoformat(),
        }
        return prepared_payload

    def _build_filter_clause(
        self,
        created_from: date | datetime | str | None = None,
        created_to: date | datetime | str | None = None,
        severity: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        entity: str | None = None,
        entity_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        parameters: dict[str, Any] = {}

        if created_from is not None:
            parameters["created_from"] = self._normalize_datetime(created_from).isoformat()
            clauses.append("timestamp >= TIMESTAMP(@created_from)")

        if created_to is not None:
            parameters["created_to"] = self._normalize_datetime(created_to).isoformat()
            clauses.append("timestamp <= TIMESTAMP(@created_to)")

        if created_from is not None and created_to is not None:
            if parameters["created_from"] > parameters["created_to"]:
                raise InvalidAuditRecordError(
                    "created_from must be less than or equal to created_to"
                )

        if severity is not None:
            parameters["severity"] = self._normalize_severity(severity)
            clauses.append("severity = @severity")

        if status is not None:
            parameters["status"] = self._normalize_status(status)
            clauses.append("status = @status")

        if user_id is not None:
            parameters["user_id"] = self._normalize_required_string(user_id, "user_id")
            clauses.append("user_id = @user_id")

        if entity is not None:
            parameters["entity"] = self._normalize_required_string(entity, "entity")
            clauses.append("LOWER(entity) = LOWER(@entity)")

        if entity_id is not None:
            parameters["entity_id"] = self._normalize_required_string(entity_id, "entity_id")
            clauses.append("entity_id = @entity_id")

        if not clauses:
            return "", parameters
        return "WHERE " + " AND ".join(clauses), parameters

    def _build_paginated_response(
        self,
        rows: Sequence[Mapping[str, Any]],
        page: int,
        page_size: int,
        total: int,
    ) -> dict[str, Any]:
        return {
            "items": [self._row_to_record(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": ceil(total / page_size) if total else 0,
            "has_next": (page - 1) * page_size + page_size < total,
            "has_previous": page > 1,
        }

    def _group_statistics(self, column_name: str) -> dict[str, int]:
        sql = (
            f"SELECT COALESCE(CAST({column_name} AS STRING), 'UNKNOWN') AS grouping_key, "
            "COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            f"GROUP BY {column_name} "
            "ORDER BY total_count DESC, grouping_key ASC"
        )
        rows = self._repository.fetch_all(sql)
        return {str(row["grouping_key"]): int(row["total_count"]) for row in rows}

    def _group_by_day(self) -> dict[str, int]:
        sql = (
            "SELECT CAST(DATE(timestamp) AS STRING) AS event_day, COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            "GROUP BY DATE(timestamp) "
            "ORDER BY DATE(timestamp) ASC"
        )
        rows = self._repository.fetch_all(sql)
        return {str(row["event_day"]): int(row["total_count"]) for row in rows}

    def _get_by_entity(self, entity: str, entity_id: str) -> list[AuditRecord]:
        entity_clean = self._normalize_required_string(entity, "entity")
        entity_id_clean = self._normalize_required_string(entity_id, "entity_id")
        sql = (
            "SELECT audit_id, user_id, action, entity, entity_id, description, "
            "ip_address, status, severity, timestamp "
            f"FROM {self._table_sql()} "
            "WHERE LOWER(entity) = LOWER(@entity) AND entity_id = @entity_id "
            "ORDER BY timestamp DESC, audit_id ASC"
        )
        try:
            rows = self._repository.fetch_all(
                sql,
                parameters={"entity": entity_clean, "entity_id": entity_id_clean},
            )
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error(
                "get_by_entity_failed",
                exc,
                entity=entity_clean,
                entity_id=entity_id_clean,
            )
            raise AuditRepositoryError("Failed to fetch audit records") from exc
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: Mapping[str, Any]) -> AuditRecord:
        return AuditRecord(
            audit_id=self._normalize_required_string(row.get("audit_id"), "audit_id"),
            user_id=self._normalize_optional_string(row.get("user_id")),
            action=self._normalize_required_string(row.get("action"), "action"),
            entity=self._normalize_optional_string(row.get("entity")),
            entity_id=self._normalize_optional_string(row.get("entity_id")),
            description=self._normalize_optional_string(row.get("description")),
            ip_address=self._normalize_optional_string(row.get("ip_address")),
            status=self._normalize_status(row.get("status")),
            severity=self._normalize_severity(row.get("severity")),
            timestamp=self._normalize_datetime(row.get("timestamp")),
        )

    def _normalize_severity(self, value: Any) -> str:
        severity = self._normalize_required_string(value, "severity").upper()
        if severity not in self._valid_severities:
            raise InvalidAuditRecordError(f"Invalid severity: {severity}")
        return severity

    def _normalize_status(self, value: Any) -> str:
        status = self._normalize_required_string(value, "status").upper()
        if status not in self._valid_statuses:
            raise InvalidAuditRecordError(f"Invalid status: {status}")
        return status

    def _normalize_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
        if value is None or value == "":
            raise InvalidAuditRecordError("timestamp is required")

        text_value = str(value).strip()
        if not text_value:
            raise InvalidAuditRecordError("timestamp is required")
        normalized_text = text_value.replace("Z", "+00:00")
        try:
            parsed_value = datetime.fromisoformat(normalized_text)
        except ValueError as exc:
            raise InvalidAuditRecordError(f"Invalid timestamp value: {value}") from exc
        return parsed_value if parsed_value.tzinfo else parsed_value.replace(tzinfo=timezone.utc)

    def _normalize_required_string(self, value: Any, field_name: str) -> str:
        if value is None:
            raise InvalidAuditRecordError(f"{field_name} is required")
        normalized_value = str(value).strip()
        if not normalized_value:
            raise InvalidAuditRecordError(f"{field_name} is required")
        return normalized_value

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        normalized_value = str(value).strip()
        return normalized_value or None

    @staticmethod
    def _normalize_email(value: Any) -> str | None:
        if value is None or value == "":
            return None
        normalized_value = str(value).strip().lower()
        if not EMAIL_PATTERN.match(normalized_value):
            raise InvalidAuditRecordError(f"Invalid email format: {normalized_value}")
        return normalized_value

    def _table_sql(self) -> str:
        return f"`{self._repository.project}`.`{self._repository.dataset}`.`{AUDIT_TABLE}`"

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def _validate_pagination(self, page: int, page_size: int) -> tuple[int, int, int]:
        if page < 1:
            raise InvalidAuditRecordError("page must be greater than or equal to 1")
        if page_size < 1:
            raise InvalidAuditRecordError("page_size must be greater than or equal to 1")
        return page, page_size, (page - 1) * page_size

    def _log_error(self, event: str, error: BaseException, **context: Any) -> None:
        self.logger.error(
            "Audit repository operation failed",
            extra={
                "event": event,
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                **context,
            },
        )
