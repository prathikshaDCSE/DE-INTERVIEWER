from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

from repository.audit_repository import (
    AuditRecord,
    AuditRecordNotFoundError,
    AuditRepository,
    AuditRepositoryError,
    InvalidAuditRecordError,
)


# ============================================================
# Exceptions
# ============================================================


class AuditServiceError(Exception):
    """Base exception raised by AuditService."""


class AuditValidationError(AuditServiceError):
    """Raised when audit input fails business-layer validation."""


class AuditOperationError(AuditServiceError):
    """Raised when a repository-backed audit operation fails."""


class AuditNotFoundError(AuditServiceError):
    """Raised when a requested audit record cannot be found."""


class AuditExportError(AuditServiceError):
    """Raised when an audit export request cannot be satisfied."""


# ============================================================
# Audit Service
# ============================================================


class AuditService:
    """
    Business layer responsible for audit event orchestration.

    Responsibilities

    • Business-level validation of audit input
    • Composition of higher-level audit workflows from the
      existing ``AuditRepository`` API
    • Repository exception translation
    • Structured logging of every business operation

    This service never communicates with BigQuery directly. It only
    communicates through ``AuditRepository``.

    Business Notes:
        ``AuditRepository`` is treated as the source of truth for
        persistence, schema, and enum validation (severities and
        statuses). This service intentionally does not duplicate or
        re-implement the repository's authoritative enum checks; it
        only performs lightweight, format-level validation before
        delegating, and translates whatever the repository ultimately
        raises. Several public methods described by upstream product
        specifications (bulk logging, recent events, action history,
        entity history, timelines, and exports) have no dedicated
        repository query, so they are composed here from the existing
        repository methods rather than by extending the repository.
    """

    # =======================================================
    # Class Constants (Business Rules)
    # =======================================================

    DEFAULT_PAGE_SIZE = 50

    MAX_PAGE_SIZE = 500

    DEFAULT_RECENT_LIMIT = 10

    MAX_RECENT_LIMIT = 500

    DEFAULT_ENTITY_HISTORY_LIMIT = 200

    MAX_ENTITY_HISTORY_LIMIT = 1000

    DEFAULT_TIMELINE_PAGE_SIZE = 200

    MAX_TIMELINE_PAGE_SIZE = 1000

    DEFAULT_EXPORT_PAGE_SIZE = 1000

    MAX_EXPORT_PAGE_SIZE = 5000

    # Maps caller-facing entity aliases onto the entity type actually
    # persisted by AuditRepository / the specialized repository
    # methods available for it.
    ENTITY_ALIAS_MAP: Mapping[str, str] = {
        "candidate": "candidate",
        "candidate_sessions": "candidate_sessions",
        "session": "candidate_sessions",
        "interview_session": "candidate_sessions",
        "final_reports": "final_reports",
        "report": "final_reports",
        "final_report": "final_reports",
    }

    def __init__(
        self,
        audit_repository: AuditRepository,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the service.

        Args:
            audit_repository: Repository dependency used for all
                audit persistence and retrieval.
            logger: Optional logger instance. A class-scoped logger is
                created when omitted.
        """

        self.audit_repository = audit_repository

        self.logger = logger or logging.getLogger(
            self.__class__.__name__
        )

        self._lock = threading.RLock()

    # =======================================================
    # Part 1: Writing Audit Events
    # =======================================================

    def log_event(self, payload: Mapping[str, Any]) -> AuditRecord:
        """
        Validate and persist a single audit event.

        Args:
            payload: Audit event attributes. Must contain non-empty
                ``action``, ``status``, and ``severity`` values, as
                required by ``AuditRepository``.

        Returns:
            The persisted audit record.

        Raises:
            AuditValidationError: If ``payload`` is malformed or the
                repository rejects the event as invalid.
            AuditOperationError: If persistence otherwise fails.
        """

        self.logger.info("Start log_event")

        with self._lock:

            self._validate_payload(payload)

            try:
                record = self.audit_repository.log_event(payload)
            except (InvalidAuditRecordError, AuditRepositoryError) as exc:
                self._log_failure("log_event_failed", exc)
                raise self._translate_repository_exception(exc) from exc

            self.logger.info(
                "End log_event: audit_id=%s action=%s",
                record.audit_id,
                record.action,
            )

            return record

    def bulk_log_events(
        self,
        payloads: Sequence[Mapping[str, Any]],
        stop_on_error: bool = True,
    ) -> list[AuditRecord]:
        """
        Persist multiple audit events by delegating to ``log_event``.

        Business Notes:
            ``AuditRepository`` does not expose a bulk-insert
            operation, so this method composes bulk behavior by
            calling ``log_event`` once per payload under a single
            lock acquisition, guaranteeing the batch is processed as
            one atomic business operation from the caller's
            perspective.

        Args:
            payloads: Collection of audit event payloads.
            stop_on_error: When ``True`` (default), the first invalid
                or failed event aborts the batch and raises. When
                ``False``, failed events are logged and skipped, and
                only the successfully persisted records are returned.

        Returns:
            The audit records that were successfully persisted.

        Raises:
            AuditValidationError: If ``payloads`` is empty, or if
                ``stop_on_error`` is ``True`` and an event fails
                validation.
            AuditOperationError: If ``stop_on_error`` is ``True`` and
                persistence otherwise fails for an event.
        """

        self.logger.info(
            "Start bulk_log_events: count=%d stop_on_error=%s",
            len(payloads) if payloads else 0,
            stop_on_error,
        )

        if not payloads:
            raise AuditValidationError("payloads must be a non-empty sequence")

        created_records: list[AuditRecord] = []

        with self._lock:

            for index, payload in enumerate(payloads):
                try:
                    created_records.append(self.log_event(payload))
                except AuditServiceError:
                    if stop_on_error:
                        raise
                    self.logger.warning(
                        "bulk_log_events: skipping invalid event at index %d",
                        index,
                    )
                    continue

        self.logger.info(
            "End bulk_log_events: requested=%d persisted=%d",
            len(payloads),
            len(created_records),
        )

        return created_records

    # =======================================================
    # Part 2: Retrieving Individual Events
    # =======================================================

    def get_event(self, audit_id: str) -> AuditRecord:
        """
        Retrieve a single audit event by identifier.

        Args:
            audit_id: Unique audit identifier.

        Returns:
            The matching audit record.

        Raises:
            AuditValidationError: If ``audit_id`` is blank.
            AuditNotFoundError: If no matching audit record exists.
            AuditOperationError: If retrieval otherwise fails.
        """

        audit_id_clean = self._validate_non_empty_string(audit_id, "audit_id")

        self.logger.info("Start get_event: audit_id=%s", audit_id_clean)

        with self._lock:

            try:
                record = self.audit_repository.get_audit(audit_id_clean)
            except (AuditRecordNotFoundError, AuditRepositoryError) as exc:
                self._log_failure("get_event_failed", exc, audit_id=audit_id_clean)
                raise self._translate_repository_exception(exc) from exc

        self.logger.info("End get_event: audit_id=%s", audit_id_clean)

        return record

    def get_events(self, audit_ids: Sequence[str]) -> list[AuditRecord]:
        """
        Retrieve multiple audit events by identifier.

        Args:
            audit_ids: Collection of audit identifiers.

        Returns:
            Matching audit records.

        Raises:
            AuditValidationError: If ``audit_ids`` is empty.
            AuditOperationError: If retrieval fails.
        """

        if not audit_ids:
            raise AuditValidationError("audit_ids must be a non-empty sequence")

        normalized_ids = [
            self._validate_non_empty_string(audit_id, "audit_id")
            for audit_id in audit_ids
        ]

        self.logger.info("Start get_events: count=%d", len(normalized_ids))

        with self._lock:

            try:
                records = self.audit_repository.get_audits(normalized_ids)
            except AuditRepositoryError as exc:
                self._log_failure("get_events_failed", exc)
                raise self._translate_repository_exception(exc) from exc

        self.logger.info(
            "End get_events: requested=%d found=%d",
            len(normalized_ids),
            len(records),
        )

        return records

    def event_exists(self, audit_id: str) -> bool:
        """
        Check whether an audit event exists.

        Args:
            audit_id: Unique audit identifier.

        Returns:
            ``True`` if the audit record exists, otherwise ``False``.

        Raises:
            AuditValidationError: If ``audit_id`` is blank.
            AuditOperationError: If the existence check fails.
        """

        audit_id_clean = self._validate_non_empty_string(audit_id, "audit_id")

        with self._lock:

            try:
                exists = self.audit_repository.audit_exists(audit_id_clean)
            except AuditRepositoryError as exc:
                self._log_failure("event_exists_failed", exc, audit_id=audit_id_clean)
                raise self._translate_repository_exception(exc) from exc

        return exists

    # =======================================================
    # Part 3: Search and Filtered History
    # =======================================================

    def search_events(
        self,
        search_term: str,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """
        Search audit events by free text.

        Args:
            search_term: Free-text search input.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and matching audit records, as
            returned by ``AuditRepository.search_audits``.

        Raises:
            AuditValidationError: If ``search_term`` is blank or
                pagination arguments are invalid.
            AuditOperationError: If the search otherwise fails.
        """

        search_term_clean = self._validate_non_empty_string(
            search_term,
            "search_term",
        )
        normalized_page, normalized_page_size = self._validate_pagination(
            page,
            page_size,
        )

        self.logger.info(
            "Start search_events: term=%s page=%d page_size=%d",
            search_term_clean,
            normalized_page,
            normalized_page_size,
        )

        with self._lock:

            try:
                result = self.audit_repository.search_audits(
                    search_term_clean,
                    page=normalized_page,
                    page_size=normalized_page_size,
                )
            except (InvalidAuditRecordError, AuditRepositoryError) as exc:
                self._log_failure("search_events_failed", exc, term=search_term_clean)
                raise self._translate_repository_exception(exc) from exc

        self.logger.info(
            "End search_events: term=%s total=%d",
            search_term_clean,
            result.get("total", 0),
        )

        return result

    def get_user_history(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """
        Retrieve paginated audit history for a user.

        Args:
            user_id: User identifier.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and matching audit records.

        Raises:
            AuditValidationError: If ``user_id`` is blank or
                pagination arguments are invalid.
            AuditOperationError: If retrieval otherwise fails.
        """

        user_id_clean = self._validate_non_empty_string(user_id, "user_id")
        normalized_page, normalized_page_size = self._validate_pagination(
            page,
            page_size,
        )

        self.logger.info("Start get_user_history: user_id=%s", user_id_clean)

        with self._lock:

            try:
                result = self.audit_repository.get_user_audits(
                    user_id=user_id_clean,
                    page=normalized_page,
                    page_size=normalized_page_size,
                )
            except AuditRepositoryError as exc:
                self._log_failure(
                    "get_user_history_failed",
                    exc,
                    user_id=user_id_clean,
                )
                raise self._translate_repository_exception(exc) from exc

        self.logger.info(
            "End get_user_history: user_id=%s total=%d",
            user_id_clean,
            result.get("total", 0),
        )

        return result

    def get_action_history(
        self,
        action: str,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """
        Retrieve audit events matching a specific action.

        Business Notes:
            ``AuditRepository`` has no dedicated action-only filter.
            This method composes the result by delegating to
            ``search_audits`` (a free-text search across ``action``,
            ``description``, ``entity``, and ``entity_id``) and then
            narrowing the returned page to records whose ``action``
            matches exactly (case-insensitively). Because the
            narrowing happens after pagination at the repository
            layer, ``total`` reflects the underlying free-text search
            hit count rather than the exact-action match count; the
            precise number of exact matches on this page is reported
            separately as ``matched_count``.

        Args:
            action: Exact action name to match.
            page: Page number starting at 1, applied to the
                underlying search before exact-action narrowing.
            page_size: Number of records per page requested from the
                underlying search.

        Returns:
            A dictionary with ``items`` (exact-action matches on this
            page), ``matched_count``, ``page``, ``page_size``, and
            ``total`` (the underlying search's total hit count).

        Raises:
            AuditValidationError: If ``action`` is blank or pagination
                arguments are invalid.
            AuditOperationError: If the underlying search fails.
        """

        action_clean = self._validate_non_empty_string(action, "action")
        normalized_page, normalized_page_size = self._validate_pagination(
            page,
            page_size,
        )

        self.logger.info("Start get_action_history: action=%s", action_clean)

        with self._lock:

            try:
                search_result = self.audit_repository.search_audits(
                    action_clean,
                    page=normalized_page,
                    page_size=normalized_page_size,
                )
            except (InvalidAuditRecordError, AuditRepositoryError) as exc:
                self._log_failure(
                    "get_action_history_failed",
                    exc,
                    action=action_clean,
                )
                raise self._translate_repository_exception(exc) from exc

            matched_items = [
                record
                for record in search_result.get("items", [])
                if record.action.strip().upper() == action_clean.upper()
            ]

        result = {
            "items": matched_items,
            "matched_count": len(matched_items),
            "page": normalized_page,
            "page_size": normalized_page_size,
            "total": search_result.get("total", 0),
        }

        self.logger.info(
            "End get_action_history: action=%s matched_count=%d",
            action_clean,
            len(matched_items),
        )

        return result

    def get_entity_history(
        self,
        entity: str,
        entity_id: str,
        limit: int = DEFAULT_ENTITY_HISTORY_LIMIT,
    ) -> list[AuditRecord]:
        """
        Retrieve audit history for a specific entity instance.

        Business Notes:
            Dispatches to the most specific repository method
            available for the resolved entity type: candidate
            histories delegate to ``get_candidate_audits``, interview
            session histories to ``get_session_audits``, final report
            histories to ``get_report_audits``. Entity types with no
            dedicated repository method fall back to the generic
            ``list_audits(entity=..., entity_id=...)`` filter.

        Args:
            entity: Entity type (for example ``"candidate"``,
                ``"session"``, or ``"report"``). Aliases are resolved
                via ``ENTITY_ALIAS_MAP``.
            entity_id: Entity instance identifier.
            limit: Maximum number of records to return for entity
                types resolved through a paginated repository method.

        Returns:
            Matching audit records, most recent first.

        Raises:
            AuditValidationError: If ``entity`` or ``entity_id`` is
                blank, or ``limit`` is out of range.
            AuditOperationError: If retrieval otherwise fails.
        """

        entity_clean = self._validate_non_empty_string(entity, "entity")
        entity_id_clean = self._validate_non_empty_string(entity_id, "entity_id")
        normalized_limit = self._validate_bounded_limit(
            limit,
            self.MAX_ENTITY_HISTORY_LIMIT,
            "limit",
        )
        resolved_entity = self.ENTITY_ALIAS_MAP.get(
            entity_clean.lower(),
            entity_clean.lower(),
        )

        self.logger.info(
            "Start get_entity_history: entity=%s entity_id=%s",
            resolved_entity,
            entity_id_clean,
        )

        with self._lock:

            try:
                records = self._dispatch_entity_history(
                    resolved_entity=resolved_entity,
                    entity_id=entity_id_clean,
                    limit=normalized_limit,
                )
            except AuditRepositoryError as exc:
                self._log_failure(
                    "get_entity_history_failed",
                    exc,
                    entity=resolved_entity,
                    entity_id=entity_id_clean,
                )
                raise self._translate_repository_exception(exc) from exc

        self.logger.info(
            "End get_entity_history: entity=%s entity_id=%s count=%d",
            resolved_entity,
            entity_id_clean,
            len(records),
        )

        return records

    def _dispatch_entity_history(
        self,
        resolved_entity: str,
        entity_id: str,
        limit: int,
    ) -> list[AuditRecord]:
        """
        Route entity-history retrieval to the appropriate repository
        method.

        Caller must already hold ``self._lock``.

        Args:
            resolved_entity: Canonical entity type after alias
                resolution.
            entity_id: Entity instance identifier.
            limit: Maximum number of records to return for paginated
                lookups.

        Returns:
            Matching audit records.
        """

        if resolved_entity == "candidate":
            paginated = self.audit_repository.get_candidate_audits(
                candidate_id=entity_id,
                page=1,
                page_size=limit,
            )
            return list(paginated.get("items", []))

        if resolved_entity == "candidate_sessions":
            return self.audit_repository.get_session_audits(entity_id)

        if resolved_entity == "final_reports":
            return self.audit_repository.get_report_audits(entity_id)

        fallback = self.audit_repository.list_audits(
            entity=resolved_entity,
            entity_id=entity_id,
            page=1,
            page_size=limit,
        )
        return list(fallback.get("items", []))

    def get_recent_events(
        self,
        limit: int = DEFAULT_RECENT_LIMIT,
    ) -> list[AuditRecord]:
        """
        Retrieve the most recently recorded audit events.

        Business Notes:
            Composed from ``list_audits(page=1, page_size=limit)``,
            since ``AuditRepository`` has no dedicated "latest N"
            query. Repository ordering (most recent first) is
            preserved as-is.

        Args:
            limit: Maximum number of events to return.

        Returns:
            The most recent audit records, most recent first.

        Raises:
            AuditValidationError: If ``limit`` is out of range.
            AuditOperationError: If retrieval otherwise fails.
        """

        normalized_limit = self._validate_bounded_limit(
            limit,
            self.MAX_RECENT_LIMIT,
            "limit",
        )

        self.logger.info("Start get_recent_events: limit=%d", normalized_limit)

        with self._lock:

            try:
                result = self.audit_repository.list_audits(
                    page=1,
                    page_size=normalized_limit,
                )
            except AuditRepositoryError as exc:
                self._log_failure("get_recent_events_failed", exc)
                raise self._translate_repository_exception(exc) from exc

        records = list(result.get("items", []))

        self.logger.info(
            "End get_recent_events: limit=%d returned=%d",
            normalized_limit,
            len(records),
        )

        return records

    def get_events_page(
        self,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
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
        Retrieve a filtered, paginated page of audit events.

        Args:
            page: Page number starting at 1.
            page_size: Number of records per page.
            created_from: Optional inclusive timestamp lower bound.
            created_to: Optional inclusive timestamp upper bound.
            severity: Optional severity filter.
            category: Optional entity-type filter alias.
            status: Optional status filter.
            user_id: Optional user identifier filter.
            candidate_id: Optional candidate entity identifier filter.
            entity: Optional entity type filter.
            entity_id: Optional entity identifier filter.

        Returns:
            Pagination metadata and matching audit records.

        Raises:
            AuditValidationError: If pagination arguments or the date
                range are invalid.
            AuditOperationError: If retrieval otherwise fails.
        """

        normalized_page, normalized_page_size = self._validate_pagination(
            page,
            page_size,
        )
        self._validate_date_range(created_from, created_to)

        self.logger.info(
            "Start get_events_page: page=%d page_size=%d",
            normalized_page,
            normalized_page_size,
        )

        with self._lock:

            try:
                result = self.audit_repository.list_audits(
                    page=normalized_page,
                    page_size=normalized_page_size,
                    created_from=created_from,
                    created_to=created_to,
                    severity=severity,
                    category=category,
                    status=status,
                    user_id=user_id,
                    candidate_id=candidate_id,
                    entity=entity,
                    entity_id=entity_id,
                )
            except (InvalidAuditRecordError, AuditRepositoryError) as exc:
                self._log_failure("get_events_page_failed", exc)
                raise self._translate_repository_exception(exc) from exc

        self.logger.info(
            "End get_events_page: total=%d",
            result.get("total", 0),
        )

        return result

    # =======================================================
    # Part 4: Counting and Statistics
    # =======================================================

    def count_events(
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
        Count audit events matching the given filters.

        Args:
            created_from: Optional inclusive timestamp lower bound.
            created_to: Optional inclusive timestamp upper bound.
            severity: Optional severity filter.
            category: Optional entity-type filter.
            status: Optional status filter.
            user_id: Optional user identifier filter.
            candidate_id: Optional candidate entity identifier filter.
            entity_id: Optional explicit entity identifier filter.

        Returns:
            The number of matching audit records.

        Raises:
            AuditValidationError: If the date range is invalid.
            AuditOperationError: If counting otherwise fails.
        """

        self._validate_date_range(created_from, created_to)

        with self._lock:

            try:
                total = self.audit_repository.count_audits(
                    created_from=created_from,
                    created_to=created_to,
                    severity=severity,
                    category=category,
                    status=status,
                    user_id=user_id,
                    candidate_id=candidate_id,
                    entity_id=entity_id,
                )
            except (InvalidAuditRecordError, AuditRepositoryError) as exc:
                self._log_failure("count_events_failed", exc)
                raise self._translate_repository_exception(exc) from exc

        return total

    def get_statistics(self) -> dict[str, Any]:
        """
        Retrieve aggregated audit statistics.

        Returns:
            Summary counts and distributions for audit events, as
            returned by ``AuditRepository.get_statistics``.

        Raises:
            AuditOperationError: If statistics computation fails.
        """

        self.logger.info("Start get_statistics")

        with self._lock:

            try:
                statistics_payload = self.audit_repository.get_statistics()
            except AuditRepositoryError as exc:
                self._log_failure("get_statistics_failed", exc)
                raise self._translate_repository_exception(exc) from exc

        self.logger.info(
            "End get_statistics: total_events=%d",
            statistics_payload.get("total_events", 0),
        )

        return statistics_payload

    # =======================================================
    # Part 5: Timeline Composition
    # =======================================================

    def generate_timeline(
        self,
        user_id: str | None = None,
        entity: str | None = None,
        entity_id: str | None = None,
        created_from: date | datetime | str | None = None,
        created_to: date | datetime | str | None = None,
        severity: str | None = None,
        status: str | None = None,
        page_size: int = DEFAULT_TIMELINE_PAGE_SIZE,
    ) -> list[dict[str, Any]]:
        """
        Compose a chronological timeline of audit events.

        Business Notes:
            ``AuditRepository`` has no bucketed or streaming timeline
            query. This method fetches a single filtered page via
            ``list_audits`` (which orders results most-recent-first)
            and re-orders it chronologically (oldest first) to build
            the timeline. The timeline is therefore bounded by
            ``page_size``; callers needing a longer horizon should use
            ``get_events_page`` directly with repository-level
            pagination.

        Args:
            user_id: Optional user identifier filter.
            entity: Optional entity type filter.
            entity_id: Optional entity identifier filter.
            created_from: Optional inclusive timestamp lower bound.
            created_to: Optional inclusive timestamp upper bound.
            severity: Optional severity filter.
            status: Optional status filter.
            page_size: Maximum number of events to include, capped at
                ``MAX_TIMELINE_PAGE_SIZE``.

        Returns:
            Timeline entries ordered oldest to newest, each containing
            ``timestamp``, ``action``, ``description``, ``user_id``,
            ``entity``, ``entity_id``, ``status``, and ``severity``.

        Raises:
            AuditValidationError: If the date range or ``page_size``
                is invalid.
            AuditOperationError: If retrieval otherwise fails.
        """

        self._validate_date_range(created_from, created_to)
        normalized_page_size = self._validate_bounded_limit(
            page_size,
            self.MAX_TIMELINE_PAGE_SIZE,
            "page_size",
        )

        self.logger.info(
            "Start generate_timeline: page_size=%d",
            normalized_page_size,
        )

        with self._lock:

            try:
                result = self.audit_repository.list_audits(
                    page=1,
                    page_size=normalized_page_size,
                    created_from=created_from,
                    created_to=created_to,
                    severity=severity,
                    status=status,
                    user_id=user_id,
                    entity=entity,
                    entity_id=entity_id,
                )
            except (InvalidAuditRecordError, AuditRepositoryError) as exc:
                self._log_failure("generate_timeline_failed", exc)
                raise self._translate_repository_exception(exc) from exc

            chronological_records = list(reversed(result.get("items", [])))
            timeline = [
                self._to_timeline_entry(record)
                for record in chronological_records
            ]

        self.logger.info(
            "End generate_timeline: entries=%d",
            len(timeline),
        )

        return timeline

    @staticmethod
    def _to_timeline_entry(record: AuditRecord) -> dict[str, Any]:
        """
        Convert an audit record into a timeline entry.

        Args:
            record: Source audit record.

        Returns:
            A dictionary describing the event for timeline display.
        """

        return {
            "timestamp": record.timestamp.isoformat(),
            "action": record.action,
            "description": record.description,
            "user_id": record.user_id,
            "entity": record.entity,
            "entity_id": record.entity_id,
            "status": record.status,
            "severity": record.severity,
        }

    # =======================================================
    # Part 6: Export
    # =======================================================

    def export_events(
        self,
        created_from: date | datetime | str | None = None,
        created_to: date | datetime | str | None = None,
        severity: str | None = None,
        category: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        candidate_id: str | None = None,
        entity: str | None = None,
        entity_id: str | None = None,
        page_size: int = DEFAULT_EXPORT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """
        Build a structured, export-ready audit payload.

        Business Notes:
            This method returns an in-memory structured payload only;
            it never writes CSV, files, or any other artifact. Export
            volume is bounded by ``page_size`` (capped at
            ``MAX_EXPORT_PAGE_SIZE``) because ``AuditRepository``
            exposes only offset-based pagination and no dedicated bulk
            export cursor. Callers needing the full result set beyond
            the cap should page through ``get_events_page`` and
            concatenate the pages themselves.

        Args:
            created_from: Optional inclusive timestamp lower bound.
            created_to: Optional inclusive timestamp upper bound.
            severity: Optional severity filter.
            category: Optional entity-type filter.
            status: Optional status filter.
            user_id: Optional user identifier filter.
            candidate_id: Optional candidate entity identifier filter.
            entity: Optional entity type filter.
            entity_id: Optional entity identifier filter.
            page_size: Maximum number of records to include, capped at
                ``MAX_EXPORT_PAGE_SIZE``.

        Returns:
            A dictionary with ``generated_at``, ``filters``,
            ``total_matching``, ``exported_count``, and ``events``
            (each event serialized via ``AuditRecord.to_dict``).

        Raises:
            AuditValidationError: If the date range is invalid.
            AuditExportError: If ``page_size`` exceeds the allowed
                maximum.
            AuditOperationError: If retrieval otherwise fails.
        """

        self._validate_date_range(created_from, created_to)

        if page_size > self.MAX_EXPORT_PAGE_SIZE:
            raise AuditExportError(
                f"page_size must not exceed {self.MAX_EXPORT_PAGE_SIZE} "
                f"for a single export; got {page_size}"
            )

        normalized_page_size = self._validate_bounded_limit(
            page_size,
            self.MAX_EXPORT_PAGE_SIZE,
            "page_size",
        )

        applied_filters = {
            "created_from": self._filter_to_str(created_from),
            "created_to": self._filter_to_str(created_to),
            "severity": severity,
            "category": category,
            "status": status,
            "user_id": user_id,
            "candidate_id": candidate_id,
            "entity": entity,
            "entity_id": entity_id,
        }

        self.logger.info("Start export_events: page_size=%d", normalized_page_size)

        with self._lock:

            try:
                result = self.audit_repository.list_audits(
                    page=1,
                    page_size=normalized_page_size,
                    created_from=created_from,
                    created_to=created_to,
                    severity=severity,
                    category=category,
                    status=status,
                    user_id=user_id,
                    candidate_id=candidate_id,
                    entity=entity,
                    entity_id=entity_id,
                )
            except (InvalidAuditRecordError, AuditRepositoryError) as exc:
                self._log_failure("export_events_failed", exc)
                raise self._translate_repository_exception(exc) from exc

            items = result.get("items", [])
            export_payload = {
                "generated_at": self._utcnow().isoformat(),
                "filters": applied_filters,
                "total_matching": result.get("total", 0),
                "exported_count": len(items),
                "events": [record.to_dict() for record in items],
            }

        self.logger.info(
            "End export_events: exported_count=%d total_matching=%d",
            export_payload["exported_count"],
            export_payload["total_matching"],
        )

        return export_payload

    @staticmethod
    def _filter_to_str(value: date | datetime | str | None) -> str | None:
        """
        Render an optional date filter value for inclusion in an
        export's filter metadata.

        Args:
            value: The raw filter value supplied by the caller.

        Returns:
            An ISO-formatted string, or ``None`` when no value was
            supplied.
        """

        if value is None:
            return None
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)

    # =======================================================
    # Part 7: Retention (Unsupported by Persistence Layer)
    # =======================================================

    def delete_old_events(self, retention_days: int) -> None:
        """
        Delete audit events older than a retention window.

        Business Notes:
            ``AuditRepository`` exposes no delete, retention, or
            time-to-live capability; it is an append-only audit log.
            This method exists to satisfy the service's business
            interface but always raises, so retention policy
            enforcement is not silently skipped or approximated.

        Args:
            retention_days: Requested retention window in days.

        Raises:
            AuditOperationError: Always. Retention-based deletion is
                not supported by the current persistence layer.
        """

        self.logger.warning(
            "delete_old_events requested but unsupported: retention_days=%s",
            retention_days,
        )

        raise AuditOperationError(
            "delete_old_events is not supported: AuditRepository provides "
            "append-only persistence with no deletion or retention-policy "
            "capability."
        )

    # =======================================================
    # Repository Exception Translation
    # =======================================================

    def _translate_repository_exception(
        self,
        exc: Exception,
    ) -> AuditServiceError:
        """
        Translate a repository-layer exception into a service-layer
        exception.

        Repository exceptions (``AuditRecordNotFoundError``,
        ``InvalidAuditRecordError``, ``AuditRepositoryError``) must
        never leak past this service.

        Args:
            exc: The exception raised by the repository.

        Returns:
            The corresponding ``AuditServiceError`` subclass, ready to
            be raised by the caller.
        """

        if isinstance(exc, AuditRecordNotFoundError):
            return AuditNotFoundError(str(exc))

        if isinstance(exc, InvalidAuditRecordError):
            return AuditValidationError(str(exc))

        if isinstance(exc, AuditRepositoryError):
            return AuditOperationError(str(exc))

        return AuditServiceError(str(exc))

    def _log_failure(
        self,
        operation: str,
        error: BaseException,
        **context: Any,
    ) -> None:
        """
        Log a repository-call failure with structured context.

        Args:
            operation: Short machine-readable operation identifier.
            error: The exception raised by the repository.
            **context: Additional structured context to attach to the
                log record.
        """

        self.logger.exception(
            "Audit service operation failed",
            extra={
                "event": operation,
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                **context,
            },
        )

    # =======================================================
    # Validation Helpers
    # =======================================================

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        """
        Validate that a log_event payload contains the fields
        required by ``AuditRepository`` before delegating.

        Args:
            payload: Audit event payload.

        Raises:
            AuditValidationError: If ``payload`` is not a mapping, or
                a required field is missing or blank.
        """

        if not isinstance(payload, Mapping):
            raise AuditValidationError(
                "payload must be a mapping of audit attributes"
            )

        required_fields = getattr(
            self.audit_repository,
            "REQUIRED_FIELDS",
            frozenset({"action", "status", "severity"}),
        )
        missing_fields = [
            field_name
            for field_name in sorted(required_fields)
            if payload.get(field_name) is None
            or str(payload.get(field_name)).strip() == ""
        ]
        if missing_fields:
            raise AuditValidationError(
                f"Missing required audit fields: {', '.join(missing_fields)}"
            )

    def _validate_non_empty_string(self, value: Any, field_name: str) -> str:
        """
        Validate that a value is a non-blank string.

        Args:
            value: Candidate value.
            field_name: Field name used in the error message.

        Returns:
            The stripped string value.

        Raises:
            AuditValidationError: If ``value`` is ``None`` or blank
                after stripping.
        """

        if value is None:
            raise AuditValidationError(f"{field_name} must be a non-empty string")

        normalized_value = str(value).strip()
        if not normalized_value:
            raise AuditValidationError(f"{field_name} must be a non-empty string")

        return normalized_value

    def _validate_pagination(
        self,
        page: int,
        page_size: int,
    ) -> tuple[int, int]:
        """
        Validate pagination arguments.

        Args:
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            The validated ``(page, page_size)`` tuple.

        Raises:
            AuditValidationError: If ``page`` or ``page_size`` is not
                a positive integer, or ``page_size`` exceeds
                ``MAX_PAGE_SIZE``.
        """

        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise AuditValidationError(
                f"page must be an integer >= 1, got: {page!r}"
            )

        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or page_size < 1
        ):
            raise AuditValidationError(
                f"page_size must be an integer >= 1, got: {page_size!r}"
            )

        if page_size > self.MAX_PAGE_SIZE:
            raise AuditValidationError(
                f"page_size must not exceed {self.MAX_PAGE_SIZE}, "
                f"got: {page_size}"
            )

        return page, page_size

    def _validate_bounded_limit(
        self,
        value: int,
        max_value: int,
        field_name: str,
    ) -> int:
        """
        Validate that an integer limit falls within ``[1, max_value]``.

        Args:
            value: Candidate limit value.
            max_value: Inclusive upper bound.
            field_name: Field name used in the error message.

        Returns:
            The validated limit.

        Raises:
            AuditValidationError: If ``value`` is not an integer in
                range.
        """

        if isinstance(value, bool) or not isinstance(value, int):
            raise AuditValidationError(
                f"{field_name} must be an integer, got: {type(value).__name__}"
            )

        if value < 1 or value > max_value:
            raise AuditValidationError(
                f"{field_name} must be between 1 and {max_value}, got: {value}"
            )

        return value

    def _validate_date_range(
        self,
        created_from: date | datetime | str | None,
        created_to: date | datetime | str | None,
    ) -> None:
        """
        Validate that a date range is internally consistent.

        Business Notes:
            Full parsing and normalization of string-typed date
            filters is the repository's responsibility (it is the
            authoritative source for timestamp semantics). This
            helper only performs an early, defensive check when both
            bounds are already directly comparable ``date`` or
            ``datetime`` instances, to fail fast with a clear service-
            layer error before a query is issued.

        Args:
            created_from: Optional inclusive lower bound.
            created_to: Optional inclusive upper bound.

        Raises:
            AuditValidationError: If both bounds are directly
                comparable and ``created_from`` is after
                ``created_to``.
        """

        if created_from is None or created_to is None:
            return

        both_are_dates = isinstance(created_from, (date, datetime)) and isinstance(
            created_to, (date, datetime)
        )
        if both_are_dates and created_from > created_to:
            raise AuditValidationError(
                "created_from must be less than or equal to created_to"
            )

    @staticmethod
    def _utcnow() -> datetime:
        """
        Return the current UTC timestamp.

        Returns:
            A timezone-aware ``datetime`` in UTC.
        """

        return datetime.now(timezone.utc)