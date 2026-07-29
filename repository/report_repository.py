from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from repository.bigquery_repository import (
    BigQueryRepository,
    BigQueryRepositoryError,
    InsertError,
    QueryExecutionError,
    UpdateError,
    ValidationError as BigQueryValidationError,
)

EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
REPORT_TABLE = "interview_reports"


class ReportRepositoryError(Exception):
    """Raised when the report repository encounters a general failure."""


class ReportAlreadyExistsError(ReportRepositoryError):
    """Raised when a report already exists for a unique attribute."""


class ReportNotFoundError(ReportRepositoryError):
    """Raised when a requested report cannot be found."""


class InvalidReportError(ReportRepositoryError):
    """Raised when report input fails validation."""


@dataclass(frozen=True)
class ReportRecord:
    """Immutable representation of a persisted interview report."""

    report_id: str
    session_id: str
    candidate_id: str
    candidate_name: str
    candidate_email: str
    interviewer: str
    interview_stage: str
    overall_score: float
    overall_rating: str | None
    recommendation: str
    strengths: str | None
    weaknesses: str | None
    feedback: str | None
    question_count: int
    questions_answered: int
    average_response_time: float | None
    interview_duration: float
    report_status: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation of the record."""
        payload = asdict(self)
        for key in ("created_at", "updated_at"):
            payload[key] = payload[key].isoformat()
        return payload


class ReportRepository:
    """Repository responsible for interview report persistence in BigQuery."""

    DEFAULT_REPORT_STATUSES = frozenset(
        {"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "DELETED"}
    )
    CREATE_REQUIRED_FIELDS = frozenset(
        {
            "report_id",
            "session_id",
            "candidate_id",
            "candidate_name",
            "candidate_email",
            "interviewer",
            "interview_stage",
            "overall_score",
            "recommendation",
            "question_count",
            "questions_answered",
            "interview_duration",
            "created_by",
        }
    )
    UPDATE_ALLOWED_FIELDS = frozenset(
        {
            "candidate_name",
            "candidate_email",
            "interviewer",
            "interview_stage",
            "overall_score",
            "overall_rating",
            "recommendation",
            "strengths",
            "weaknesses",
            "feedback",
            "question_count",
            "questions_answered",
            "average_response_time",
            "interview_duration",
            "report_status",
        }
    )

    def __init__(
        self,
        bigquery_repository: BigQueryRepository,
        logger: logging.Logger | None = None,
        config_path: str | Path | None = None,
        valid_report_statuses: Iterable[str] | None = None,
    ) -> None:
        """
        Initialize the repository.

        Args:
            bigquery_repository: Shared BigQuery repository dependency.
            logger: Optional logger instance.
            config_path: Optional path to the application config directory.
            valid_report_statuses: Optional supported report statuses.
        """
        self._repository = bigquery_repository
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()

        project_root = Path(__file__).resolve().parent.parent
        self._config_path = (
            Path(config_path) if config_path is not None else project_root / "config"
        )
        self._valid_report_statuses = {
            self._normalize_required_string(status, "report_status").upper()
            for status in (valid_report_statuses or self.DEFAULT_REPORT_STATUSES)
        }
        self._valid_stages = self._load_valid_stages()
        self._valid_recommendations = self._load_valid_recommendations()

    def create_report(self, payload: Mapping[str, Any]) -> ReportRecord:
        """
        Create a new interview report.

        Args:
            payload: Report payload containing required report fields.

        Returns:
            The created report record.
        """
        if not isinstance(payload, Mapping):
            raise InvalidReportError("payload must be a mapping of report attributes")

        with self._lock:
            prepared_payload = self._prepare_create_payload(payload)
            if self.report_exists(report_id=prepared_payload["report_id"]):
                raise ReportAlreadyExistsError(
                    f"Report ID already exists: {prepared_payload['report_id']}"
                )
            if self.report_exists(session_id=prepared_payload["session_id"]):
                raise ReportAlreadyExistsError(
                    f"Session already has a report: {prepared_payload['session_id']}"
                )

            try:
                self._repository.insert(REPORT_TABLE, prepared_payload)
            except (InsertError, BigQueryValidationError, BigQueryRepositoryError) as exc:
                self._log_error(
                    "create_report_failed",
                    exc,
                    report_id=prepared_payload["report_id"],
                )
                raise ReportRepositoryError("Failed to create report") from exc

            record = self.get_report(prepared_payload["report_id"])
            self.logger.info(
                "Interview report created",
                extra={
                    "event": "report_created",
                    "report_id": record.report_id,
                    "candidate_id": record.candidate_id,
                    "session_id": record.session_id,
                },
            )
            return record

    def get_report(self, report_id: str) -> ReportRecord:
        """
        Retrieve an interview report by identifier.

        Args:
            report_id: Unique report identifier.

        Returns:
            The matching report record.
        """
        report_id_clean = self._normalize_required_string(report_id, "report_id")
        sql = (
            "SELECT report_id, session_id, candidate_id, candidate_name, candidate_email, "
            "interviewer, interview_stage, overall_score, overall_rating, recommendation, "
            "strengths, weaknesses, feedback, question_count, questions_answered, "
            "average_response_time, interview_duration, report_status, created_at, "
            "updated_at, created_by, updated_by "
            f"FROM {self._table_sql()} "
            "WHERE report_id = @report_id "
            "LIMIT 1"
        )
        try:
            row = self._repository.fetch_one(sql, parameters={"report_id": report_id_clean})
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("get_report_failed", exc, report_id=report_id_clean)
            raise ReportRepositoryError("Failed to fetch report") from exc

        if row is None:
            raise ReportNotFoundError(f"Report not found: {report_id_clean}")
        self.logger.info(
            "Interview report retrieved",
            extra={"event": "report_retrieved", "report_id": report_id_clean},
        )
        return self._row_to_record(row)

    def update_report(
        self,
        report_id: str,
        updates: Mapping[str, Any],
        updated_by: str,
    ) -> ReportRecord:
        """
        Update allowed columns for an interview report.

        Args:
            report_id: Unique report identifier.
            updates: Partial update payload.
            updated_by: Actor performing the update.

        Returns:
            The updated report record.
        """
        if not isinstance(updates, Mapping) or not updates:
            raise InvalidReportError("updates must be a non-empty mapping")

        with self._lock:
            current = self.get_report(report_id)
            sanitized_updates = self._prepare_update_payload(
                current=current,
                updates=updates,
                updated_by=updated_by,
            )
            assignments, parameters = self._build_update_statement(
                report_id=current.report_id,
                updates=sanitized_updates,
            )
            sql = (
                f"UPDATE {self._table_sql()} "
                f"SET {assignments} "
                "WHERE report_id = @report_id"
            )
            try:
                affected_rows = self._repository.update(sql, parameters=parameters)
            except (UpdateError, BigQueryValidationError, BigQueryRepositoryError) as exc:
                self._log_error("update_report_failed", exc, report_id=current.report_id)
                raise ReportRepositoryError("Failed to update report") from exc

            if affected_rows == 0:
                raise ReportNotFoundError(f"Report not found: {current.report_id}")

            record = self.get_report(current.report_id)
            self.logger.info(
                "Interview report updated",
                extra={
                    "event": "report_updated",
                    "report_id": record.report_id,
                    "updated_fields": sorted(sanitized_updates.keys()),
                },
            )
            return record

    def delete_report(self, report_id: str, updated_by: str) -> ReportRecord:
        """
        Soft delete a report by marking its status as DELETED.

        Args:
            report_id: Unique report identifier.
            updated_by: Actor performing the operation.

        Returns:
            The updated deleted report record.
        """
        record = self.update_report(
            report_id=report_id,
            updates={"report_status": "DELETED"},
            updated_by=updated_by,
        )
        self.logger.info(
            "Interview report soft deleted",
            extra={"event": "report_deleted", "report_id": record.report_id},
        )
        return record

    def list_reports(
        self,
        status: str | None = None,
        candidate: str | None = None,
        recommendation: str | None = None,
        stage: str | None = None,
        created_from: date | datetime | str | None = None,
        created_to: date | datetime | str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        List reports using filters and offset-based pagination.

        Args:
            status: Optional report status filter.
            candidate: Optional candidate identifier, name, or email filter.
            recommendation: Optional recommendation filter.
            stage: Optional interview stage filter.
            created_from: Optional inclusive creation date lower bound.
            created_to: Optional inclusive creation date upper bound.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and report items.
        """
        normalized_page, normalized_page_size, offset = self._validate_pagination(
            page=page,
            page_size=page_size,
        )
        where_clause, parameters = self._build_filter_clause(
            status=status,
            candidate=candidate,
            recommendation=recommendation,
            stage=stage,
            created_from=created_from,
            created_to=created_to,
        )
        total = self.count_reports(
            status=status,
            candidate=candidate,
            recommendation=recommendation,
            stage=stage,
            created_from=created_from,
            created_to=created_to,
        )
        sql = (
            "SELECT report_id, session_id, candidate_id, candidate_name, candidate_email, "
            "interviewer, interview_stage, overall_score, overall_rating, recommendation, "
            "strengths, weaknesses, feedback, question_count, questions_answered, "
            "average_response_time, interview_duration, report_status, created_at, "
            "updated_at, created_by, updated_by "
            f"FROM {self._table_sql()} "
            f"{where_clause} "
            "ORDER BY created_at DESC, report_id ASC "
            "LIMIT @limit OFFSET @offset"
        )
        parameters.update({"limit": normalized_page_size, "offset": offset})

        try:
            rows = self._repository.fetch_all(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("list_reports_failed", exc)
            raise ReportRepositoryError("Failed to list reports") from exc

        items = [self._row_to_record(row) for row in rows]
        return {
            "items": items,
            "page": normalized_page,
            "page_size": normalized_page_size,
            "total": total,
            "total_pages": ceil(total / normalized_page_size) if total else 0,
            "has_next": offset + normalized_page_size < total,
            "has_previous": normalized_page > 1,
        }

    def search_reports(
        self,
        search_term: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        Search reports by candidate name, email, report id, or session id.

        Args:
            search_term: Free-text search input.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and matching reports.
        """
        normalized_search = self._normalize_required_string(search_term, "search_term")
        normalized_page, normalized_page_size, offset = self._validate_pagination(
            page=page,
            page_size=page_size,
        )
        search_pattern = f"%{normalized_search.lower()}%"
        count_parameters = {"search_pattern": search_pattern}
        list_parameters = {
            "search_pattern": search_pattern,
            "limit": normalized_page_size,
            "offset": offset,
        }
        predicate = (
            "LOWER(candidate_name) LIKE @search_pattern "
            "OR LOWER(candidate_email) LIKE @search_pattern "
            "OR LOWER(report_id) LIKE @search_pattern "
            "OR LOWER(session_id) LIKE @search_pattern"
        )
        count_sql = (
            "SELECT COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            f"WHERE {predicate}"
        )
        sql = (
            "SELECT report_id, session_id, candidate_id, candidate_name, candidate_email, "
            "interviewer, interview_stage, overall_score, overall_rating, recommendation, "
            "strengths, weaknesses, feedback, question_count, questions_answered, "
            "average_response_time, interview_duration, report_status, created_at, "
            "updated_at, created_by, updated_by "
            f"FROM {self._table_sql()} "
            f"WHERE {predicate} "
            "ORDER BY updated_at DESC, report_id ASC "
            "LIMIT @limit OFFSET @offset"
        )

        try:
            total_row = self._repository.fetch_one(count_sql, parameters=count_parameters)
            rows = self._repository.fetch_all(sql, parameters=list_parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("search_reports_failed", exc, search_term=normalized_search)
            raise ReportRepositoryError("Failed to search reports") from exc

        total = int(total_row["total_count"]) if total_row else 0
        return {
            "items": [self._row_to_record(row) for row in rows],
            "page": normalized_page,
            "page_size": normalized_page_size,
            "total": total,
            "total_pages": ceil(total / normalized_page_size) if total else 0,
            "has_next": offset + normalized_page_size < total,
            "has_previous": normalized_page > 1,
        }

    def get_reports_by_candidate(
        self,
        candidate_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        List reports for a candidate.

        Args:
            candidate_id: Candidate identifier.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and report items.
        """
        candidate_id_clean = self._normalize_required_string(candidate_id, "candidate_id")
        return self.list_reports(
            candidate=candidate_id_clean,
            page=page,
            page_size=page_size,
        )

    def get_reports_by_session(self, session_id: str) -> list[ReportRecord]:
        """
        Retrieve reports by interview session.

        Args:
            session_id: Interview session identifier.

        Returns:
            Matching report records.
        """
        session_id_clean = self._normalize_required_string(session_id, "session_id")
        sql = (
            "SELECT report_id, session_id, candidate_id, candidate_name, candidate_email, "
            "interviewer, interview_stage, overall_score, overall_rating, recommendation, "
            "strengths, weaknesses, feedback, question_count, questions_answered, "
            "average_response_time, interview_duration, report_status, created_at, "
            "updated_at, created_by, updated_by "
            f"FROM {self._table_sql()} "
            "WHERE session_id = @session_id "
            "ORDER BY created_at DESC, report_id ASC"
        )
        try:
            rows = self._repository.fetch_all(sql, parameters={"session_id": session_id_clean})
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("get_reports_by_session_failed", exc, session_id=session_id_clean)
            raise ReportRepositoryError("Failed to fetch reports by session") from exc
        return [self._row_to_record(row) for row in rows]

    def report_exists(
        self,
        report_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """
        Check whether a report exists by report id or session id.

        Args:
            report_id: Optional report identifier.
            session_id: Optional session identifier.

        Returns:
            True if a matching report exists, else False.
        """
        if report_id is None and session_id is None:
            raise InvalidReportError("Either report_id or session_id must be provided")

        clauses: list[str] = []
        parameters: dict[str, Any] = {}
        if report_id is not None:
            clauses.append("report_id = @report_id")
            parameters["report_id"] = self._normalize_required_string(report_id, "report_id")
        if session_id is not None:
            clauses.append("session_id = @session_id")
            parameters["session_id"] = self._normalize_required_string(
                session_id,
                "session_id",
            )

        sql = (
            "SELECT COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            f"WHERE {' OR '.join(clauses)}"
        )
        try:
            row = self._repository.fetch_one(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("report_exists_failed", exc, report_id=report_id, session_id=session_id)
            raise ReportRepositoryError("Failed to check report existence") from exc
        return bool(row and int(row["total_count"]) > 0)

    def count_reports(
        self,
        status: str | None = None,
        candidate: str | None = None,
        recommendation: str | None = None,
        stage: str | None = None,
        created_from: date | datetime | str | None = None,
        created_to: date | datetime | str | None = None,
    ) -> int:
        """
        Count reports using the same filters as list_reports.

        Args:
            status: Optional report status filter.
            candidate: Optional candidate identifier, name, or email filter.
            recommendation: Optional recommendation filter.
            stage: Optional interview stage filter.
            created_from: Optional inclusive creation date lower bound.
            created_to: Optional inclusive creation date upper bound.

        Returns:
            Matching report count.
        """
        where_clause, parameters = self._build_filter_clause(
            status=status,
            candidate=candidate,
            recommendation=recommendation,
            stage=stage,
            created_from=created_from,
            created_to=created_to,
        )
        sql = (
            "SELECT COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            f"{where_clause}"
        )
        try:
            row = self._repository.fetch_one(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("count_reports_failed", exc)
            raise ReportRepositoryError("Failed to count reports") from exc
        return int(row["total_count"]) if row else 0

    def get_statistics(self) -> dict[str, Any]:
        """
        Return aggregated report statistics.

        Returns:
            Summary counts and distributions for interview reports.
        """
        stats_sql = (
            "SELECT COUNT(1) AS total_reports, "
            "COUNTIF(report_status = 'COMPLETED') AS completed_reports, "
            "COUNTIF(report_status = 'FAILED') AS failed_reports, "
            "AVG(overall_score) AS average_score, "
            "AVG(interview_duration) AS average_duration, "
            "AVG(question_count) AS average_questions "
            f"FROM {self._table_sql()} "
            "WHERE report_status != 'DELETED'"
        )
        try:
            base_row = self._repository.fetch_one(stats_sql) or {}
            recommendation_distribution = self._group_statistics("recommendation")
            reports_per_stage = self._group_statistics("interview_stage")
            reports_per_day = self._group_by_day()
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("get_statistics_failed", exc)
            raise ReportRepositoryError("Failed to compute report statistics") from exc

        statistics = {
            "total_reports": int(base_row.get("total_reports", 0) or 0),
            "completed_reports": int(base_row.get("completed_reports", 0) or 0),
            "failed_reports": int(base_row.get("failed_reports", 0) or 0),
            "average_score": self._safe_float(base_row.get("average_score")),
            "average_duration": self._safe_float(base_row.get("average_duration")),
            "average_questions": self._safe_float(base_row.get("average_questions")),
            "recommendation_distribution": recommendation_distribution,
            "reports_per_stage": reports_per_stage,
            "reports_per_day": reports_per_day,
        }
        self.logger.info(
            "Interview report statistics generated",
            extra={
                "event": "report_statistics_generated",
                "total_reports": statistics["total_reports"],
            },
        )
        return statistics

    def _prepare_create_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        missing_fields = [
            field
            for field in sorted(self.CREATE_REQUIRED_FIELDS)
            if payload.get(field) is None or str(payload.get(field)).strip() == ""
        ]
        if missing_fields:
            raise InvalidReportError(
                f"Missing required report fields: {', '.join(missing_fields)}"
            )

        now = self._utcnow()
        prepared_payload = {
            "report_id": self._normalize_required_string(payload["report_id"], "report_id"),
            "session_id": self._normalize_required_string(payload["session_id"], "session_id"),
            "candidate_id": self._normalize_required_string(payload["candidate_id"], "candidate_id"),
            "candidate_name": self._normalize_required_string(
                payload["candidate_name"],
                "candidate_name",
            ),
            "candidate_email": self._normalize_email(payload["candidate_email"]),
            "interviewer": self._normalize_required_string(payload["interviewer"], "interviewer"),
            "interview_stage": self._normalize_stage(payload["interview_stage"]),
            "overall_score": self._normalize_score(payload["overall_score"]),
            "overall_rating": self._normalize_optional_string(payload.get("overall_rating")),
            "recommendation": self._normalize_recommendation(payload["recommendation"]),
            "strengths": self._normalize_optional_string(payload.get("strengths")),
            "weaknesses": self._normalize_optional_string(payload.get("weaknesses")),
            "feedback": self._normalize_optional_string(payload.get("feedback")),
            "question_count": self._normalize_non_negative_int(
                payload["question_count"],
                "question_count",
            ),
            "questions_answered": self._normalize_non_negative_int(
                payload["questions_answered"],
                "questions_answered",
            ),
            "average_response_time": self._normalize_non_negative_float(
                payload.get("average_response_time"),
                "average_response_time",
                required=False,
            ),
            "interview_duration": self._normalize_non_negative_float(
                payload["interview_duration"],
                "interview_duration",
                required=True,
            ),
            "report_status": self._normalize_report_status(
                payload.get("report_status", "COMPLETED")
            ),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "created_by": self._normalize_required_string(payload["created_by"], "created_by"),
            "updated_by": self._normalize_optional_string(payload.get("updated_by"))
            or self._normalize_required_string(payload["created_by"], "created_by"),
        }
        self._validate_question_metrics(
            question_count=prepared_payload["question_count"],
            questions_answered=prepared_payload["questions_answered"],
        )
        return prepared_payload

    def _prepare_update_payload(
        self,
        current: ReportRecord,
        updates: Mapping[str, Any],
        updated_by: str,
    ) -> dict[str, Any]:
        invalid_fields = sorted(set(updates.keys()) - self.UPDATE_ALLOWED_FIELDS)
        if invalid_fields:
            raise InvalidReportError(
                f"Unsupported report update fields: {', '.join(invalid_fields)}"
            )

        sanitized_updates: dict[str, Any] = {}
        if "candidate_name" in updates:
            sanitized_updates["candidate_name"] = self._normalize_required_string(
                updates["candidate_name"],
                "candidate_name",
            )
        if "candidate_email" in updates:
            sanitized_updates["candidate_email"] = self._normalize_email(
                updates["candidate_email"]
            )
        if "interviewer" in updates:
            sanitized_updates["interviewer"] = self._normalize_required_string(
                updates["interviewer"],
                "interviewer",
            )
        if "interview_stage" in updates:
            sanitized_updates["interview_stage"] = self._normalize_stage(
                updates["interview_stage"]
            )
        if "overall_score" in updates:
            sanitized_updates["overall_score"] = self._normalize_score(updates["overall_score"])
        if "overall_rating" in updates:
            sanitized_updates["overall_rating"] = self._normalize_optional_string(
                updates["overall_rating"]
            )
        if "recommendation" in updates:
            sanitized_updates["recommendation"] = self._normalize_recommendation(
                updates["recommendation"]
            )
        if "strengths" in updates:
            sanitized_updates["strengths"] = self._normalize_optional_string(updates["strengths"])
        if "weaknesses" in updates:
            sanitized_updates["weaknesses"] = self._normalize_optional_string(updates["weaknesses"])
        if "feedback" in updates:
            sanitized_updates["feedback"] = self._normalize_optional_string(updates["feedback"])
        if "question_count" in updates:
            sanitized_updates["question_count"] = self._normalize_non_negative_int(
                updates["question_count"],
                "question_count",
            )
        if "questions_answered" in updates:
            sanitized_updates["questions_answered"] = self._normalize_non_negative_int(
                updates["questions_answered"],
                "questions_answered",
            )
        if "average_response_time" in updates:
            sanitized_updates["average_response_time"] = self._normalize_non_negative_float(
                updates["average_response_time"],
                "average_response_time",
                required=False,
            )
        if "interview_duration" in updates:
            sanitized_updates["interview_duration"] = self._normalize_non_negative_float(
                updates["interview_duration"],
                "interview_duration",
                required=True,
            )
        if "report_status" in updates:
            sanitized_updates["report_status"] = self._normalize_report_status(
                updates["report_status"]
            )

        question_count = sanitized_updates.get("question_count", current.question_count)
        questions_answered = sanitized_updates.get(
            "questions_answered",
            current.questions_answered,
        )
        self._validate_question_metrics(
            question_count=question_count,
            questions_answered=questions_answered,
        )

        sanitized_updates["updated_by"] = self._normalize_required_string(
            updated_by,
            "updated_by",
        )
        sanitized_updates["updated_at"] = self._utcnow().isoformat()
        return sanitized_updates

    def _build_filter_clause(
        self,
        status: str | None = None,
        candidate: str | None = None,
        recommendation: str | None = None,
        stage: str | None = None,
        created_from: date | datetime | str | None = None,
        created_to: date | datetime | str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        parameters: dict[str, Any] = {}

        if status is not None:
            parameters["report_status"] = self._normalize_report_status(status)
            clauses.append("report_status = @report_status")

        if candidate is not None:
            candidate_search = self._normalize_required_string(candidate, "candidate")
            parameters["candidate_search"] = f"%{candidate_search.lower()}%"
            clauses.append(
                "("
                "LOWER(candidate_id) LIKE @candidate_search "
                "OR LOWER(candidate_name) LIKE @candidate_search "
                "OR LOWER(candidate_email) LIKE @candidate_search"
                ")"
            )

        if recommendation is not None:
            parameters["recommendation"] = self._normalize_recommendation(recommendation)
            clauses.append("recommendation = @recommendation")

        if stage is not None:
            parameters["interview_stage"] = self._normalize_stage(stage)
            clauses.append("interview_stage = @interview_stage")

        if created_from is not None:
            from_value = self._normalize_date_value(created_from, "created_from")
            parameters["created_from"] = from_value
            clauses.append("DATE(created_at) >= DATE(@created_from)")

        if created_to is not None:
            to_value = self._normalize_date_value(created_to, "created_to")
            parameters["created_to"] = to_value
            clauses.append("DATE(created_at) <= DATE(@created_to)")

        if created_from is not None and created_to is not None:
            from_value = self._normalize_date_value(created_from, "created_from")
            to_value = self._normalize_date_value(created_to, "created_to")
            if from_value > to_value:
                raise InvalidReportError("created_from must be less than or equal to created_to")

        if not clauses:
            return "", parameters
        return "WHERE " + " AND ".join(clauses), parameters

    def _build_update_statement(
        self,
        report_id: str,
        updates: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        assignments: list[str] = []
        parameters: dict[str, Any] = {"report_id": report_id}

        for field_name, value in updates.items():
            parameter_name = f"set_{field_name}"
            if field_name == "updated_at":
                assignments.append(f"{field_name} = TIMESTAMP(@{parameter_name})")
                parameters[parameter_name] = value
                continue

            assignments.append(f"{field_name} = @{parameter_name}")
            parameters[parameter_name] = value

        return ", ".join(assignments), parameters

    def _group_statistics(self, column_name: str) -> dict[str, int]:
        sql = (
            f"SELECT COALESCE(CAST({column_name} AS STRING), 'UNKNOWN') AS grouping_key, "
            "COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            "WHERE report_status != 'DELETED' "
            f"GROUP BY {column_name} "
            "ORDER BY total_count DESC, grouping_key ASC"
        )
        rows = self._repository.fetch_all(sql)
        return {str(row["grouping_key"]): int(row["total_count"]) for row in rows}

    def _group_by_day(self) -> dict[str, int]:
        sql = (
            "SELECT CAST(DATE(created_at) AS STRING) AS created_day, COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            "WHERE report_status != 'DELETED' "
            "GROUP BY DATE(created_at) "
            "ORDER BY DATE(created_at) ASC"
        )
        rows = self._repository.fetch_all(sql)
        return {str(row["created_day"]): int(row["total_count"]) for row in rows}

    def _row_to_record(self, row: Mapping[str, Any]) -> ReportRecord:
        return ReportRecord(
            report_id=self._normalize_required_string(row.get("report_id"), "report_id"),
            session_id=self._normalize_required_string(row.get("session_id"), "session_id"),
            candidate_id=self._normalize_required_string(row.get("candidate_id"), "candidate_id"),
            candidate_name=self._normalize_required_string(
                row.get("candidate_name"),
                "candidate_name",
            ),
            candidate_email=self._normalize_email(row.get("candidate_email")),
            interviewer=self._normalize_required_string(row.get("interviewer"), "interviewer"),
            interview_stage=self._normalize_stage(row.get("interview_stage")),
            overall_score=self._normalize_score(row.get("overall_score")),
            overall_rating=self._normalize_optional_string(row.get("overall_rating")),
            recommendation=self._normalize_recommendation(row.get("recommendation")),
            strengths=self._normalize_optional_string(row.get("strengths")),
            weaknesses=self._normalize_optional_string(row.get("weaknesses")),
            feedback=self._normalize_optional_string(row.get("feedback")),
            question_count=self._normalize_non_negative_int(row.get("question_count"), "question_count"),
            questions_answered=self._normalize_non_negative_int(
                row.get("questions_answered"),
                "questions_answered",
            ),
            average_response_time=self._normalize_non_negative_float(
                row.get("average_response_time"),
                "average_response_time",
                required=False,
            ),
            interview_duration=self._normalize_non_negative_float(
                row.get("interview_duration"),
                "interview_duration",
                required=True,
            ),
            report_status=self._normalize_report_status(row.get("report_status")),
            created_at=self._normalize_datetime(row.get("created_at"), required=True),
            updated_at=self._normalize_datetime(row.get("updated_at"), required=True),
            created_by=self._normalize_required_string(row.get("created_by"), "created_by"),
            updated_by=self._normalize_required_string(row.get("updated_by"), "updated_by"),
        )

    def _load_valid_stages(self) -> set[str]:
        content = self._load_yaml_file(self._config_path / "stages.yaml")
        valid_stages = {
            str(item["id"]).strip()
            for item in content.get("stages", [])
            if isinstance(item, dict) and item.get("id")
        }
        if not valid_stages:
            raise ReportRepositoryError("No stages were loaded from stages.yaml")
        return valid_stages

    def _load_valid_recommendations(self) -> set[str]:
        content = self._load_yaml_file(self._config_path / "thresholds.yaml")
        decision_thresholds = content.get("decision_thresholds", {})
        if not isinstance(decision_thresholds, dict) or not decision_thresholds:
            raise ReportRepositoryError(
                "No decision thresholds were loaded from thresholds.yaml"
            )
        return {str(key).strip().upper() for key in decision_thresholds.keys() if str(key).strip()}

    def _load_yaml_file(self, path: Path) -> dict[str, Any]:
        if yaml is None:
            raise ReportRepositoryError(
                "PyYAML is required to load config files; install pyyaml"
            )
        if not path.exists() or not path.is_file():
            raise ReportRepositoryError(f"Config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                content = yaml.safe_load(handle)
        except Exception as exc:
            raise ReportRepositoryError(
                f"Failed to load YAML config file {path.name}: {exc}"
            ) from exc
        if not isinstance(content, dict):
            raise ReportRepositoryError(
                f"Config file {path.name} must contain a YAML mapping at the root"
            )
        return content

    def _validate_question_metrics(self, question_count: int, questions_answered: int) -> None:
        if questions_answered > question_count:
            raise InvalidReportError("questions_answered cannot exceed question_count")

    def _normalize_stage(self, value: Any) -> str:
        stage = self._normalize_required_string(value, "interview_stage")
        normalized_stage = stage.upper()
        if normalized_stage not in {item.upper() for item in self._valid_stages}:
            raise InvalidReportError(f"Invalid interview_stage: {stage}")
        for valid_stage in self._valid_stages:
            if valid_stage.upper() == normalized_stage:
                return valid_stage
        raise InvalidReportError(f"Invalid interview_stage: {stage}")

    def _normalize_recommendation(self, value: Any) -> str:
        recommendation = self._normalize_required_string(value, "recommendation").upper()
        if recommendation not in self._valid_recommendations:
            raise InvalidReportError(f"Invalid recommendation: {recommendation}")
        return recommendation

    def _normalize_report_status(self, value: Any) -> str:
        report_status = self._normalize_required_string(value, "report_status").upper()
        if report_status not in self._valid_report_statuses:
            raise InvalidReportError(f"Invalid report_status: {report_status}")
        return report_status

    def _normalize_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidReportError("overall_score must be numeric") from exc
        if score < 0 or score > 5:
            raise InvalidReportError("overall_score must be between 0 and 5")
        return score

    def _normalize_non_negative_int(self, value: Any, field_name: str) -> int:
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidReportError(f"{field_name} must be an integer") from exc
        if integer_value < 0:
            raise InvalidReportError(f"{field_name} must be greater than or equal to 0")
        return integer_value

    def _normalize_non_negative_float(
        self,
        value: Any,
        field_name: str,
        *,
        required: bool,
    ) -> float | None:
        if value is None or value == "":
            if required:
                raise InvalidReportError(f"{field_name} is required")
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidReportError(f"{field_name} must be numeric") from exc
        if numeric_value < 0:
            raise InvalidReportError(f"{field_name} must be greater than or equal to 0")
        return numeric_value

    def _normalize_email(self, value: Any) -> str:
        email = self._normalize_required_string(value, "candidate_email").lower()
        if not EMAIL_PATTERN.match(email):
            raise InvalidReportError(f"Invalid candidate_email format: {email}")
        return email

    def _normalize_datetime(
        self,
        value: Any,
        *,
        required: bool,
    ) -> datetime:
        if value is None or value == "":
            if required:
                raise InvalidReportError("Required datetime value is missing")
            raise InvalidReportError("Datetime value cannot be empty")
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

        text_value = str(value).strip()
        if not text_value:
            raise InvalidReportError("Required datetime value is missing")
        normalized_text = text_value.replace("Z", "+00:00")
        try:
            parsed_value = datetime.fromisoformat(normalized_text)
        except ValueError as exc:
            raise InvalidReportError(f"Invalid datetime value: {value}") from exc
        return parsed_value if parsed_value.tzinfo else parsed_value.replace(tzinfo=timezone.utc)

    def _normalize_date_value(self, value: date | datetime | str, field_name: str) -> str:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        text_value = self._normalize_required_string(value, field_name)
        try:
            return date.fromisoformat(text_value).isoformat()
        except ValueError as exc:
            raise InvalidReportError(f"{field_name} must be an ISO date string") from exc

    def _normalize_required_string(self, value: Any, field_name: str) -> str:
        if value is None:
            raise InvalidReportError(f"{field_name} is required")
        normalized_value = str(value).strip()
        if not normalized_value:
            raise InvalidReportError(f"{field_name} is required")
        return normalized_value

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        normalized_value = str(value).strip()
        return normalized_value or None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def _table_sql(self) -> str:
        return f"`{self._repository.project}`.`{self._repository.dataset}`.`{REPORT_TABLE}`"

    def _validate_pagination(self, page: int, page_size: int) -> tuple[int, int, int]:
        if page < 1:
            raise InvalidReportError("page must be greater than or equal to 1")
        if page_size < 1:
            raise InvalidReportError("page_size must be greater than or equal to 1")
        return page, page_size, (page - 1) * page_size

    def _log_error(self, event: str, error: BaseException, **context: Any) -> None:
        self.logger.error(
            "Report repository operation failed",
            extra={
                "event": event,
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                **context,
            },
        )
