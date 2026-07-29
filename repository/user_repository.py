from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

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
USER_TABLE = "users"
CANDIDATE_LEVELS_COLUMN = "Candidate Levels"


class UserRepositoryError(Exception):
    """Raised when the user repository encounters a general failure."""


class UserAlreadyExistsError(UserRepositoryError):
    """Raised when a user already exists for a unique attribute."""


class UserNotFoundError(UserRepositoryError):
    """Raised when a requested user cannot be found."""


class InvalidUserError(UserRepositoryError):
    """Raised when user input fails validation."""


@dataclass(frozen=True)
class UserRecord:
    """Immutable representation of a persisted user record."""

    user_id: str
    email: str
    first_name: str
    last_name: str
    full_name: str
    role: str
    candidate_level: str
    experience_years: float
    department: str | None
    designation: str | None
    status: str
    is_active: bool
    last_login: datetime | None
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation of the record."""
        payload = asdict(self)
        for key in ("last_login", "created_at", "updated_at"):
            value = payload[key]
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
        return payload


class UserRepository:
    """Repository responsible for user persistence in BigQuery."""

    DEFAULT_STATUSES = frozenset({"ACTIVE", "INACTIVE", "PENDING", "SUSPENDED"})
    CREATE_REQUIRED_FIELDS = frozenset(
        {
            "user_id",
            "email",
            "first_name",
            "last_name",
            "role",
            "candidate_level",
            "experience_years",
            "created_by",
        }
    )
    UPDATE_ALLOWED_FIELDS = frozenset(
        {
            "email",
            "first_name",
            "last_name",
            "full_name",
            "role",
            "candidate_level",
            "experience_years",
            "department",
            "designation",
            "status",
            "is_active",
            "updated_by",
            "last_login",
        }
    )

    def __init__(
        self,
        bigquery_repository: BigQueryRepository,
        logger: logging.Logger | None = None,
        config_path: str | Path | None = None,
        question_bank_path: str | Path | None = None,
        valid_statuses: Iterable[str] | None = None,
    ) -> None:
        """
        Initialize the repository.

        Args:
            bigquery_repository: Shared BigQuery repository dependency.
            logger: Optional logger instance.
            config_path: Optional path to the application config directory.
            question_bank_path: Optional path to the question bank directory.
            valid_statuses: Optional supported user statuses.
        """
        self._repository = bigquery_repository
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()

        project_root = Path(__file__).resolve().parent.parent
        self._config_path = (
            Path(config_path) if config_path is not None else project_root / "config"
        )
        self._question_bank_path = (
            Path(question_bank_path)
            if question_bank_path is not None
            else project_root / "question_bank"
        )
        self._valid_statuses = {
            self._normalize_required_string(status, "status").upper()
            for status in (valid_statuses or self.DEFAULT_STATUSES)
        }
        self._valid_roles = self._load_valid_roles()
        self._valid_candidate_levels = self._load_valid_candidate_levels()

    def create_user(self, payload: Mapping[str, Any]) -> UserRecord:
        """
        Create a new user record.

        Args:
            payload: User payload containing required user fields.

        Returns:
            The created user record.

        Raises:
            InvalidUserError: If the payload is invalid.
            UserAlreadyExistsError: If a duplicate email or user identifier exists.
            UserRepositoryError: If persistence fails.
        """
        if not isinstance(payload, Mapping):
            raise InvalidUserError("payload must be a mapping of user attributes")

        with self._lock:
            validated_payload = self._prepare_create_payload(payload)
            if self.user_exists(user_id=validated_payload["user_id"]):
                raise UserAlreadyExistsError(
                    f"User ID already exists: {validated_payload['user_id']}"
                )
            if self.user_exists(email=validated_payload["email"]):
                raise UserAlreadyExistsError(
                    f"Email already exists: {validated_payload['email']}"
                )

            try:
                self._repository.insert(USER_TABLE, validated_payload)
            except (InsertError, BigQueryValidationError, BigQueryRepositoryError) as exc:
                self._log_error("create_user_failed", exc, user_id=validated_payload["user_id"])
                raise UserRepositoryError("Failed to create user") from exc

            record = self.get_user(validated_payload["user_id"])
            self.logger.info(
                "User created",
                extra={
                    "event": "user_created",
                    "user_id": record.user_id,
                    "email": record.email,
                    "role": record.role,
                },
            )
            return record

    def get_user(self, user_id: str) -> UserRecord:
        """
        Retrieve a user by user identifier.

        Args:
            user_id: Unique user identifier.

        Returns:
            The matching user record.

        Raises:
            UserNotFoundError: If the user does not exist.
            UserRepositoryError: If the query fails.
        """
        user_id_clean = self._normalize_required_string(user_id, "user_id")
        sql = (
            "SELECT user_id, email, first_name, last_name, full_name, role, "
            "candidate_level, experience_years, department, designation, status, "
            "is_active, last_login, created_at, updated_at, created_by, updated_by "
            f"FROM {self._table_sql()} "
            "WHERE user_id = @user_id "
            "LIMIT 1"
        )
        try:
            row = self._repository.fetch_one(sql, parameters={"user_id": user_id_clean})
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("get_user_failed", exc, user_id=user_id_clean)
            raise UserRepositoryError("Failed to fetch user") from exc

        if row is None:
            raise UserNotFoundError(f"User not found: {user_id_clean}")
        return self._row_to_record(row)

    def get_user_by_email(self, email: str) -> UserRecord:
        """
        Retrieve a user by email address using case-insensitive matching.

        Args:
            email: Email address to look up.

        Returns:
            The matching user record.

        Raises:
            UserNotFoundError: If the email does not exist.
            UserRepositoryError: If the query fails.
        """
        normalized_email = self._normalize_email(email)
        sql = (
            "SELECT user_id, email, first_name, last_name, full_name, role, "
            "candidate_level, experience_years, department, designation, status, "
            "is_active, last_login, created_at, updated_at, created_by, updated_by "
            f"FROM {self._table_sql()} "
            "WHERE LOWER(email) = LOWER(@email) "
            "LIMIT 1"
        )
        try:
            row = self._repository.fetch_one(sql, parameters={"email": normalized_email})
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("get_user_by_email_failed", exc, email=normalized_email)
            raise UserRepositoryError("Failed to fetch user by email") from exc

        if row is None:
            raise UserNotFoundError(f"User not found for email: {normalized_email}")
        return self._row_to_record(row)

    def update_user(
        self,
        user_id: str,
        updates: Mapping[str, Any],
        updated_by: str,
    ) -> UserRecord:
        """
        Update allowed columns for a user.

        Args:
            user_id: Unique user identifier.
            updates: Partial update payload.
            updated_by: Actor performing the update.

        Returns:
            The updated user record.

        Raises:
            UserNotFoundError: If the user does not exist.
            InvalidUserError: If the update payload is invalid.
            UserRepositoryError: If persistence fails.
        """
        if not isinstance(updates, Mapping) or not updates:
            raise InvalidUserError("updates must be a non-empty mapping")

        with self._lock:
            current = self.get_user(user_id)
            sanitized_updates = self._prepare_update_payload(
                current=current,
                updates=updates,
                updated_by=updated_by,
            )
            if not sanitized_updates:
                return current

            assignments, parameters = self._build_update_statement(
                user_id=current.user_id,
                updates=sanitized_updates,
            )
            sql = (
                f"UPDATE {self._table_sql()} "
                f"SET {assignments} "
                "WHERE user_id = @user_id"
            )
            try:
                affected_rows = self._repository.update(sql, parameters=parameters)
            except (UpdateError, BigQueryValidationError, BigQueryRepositoryError) as exc:
                self._log_error("update_user_failed", exc, user_id=current.user_id)
                raise UserRepositoryError("Failed to update user") from exc

            if affected_rows == 0:
                raise UserNotFoundError(f"User not found: {current.user_id}")

            record = self.get_user(current.user_id)
            self.logger.info(
                "User updated",
                extra={
                    "event": "user_updated",
                    "user_id": record.user_id,
                    "updated_fields": sorted(sanitized_updates.keys()),
                },
            )
            return record

    def delete_user(self, user_id: str, updated_by: str) -> UserRecord:
        """
        Soft delete a user by marking the record inactive.

        Args:
            user_id: Unique user identifier.
            updated_by: Actor performing the operation.

        Returns:
            The updated inactive user record.
        """
        return self.update_user(
            user_id=user_id,
            updates={"status": "INACTIVE", "is_active": False},
            updated_by=updated_by,
        )

    def activate_user(self, user_id: str, updated_by: str) -> UserRecord:
        """
        Reactivate a previously inactive user.

        Args:
            user_id: Unique user identifier.
            updated_by: Actor performing the operation.

        Returns:
            The updated active user record.
        """
        return self.update_user(
            user_id=user_id,
            updates={"status": "ACTIVE", "is_active": True},
            updated_by=updated_by,
        )

    def update_last_login(self, user_id: str) -> UserRecord:
        """
        Update the last login timestamp for a user.

        Args:
            user_id: Unique user identifier.

        Returns:
            The updated user record.
        """
        current = self.get_user(user_id)
        actor = current.updated_by or current.user_id
        record = self.update_user(
            user_id=user_id,
            updates={"last_login": self._utcnow()},
            updated_by=actor,
        )
        self.logger.info(
            "User last login updated",
            extra={"event": "user_last_login_updated", "user_id": record.user_id},
        )
        return record

    def assign_role(self, user_id: str, role: str, updated_by: str) -> UserRecord:
        """
        Assign a validated role to a user.

        Args:
            user_id: Unique user identifier.
            role: Role identifier to assign.
            updated_by: Actor performing the assignment.

        Returns:
            The updated user record.
        """
        normalized_role = self._normalize_role(role)
        record = self.update_user(
            user_id=user_id,
            updates={"role": normalized_role},
            updated_by=updated_by,
        )
        self.logger.info(
            "User role assigned",
            extra={
                "event": "user_role_assigned",
                "user_id": record.user_id,
                "role": record.role,
            },
        )
        return record

    def list_users(
        self,
        role: str | None = None,
        department: str | None = None,
        candidate_level: str | None = None,
        status: str | None = None,
        active: bool | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        List users using filters and offset-based pagination.

        Args:
            role: Optional role filter.
            department: Optional department filter.
            candidate_level: Optional candidate level filter.
            status: Optional status filter.
            active: Optional active flag filter.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and user items.
        """
        normalized_page, normalized_page_size, offset = self._validate_pagination(
            page=page,
            page_size=page_size,
        )
        where_clause, parameters = self._build_filter_clause(
            role=role,
            department=department,
            candidate_level=candidate_level,
            status=status,
            active=active,
        )
        total = self.count_users(
            role=role,
            department=department,
            candidate_level=candidate_level,
            status=status,
            active=active,
        )
        sql = (
            "SELECT user_id, email, first_name, last_name, full_name, role, "
            "candidate_level, experience_years, department, designation, status, "
            "is_active, last_login, created_at, updated_at, created_by, updated_by "
            f"FROM {self._table_sql()} "
            f"{where_clause} "
            "ORDER BY created_at DESC, email ASC "
            "LIMIT @limit OFFSET @offset"
        )
        parameters.update({"limit": normalized_page_size, "offset": offset})

        try:
            rows = self._repository.fetch_all(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("list_users_failed", exc)
            raise UserRepositoryError("Failed to list users") from exc

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

    def search_users(
        self,
        search_term: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        Search users by name, email, or department.

        Args:
            search_term: Free-text search input.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and matching users.
        """
        normalized_search = self._normalize_required_string(search_term, "search_term")
        normalized_page, normalized_page_size, offset = self._validate_pagination(
            page=page,
            page_size=page_size,
        )
        search_pattern = f"%{normalized_search.lower()}%"
        parameters = {
            "search_pattern": search_pattern,
            "limit": normalized_page_size,
            "offset": offset,
        }
        count_parameters = {"search_pattern": search_pattern}

        count_sql = (
            f"SELECT COUNT(1) AS total_count FROM {self._table_sql()} "
            "WHERE LOWER(full_name) LIKE @search_pattern "
            "OR LOWER(email) LIKE @search_pattern "
            "OR LOWER(IFNULL(department, '')) LIKE @search_pattern"
        )
        sql = (
            "SELECT user_id, email, first_name, last_name, full_name, role, "
            "candidate_level, experience_years, department, designation, status, "
            "is_active, last_login, created_at, updated_at, created_by, updated_by "
            f"FROM {self._table_sql()} "
            "WHERE LOWER(full_name) LIKE @search_pattern "
            "OR LOWER(email) LIKE @search_pattern "
            "OR LOWER(IFNULL(department, '')) LIKE @search_pattern "
            "ORDER BY updated_at DESC, email ASC "
            "LIMIT @limit OFFSET @offset"
        )

        try:
            total_row = self._repository.fetch_one(
                count_sql,
                parameters=count_parameters,
            )
            rows = self._repository.fetch_all(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("search_users_failed", exc, search_term=normalized_search)
            raise UserRepositoryError("Failed to search users") from exc

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

    def user_exists(
        self,
        user_id: str | None = None,
        email: str | None = None,
    ) -> bool:
        """
        Check whether a user exists by identifier or email.

        Args:
            user_id: Optional unique user identifier.
            email: Optional email address.

        Returns:
            True if a matching user exists, else False.
        """
        if user_id is None and email is None:
            raise InvalidUserError("Either user_id or email must be provided")

        clauses: list[str] = []
        parameters: dict[str, Any] = {}
        if user_id is not None:
            clauses.append("user_id = @user_id")
            parameters["user_id"] = self._normalize_required_string(user_id, "user_id")
        if email is not None:
            clauses.append("LOWER(email) = LOWER(@email)")
            parameters["email"] = self._normalize_email(email)

        sql = (
            "SELECT COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            f"WHERE {' OR '.join(clauses)}"
        )
        try:
            row = self._repository.fetch_one(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("user_exists_failed", exc, user_id=user_id, email=email)
            raise UserRepositoryError("Failed to check user existence") from exc
        return bool(row and int(row["total_count"]) > 0)

    def count_users(
        self,
        role: str | None = None,
        department: str | None = None,
        candidate_level: str | None = None,
        status: str | None = None,
        active: bool | None = None,
    ) -> int:
        """
        Count users using the same filters as list_users.

        Args:
            role: Optional role filter.
            department: Optional department filter.
            candidate_level: Optional candidate level filter.
            status: Optional status filter.
            active: Optional active flag filter.

        Returns:
            Matching user count.
        """
        where_clause, parameters = self._build_filter_clause(
            role=role,
            department=department,
            candidate_level=candidate_level,
            status=status,
            active=active,
        )
        sql = (
            "SELECT COUNT(1) AS total_count "
            f"FROM {self._table_sql()} "
            f"{where_clause}"
        )
        try:
            row = self._repository.fetch_one(sql, parameters=parameters)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("count_users_failed", exc)
            raise UserRepositoryError("Failed to count users") from exc
        return int(row["total_count"]) if row else 0

    def get_statistics(self) -> dict[str, Any]:
        """
        Return aggregated user statistics.

        Returns:
            Summary counts grouped by status, role, department, and candidate level.
        """
        statistics = {
            "total_users": self.count_users(),
            "active_users": self.count_users(active=True),
            "inactive_users": self.count_users(active=False),
            "users_per_role": self._group_statistics("role"),
            "users_per_department": self._group_statistics("department"),
            "candidate_levels": self._group_statistics("candidate_level"),
        }
        self.logger.info(
            "User statistics generated",
            extra={"event": "user_statistics_generated", "total_users": statistics["total_users"]},
        )
        return statistics

    def _prepare_create_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        missing_fields = [
            field
            for field in sorted(self.CREATE_REQUIRED_FIELDS)
            if payload.get(field) is None or str(payload.get(field)).strip() == ""
        ]
        if missing_fields:
            raise InvalidUserError(
                f"Missing required user fields: {', '.join(missing_fields)}"
            )

        now = self._utcnow()
        first_name = self._normalize_required_string(payload["first_name"], "first_name")
        last_name = self._normalize_required_string(payload["last_name"], "last_name")
        full_name = self._build_full_name(
            payload.get("full_name"),
            first_name=first_name,
            last_name=last_name,
        )
        status = self._normalize_status(payload.get("status", "ACTIVE"))
        is_active = self._coerce_bool(payload.get("is_active", status == "ACTIVE"), "is_active")
        experience_years = self._normalize_experience(payload["experience_years"])
        created_by = self._normalize_required_string(payload["created_by"], "created_by")
        updated_by = self._normalize_optional_string(payload.get("updated_by")) or created_by
        last_login = self._normalize_datetime(payload.get("last_login"))

        prepared_payload = {
            "user_id": self._normalize_required_string(payload["user_id"], "user_id"),
            "email": self._normalize_email(payload["email"]),
            "first_name": first_name,
            "last_name": last_name,
            "full_name": full_name,
            "role": self._normalize_role(payload["role"]),
            "candidate_level": self._normalize_candidate_level(payload["candidate_level"]),
            "experience_years": experience_years,
            "department": self._normalize_optional_string(payload.get("department")),
            "designation": self._normalize_optional_string(payload.get("designation")),
            "status": status,
            "is_active": is_active,
            "last_login": last_login.isoformat() if last_login else None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "created_by": created_by,
            "updated_by": updated_by,
        }
        self._validate_status_and_activity(
            status=prepared_payload["status"],
            is_active=prepared_payload["is_active"],
        )
        return prepared_payload

    def _prepare_update_payload(
        self,
        current: UserRecord,
        updates: Mapping[str, Any],
        updated_by: str,
    ) -> dict[str, Any]:
        invalid_fields = sorted(set(updates.keys()) - self.UPDATE_ALLOWED_FIELDS)
        if invalid_fields:
            raise InvalidUserError(
                f"Unsupported user update fields: {', '.join(invalid_fields)}"
            )

        sanitized_updates: dict[str, Any] = {}
        if "email" in updates:
            normalized_email = self._normalize_email(updates["email"])
            if normalized_email.lower() != current.email.lower() and self.user_exists(
                email=normalized_email
            ):
                raise UserAlreadyExistsError(f"Email already exists: {normalized_email}")
            sanitized_updates["email"] = normalized_email

        if "first_name" in updates:
            sanitized_updates["first_name"] = self._normalize_required_string(
                updates["first_name"],
                "first_name",
            )

        if "last_name" in updates:
            sanitized_updates["last_name"] = self._normalize_required_string(
                updates["last_name"],
                "last_name",
            )

        if "full_name" in updates:
            first_name = sanitized_updates.get("first_name", current.first_name)
            last_name = sanitized_updates.get("last_name", current.last_name)
            sanitized_updates["full_name"] = self._build_full_name(
                updates["full_name"],
                first_name=first_name,
                last_name=last_name,
            )
        elif "first_name" in sanitized_updates or "last_name" in sanitized_updates:
            first_name = sanitized_updates.get("first_name", current.first_name)
            last_name = sanitized_updates.get("last_name", current.last_name)
            sanitized_updates["full_name"] = self._build_full_name(
                None,
                first_name=first_name,
                last_name=last_name,
            )

        if "role" in updates:
            sanitized_updates["role"] = self._normalize_role(updates["role"])

        if "candidate_level" in updates:
            sanitized_updates["candidate_level"] = self._normalize_candidate_level(
                updates["candidate_level"]
            )

        if "experience_years" in updates:
            sanitized_updates["experience_years"] = self._normalize_experience(
                updates["experience_years"]
            )

        if "department" in updates:
            sanitized_updates["department"] = self._normalize_optional_string(
                updates["department"]
            )

        if "designation" in updates:
            sanitized_updates["designation"] = self._normalize_optional_string(
                updates["designation"]
            )

        status_value = current.status
        if "status" in updates:
            status_value = self._normalize_status(updates["status"])
            sanitized_updates["status"] = status_value

        is_active_value = current.is_active
        if "is_active" in updates:
            is_active_value = self._coerce_bool(updates["is_active"], "is_active")
            sanitized_updates["is_active"] = is_active_value

        self._validate_status_and_activity(
            status=status_value,
            is_active=is_active_value,
        )

        if "last_login" in updates:
            last_login = self._normalize_datetime(updates["last_login"])
            sanitized_updates["last_login"] = last_login.isoformat() if last_login else None

        sanitized_updates["updated_by"] = self._normalize_required_string(
            updated_by,
            "updated_by",
        )
        sanitized_updates["updated_at"] = self._utcnow().isoformat()
        return sanitized_updates

    def _build_filter_clause(
        self,
        role: str | None = None,
        department: str | None = None,
        candidate_level: str | None = None,
        status: str | None = None,
        active: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        parameters: dict[str, Any] = {}

        if role is not None:
            parameters["role"] = self._normalize_role(role)
            clauses.append("LOWER(role) = LOWER(@role)")

        if department is not None:
            parameters["department"] = self._normalize_required_string(
                department,
                "department",
            )
            clauses.append("LOWER(IFNULL(department, '')) = LOWER(@department)")

        if candidate_level is not None:
            parameters["candidate_level"] = self._normalize_candidate_level(
                candidate_level
            )
            clauses.append("LOWER(candidate_level) = LOWER(@candidate_level)")

        if status is not None:
            parameters["status"] = self._normalize_status(status)
            clauses.append("status = @status")

        if active is not None:
            parameters["active"] = self._coerce_bool(active, "active")
            clauses.append("is_active = @active")

        if not clauses:
            return "", parameters
        return "WHERE " + " AND ".join(clauses), parameters

    def _build_update_statement(
        self,
        user_id: str,
        updates: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        assignments: list[str] = []
        parameters: dict[str, Any] = {"user_id": user_id}

        for field_name, value in updates.items():
            parameter_name = f"set_{field_name}"
            if field_name in {"last_login", "updated_at"}:
                assignments.append(
                    f"{field_name} = "
                    f"{'NULL' if value is None else f'TIMESTAMP(@{parameter_name})'}"
                )
                if value is not None:
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
            f"GROUP BY {column_name} "
            "ORDER BY total_count DESC, grouping_key ASC"
        )
        try:
            rows = self._repository.fetch_all(sql)
        except (QueryExecutionError, BigQueryRepositoryError) as exc:
            self._log_error("group_statistics_failed", exc, column_name=column_name)
            raise UserRepositoryError("Failed to compute user statistics") from exc
        return {str(row["grouping_key"]): int(row["total_count"]) for row in rows}

    def _row_to_record(self, row: Mapping[str, Any]) -> UserRecord:
        return UserRecord(
            user_id=self._normalize_required_string(row.get("user_id"), "user_id"),
            email=self._normalize_email(row.get("email")),
            first_name=self._normalize_required_string(row.get("first_name"), "first_name"),
            last_name=self._normalize_required_string(row.get("last_name"), "last_name"),
            full_name=self._normalize_required_string(row.get("full_name"), "full_name"),
            role=self._normalize_role(row.get("role")),
            candidate_level=self._normalize_candidate_level(row.get("candidate_level")),
            experience_years=self._normalize_experience(row.get("experience_years")),
            department=self._normalize_optional_string(row.get("department")),
            designation=self._normalize_optional_string(row.get("designation")),
            status=self._normalize_status(row.get("status")),
            is_active=self._coerce_bool(row.get("is_active"), "is_active"),
            last_login=self._normalize_datetime(row.get("last_login")),
            created_at=self._normalize_datetime(row.get("created_at"), required=True),
            updated_at=self._normalize_datetime(row.get("updated_at"), required=True),
            created_by=self._normalize_required_string(row.get("created_by"), "created_by"),
            updated_by=self._normalize_required_string(row.get("updated_by"), "updated_by"),
        )

    def _load_valid_roles(self) -> set[str]:
        if yaml is None:
            raise UserRepositoryError(
                "PyYAML is required to load roles.yaml; install pyyaml"
            )

        roles_path = self._config_path / "roles.yaml"
        if not roles_path.exists() or not roles_path.is_file():
            raise UserRepositoryError(f"Role configuration file not found: {roles_path}")

        try:
            with roles_path.open("r", encoding="utf-8") as handle:
                content = yaml.safe_load(handle)
        except Exception as exc:
            raise UserRepositoryError(
                f"Failed to load role configuration from {roles_path.name}: {exc}"
            ) from exc

        if not isinstance(content, dict):
            raise UserRepositoryError("roles.yaml must contain a YAML mapping at the root")

        valid_roles = {
            str(item["id"]).strip()
            for item in content.get("roles", [])
            if isinstance(item, dict) and item.get("id")
        }
        if not valid_roles:
            raise UserRepositoryError("No roles were loaded from roles.yaml")
        return valid_roles

    def _load_valid_candidate_levels(self) -> set[str]:
        if not self._question_bank_path.exists() or not self._question_bank_path.is_dir():
            self.logger.warning(
                "Question bank path not found while loading candidate levels",
                extra={
                    "event": "candidate_level_reference_missing",
                    "path": str(self._question_bank_path),
                },
            )
            return set()

        candidate_levels: set[str] = set()
        workbook_paths = sorted(
            path
            for path in self._question_bank_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".xlsx"
        )
        for workbook_path in workbook_paths:
            try:
                frame = pd.read_excel(
                    workbook_path,
                    usecols=[CANDIDATE_LEVELS_COLUMN],
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to extract candidate levels from workbook",
                    extra={
                        "event": "candidate_level_reference_load_failed",
                        "workbook": workbook_path.name,
                        "error": str(exc),
                    },
                )
                continue

            for raw_value in frame[CANDIDATE_LEVELS_COLUMN].dropna().tolist():
                for part in str(raw_value).split(","):
                    normalized = part.strip()
                    if normalized:
                        candidate_levels.add(normalized)

        return candidate_levels

    def _validate_status_and_activity(self, status: str, is_active: bool) -> None:
        if status == "INACTIVE" and is_active:
            raise InvalidUserError("INACTIVE users cannot have is_active=True")
        if status == "ACTIVE" and not is_active:
            raise InvalidUserError("ACTIVE users cannot have is_active=False")

    def _normalize_role(self, value: Any) -> str:
        role = self._normalize_required_string(value, "role")
        normalized_role = role.lower()
        if normalized_role not in {item.lower() for item in self._valid_roles}:
            raise InvalidUserError(f"Invalid role: {role}")
        for valid_role in self._valid_roles:
            if valid_role.lower() == normalized_role:
                return valid_role
        raise InvalidUserError(f"Invalid role: {role}")

    def _normalize_candidate_level(self, value: Any) -> str:
        candidate_level = self._normalize_required_string(value, "candidate_level")
        if self._valid_candidate_levels:
            normalized_candidate_level = candidate_level.lower()
            for valid_candidate_level in self._valid_candidate_levels:
                if valid_candidate_level.lower() == normalized_candidate_level:
                    return valid_candidate_level
            raise InvalidUserError(f"Invalid candidate_level: {candidate_level}")
        return candidate_level

    def _normalize_status(self, value: Any) -> str:
        status = self._normalize_required_string(value, "status").upper()
        if status not in self._valid_statuses:
            raise InvalidUserError(f"Invalid status: {status}")
        return status

    def _normalize_experience(self, value: Any) -> float:
        try:
            experience_years = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidUserError("experience_years must be numeric") from exc

        if experience_years < 0:
            raise InvalidUserError("experience_years must be greater than or equal to 0")
        return experience_years

    def _normalize_email(self, value: Any) -> str:
        email = self._normalize_required_string(value, "email").lower()
        if not EMAIL_PATTERN.match(email):
            raise InvalidUserError(f"Invalid email format: {email}")
        return email

    def _normalize_datetime(
        self,
        value: Any,
        *,
        required: bool = False,
    ) -> datetime | None:
        if value is None or value == "":
            if required:
                raise InvalidUserError("Required datetime value is missing")
            return None

        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

        text_value = str(value).strip()
        if not text_value:
            if required:
                raise InvalidUserError("Required datetime value is missing")
            return None

        normalized_text = text_value.replace("Z", "+00:00")
        try:
            parsed_value = datetime.fromisoformat(normalized_text)
        except ValueError as exc:
            raise InvalidUserError(f"Invalid datetime value: {value}") from exc
        return parsed_value if parsed_value.tzinfo else parsed_value.replace(tzinfo=timezone.utc)

    def _normalize_required_string(self, value: Any, field_name: str) -> str:
        if value is None:
            raise InvalidUserError(f"{field_name} is required")
        normalized_value = str(value).strip()
        if not normalized_value:
            raise InvalidUserError(f"{field_name} is required")
        return normalized_value

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        normalized_value = str(value).strip()
        return normalized_value or None

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
        raise InvalidUserError(f"{field_name} must be a boolean value")

    @staticmethod
    def _build_full_name(
        provided_value: Any,
        *,
        first_name: str,
        last_name: str,
    ) -> str:
        if provided_value is not None and str(provided_value).strip():
            return str(provided_value).strip()
        return f"{first_name} {last_name}".strip()

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def _table_sql(self) -> str:
        project = self._repository.project
        dataset = self._repository.dataset
        return f"`{project}`.`{dataset}`.`{USER_TABLE}`"

    def _validate_pagination(self, page: int, page_size: int) -> tuple[int, int, int]:
        if page < 1:
            raise InvalidUserError("page must be greater than or equal to 1")
        if page_size < 1:
            raise InvalidUserError("page_size must be greater than or equal to 1")
        return page, page_size, (page - 1) * page_size

    def _log_error(self, event: str, error: BaseException, **context: Any) -> None:
        self.logger.error(
            "User repository operation failed",
            extra={
                "event": event,
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                **context,
            },
        )
