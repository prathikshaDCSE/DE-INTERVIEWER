from __future__ import annotations

import logging
import re
import threading
import uuid
from math import ceil
from typing import Any, Mapping

from repository.audit_repository import AuditRepository
from repository.user_repository import (
    InvalidUserError as RepositoryInvalidUserError,
    UserAlreadyExistsError as RepositoryUserAlreadyExistsError,
    UserNotFoundError as RepositoryUserNotFoundError,
    UserRecord,
    UserRepository,
    UserRepositoryError as RepositoryUserRepositoryError,
)


# ============================================================
# Exceptions
# ============================================================


class UserServiceError(Exception):
    """Base exception raised by UserService."""


class UserAlreadyExistsError(UserServiceError):
    """Raised when a candidate already exists for a unique attribute."""


class UserValidationError(UserServiceError):
    """Raised when candidate input fails validation."""


class UserOperationError(UserServiceError):
    """Raised when a repository operation fails for reasons other than
    validation, duplication, or a missing record."""


class CandidateNotFoundError(UserServiceError):
    """Raised when a requested candidate cannot be found."""


# ============================================================
# User Service
# ============================================================


class UserService:
    """
    Business layer responsible for candidate account management.

    Responsibilities

    • Candidate registration
    • Candidate profile updates
    • Candidate soft deletion
    • Candidate lookup and search
    • Candidate statistics
    • Audit logging

    This service never issues SQL and never talks to BigQuery
    directly; all persistence is delegated to ``UserRepository``.
    ``UserService`` is intentionally scoped to the ``candidate`` role
    only. Reviewer and Admin account management is handled by other
    services; any repository record whose role is not ``candidate``
    is treated as not found by this service.
    """

    # =======================================================
    # Class Constants (Business Rules)
    # =======================================================

    CANDIDATE_ROLE = "candidate"

    DEFAULT_PAGE = 1

    DEFAULT_PAGE_SIZE = 50

    STATISTICS_PAGE_SIZE = 100

    MIN_EXPERIENCE_YEARS = 0.0

    MAX_NAME_LENGTH = 100

    MAX_EMAIL_LENGTH = 254

    # Candidate id format. No custom identifier format (e.g.
    # "CAND-000001") was specified for this project, so ids are
    # generated as random, collision-resistant hex tokens. A sequential
    # format would require an atomic counter maintained by the
    # repository/BigQuery (uuid4 alone cannot safely produce contiguous
    # ids across concurrent service instances); if a custom format is
    # introduced later, only ``_generate_candidate_id`` needs to change.
    CANDIDATE_ID_PREFIX = ""

    # Candidate-level string values that map onto each statistics
    # tier. Statistics prefer the stored ``candidate_level`` field over
    # derived experience when the value is recognized here; unmapped
    # or unrecognized ``candidate_level`` values fall back to the
    # experience-based thresholds below. Extend this mapping if the
    # project defines additional canonical level labels.
    CANDIDATE_LEVEL_TIER_ALIASES = {
        "fresher": "freshers",
        "freshers": "freshers",
        "entry": "freshers",
        "entry-level": "freshers",
        "entry level": "freshers",
        "junior": "juniors",
        "juniors": "juniors",
        "junior-level": "juniors",
        "mid": "mid_level",
        "mid-level": "mid_level",
        "mid level": "mid_level",
        "intermediate": "mid_level",
        "senior": "senior",
        "senior-level": "senior",
        "expert": "senior",
    }

    # Fallback experience-based tiers, used only when a candidate's
    # ``candidate_level`` value is not present in
    # ``CANDIDATE_LEVEL_TIER_ALIASES``.
    FRESHER_MAX_YEARS = 1.0

    JUNIOR_MAX_YEARS = 3.0

    MID_MAX_YEARS = 7.0

    EMAIL_PATTERN = re.compile(
        r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$",
        re.IGNORECASE,
    )

    # Search scanning safeguards. ``UserRepository`` cannot filter by
    # free-text search combined with candidate_level/status/experience
    # in a single query, so an accurate (non-misleading) paginated
    # search requires scanning matching pages and re-paginating
    # in-service. These constants bound that scan.
    SEARCH_SCAN_PAGE_SIZE = 200

    MAX_SEARCH_SCAN_RECORDS = 5000

    # =======================================================
    # Audit Action Constants
    # =======================================================

    ACTION_CANDIDATE_REGISTERED = "CANDIDATE_REGISTERED"

    ACTION_CANDIDATE_REGISTRATION_FAILED = "CANDIDATE_REGISTRATION_FAILED"

    ACTION_CANDIDATE_UPDATED = "CANDIDATE_UPDATED"

    ACTION_CANDIDATE_UPDATE_FAILED = "CANDIDATE_UPDATE_FAILED"

    ACTION_CANDIDATE_DELETED = "CANDIDATE_DELETED"

    ACTION_CANDIDATE_DELETE_FAILED = "CANDIDATE_DELETE_FAILED"

    ACTION_CANDIDATE_VIEWED = "CANDIDATE_VIEWED"

    ACTION_CANDIDATE_VIEW_FAILED = "CANDIDATE_VIEW_FAILED"

    ACTION_CANDIDATE_SEARCHED = "CANDIDATE_SEARCHED"

    ACTION_CANDIDATE_SEARCH_FAILED = "CANDIDATE_SEARCH_FAILED"

    def __init__(
        self,
        user_repository: UserRepository,
        audit_repository: AuditRepository,
        logger: logging.Logger | None = None,
    ) -> None:

        self.user_repository = user_repository

        self.audit_repository = audit_repository

        self.logger = logger or logging.getLogger(
            self.__class__.__name__
        )

        self._lock = threading.RLock()

    # =======================================================
    # Internal Helpers
    # =======================================================

    def _generate_candidate_id(self) -> str:
        """
        Generate a globally unique candidate identifier.
        """
        return f"{self.CANDIDATE_ID_PREFIX}{uuid.uuid4().hex}"

    def _validate_name(
        self,
        value: Any,
        field_name: str,
    ) -> None:
        """
        Validate a candidate name field.

        Args:
            value: Name value to validate.
            field_name: Field name used in error messages.

        Raises:
            UserValidationError: If the value is blank, whitespace
                only, or exceeds ``MAX_NAME_LENGTH``.
        """

        if not isinstance(value, str) or not value.strip():
            raise UserValidationError(
                f"{field_name} must be a non-empty string"
            )

        if len(value.strip()) > self.MAX_NAME_LENGTH:
            raise UserValidationError(
                f"{field_name} must be at most "
                f"{self.MAX_NAME_LENGTH} characters"
            )

    def _validate_email(
        self,
        email: Any,
    ) -> None:
        """
        Validate that a value is a well-formed, length-bounded email
        address.

        Args:
            email: Candidate email address.

        Raises:
            UserValidationError: If the value is not a valid email or
                exceeds ``MAX_EMAIL_LENGTH``.
        """

        if not isinstance(email, str) or not self.EMAIL_PATTERN.match(
            email.strip()
        ):
            raise UserValidationError(
                f"Invalid email format: {email}"
            )

        if len(email.strip()) > self.MAX_EMAIL_LENGTH:
            raise UserValidationError(
                f"email must be at most {self.MAX_EMAIL_LENGTH} "
                "characters"
            )

    def _validate_experience(
        self,
        experience_years: Any,
    ) -> None:
        """
        Validate that a value is a non-negative numeric experience.

        Args:
            experience_years: Candidate experience in years.

        Raises:
            UserValidationError: If the value is not a non-negative
                number.
        """

        if isinstance(experience_years, bool) or not isinstance(
            experience_years, (int, float)
        ):
            raise UserValidationError(
                "experience_years must be numeric, got: "
                f"{type(experience_years).__name__}"
            )

        if experience_years < self.MIN_EXPERIENCE_YEARS:
            raise UserValidationError(
                "experience_years must be greater than or equal to "
                f"{self.MIN_EXPERIENCE_YEARS}, got: {experience_years}"
            )

    def _validate_pagination(
        self,
        page: int,
        page_size: int,
    ) -> None:
        """
        Validate pagination arguments for manually paginated results.

        Raises:
            UserValidationError: If ``page`` or ``page_size`` is not a
                positive integer.
        """

        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or page < 1
        ):
            raise UserValidationError(
                "page must be an integer greater than or equal to 1"
            )

        if (
            not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or page_size < 1
        ):
            raise UserValidationError(
                "page_size must be an integer greater than or equal "
                "to 1"
            )

    def _validate_candidate(
        self,
        payload: Mapping[str, Any],
    ) -> None:
        """
        Validate a candidate registration payload.

        ``candidate_level`` and ``status`` values are intentionally
        not cross-checked here: ``UserRepository`` loads the
        authoritative set of valid values from configuration and the
        question bank but does not expose them publicly, so
        duplicating that list here would risk drifting out of sync.
        Invalid values still surface correctly as
        ``UserValidationError`` once translated from the repository's
        ``InvalidUserError``.

        Args:
            payload: Candidate registration attributes.

        Raises:
            UserValidationError: If required fields are missing or
                invalid.
        """

        required_fields = (
            "email",
            "first_name",
            "last_name",
            "candidate_level",
            "experience_years",
        )

        missing_fields = [
            field_name
            for field_name in required_fields
            if payload.get(field_name) in (None, "")
        ]

        if missing_fields:
            raise UserValidationError(
                "Missing required candidate fields: "
                f"{', '.join(missing_fields)}"
            )

        self._validate_name(payload["first_name"], "first_name")

        self._validate_name(payload["last_name"], "last_name")

        self._validate_email(payload["email"])

        self._validate_experience(payload["experience_years"])

    def _translate_repository_exception(
        self,
        exc: Exception,
    ) -> UserServiceError:
        """
        Translate a repository-layer exception into a service-layer
        exception.

        Repository exceptions (``UserRepositoryError``,
        ``UserNotFoundError``, ``InvalidUserError``,
        ``UserAlreadyExistsError``) must never leak past this service.

        Args:
            exc: The exception raised by ``UserRepository``.

        Returns:
            The corresponding ``UserServiceError`` subclass, ready to
            be raised by the caller.
        """

        if isinstance(exc, RepositoryUserAlreadyExistsError):
            return UserAlreadyExistsError(str(exc))

        if isinstance(exc, RepositoryUserNotFoundError):
            return CandidateNotFoundError(str(exc))

        if isinstance(exc, RepositoryInvalidUserError):
            return UserValidationError(str(exc))

        if isinstance(exc, RepositoryUserRepositoryError):
            return UserOperationError(str(exc))

        return UserServiceError(str(exc))

    def _resolve_service_exception(
        self,
        exc: Exception,
    ) -> UserServiceError:
        """
        Resolve any exception raised during a service operation into a
        ``UserServiceError``, translating repository exceptions and
        passing service exceptions through unchanged.
        """

        if isinstance(exc, UserServiceError):
            return exc

        return self._translate_repository_exception(exc)

    def _log_event(
        self,
        action: str,
        description: str,
        *,
        user_id: str | None = None,
        entity_id: str | None = None,
        status: str = "SUCCESS",
        severity: str = "INFO",
    ) -> None:
        """
        Send an audit event.

        Failures here should never stop candidate management flow.
        """

        try:

            self.audit_repository.log_event(
                {
                    "user_id": user_id,
                    "action": action,
                    "entity": "candidate",
                    "entity_id": entity_id,
                    "description": description,
                    "status": status,
                    "severity": severity,
                }
            )

        except Exception:

            self.logger.exception(
                "Failed to write audit log."
            )

    def _log_failure(
        self,
        exc: Exception,
        action: str,
        *,
        user_id: str | None = None,
        entity_id: str | None = None,
    ) -> UserServiceError:
        """
        Translate a failure, audit it, and return the translated
        exception for the caller to raise.

        Args:
            exc: The exception raised during the operation.
            action: Failure audit action constant.
            user_id: Optional actor identifier.
            entity_id: Optional candidate identifier.

        Returns:
            The translated ``UserServiceError`` to raise.
        """

        translated = self._resolve_service_exception(exc)

        self._log_event(
            action=action,
            description=str(translated),
            user_id=user_id,
            entity_id=entity_id,
            status="FAILURE",
            severity="WARNING",
        )

        return translated

    def _get_candidate(
        self,
        candidate_id: str,
    ) -> UserRecord:
        """
        Fetch a candidate record, translating repository exceptions
        and enforcing that the record belongs to the candidate role.

        Args:
            candidate_id: Unique candidate identifier.

        Returns:
            The matching candidate record.

        Raises:
            CandidateNotFoundError: If no candidate record exists.
            UserOperationError: If the lookup fails for another
                reason.
        """

        try:
            record = self.user_repository.get_user(candidate_id)
        except RepositoryUserNotFoundError as exc:
            raise CandidateNotFoundError(str(exc)) from exc
        except (
            RepositoryInvalidUserError,
            RepositoryUserRepositoryError,
        ) as exc:
            raise self._translate_repository_exception(exc) from exc

        if record.role != self.CANDIDATE_ROLE:
            raise CandidateNotFoundError(
                f"Candidate not found: {candidate_id}"
            )

        return record

    def _classify_experience_tier(
        self,
        experience_years: float,
    ) -> str:
        """
        Classify an experience value into a statistics tier.

        Args:
            experience_years: Candidate experience in years.

        Returns:
            One of ``"freshers"``, ``"juniors"``, ``"mid_level"``, or
            ``"senior"``.
        """

        if experience_years < self.FRESHER_MAX_YEARS:
            return "freshers"

        if experience_years < self.JUNIOR_MAX_YEARS:
            return "juniors"

        if experience_years < self.MID_MAX_YEARS:
            return "mid_level"

        return "senior"

    def _classify_candidate_tier(
        self,
        record: UserRecord,
    ) -> str:
        """
        Classify a candidate into a statistics tier.

        The stored ``candidate_level`` is used first via
        ``CANDIDATE_LEVEL_TIER_ALIASES``; when the value is not
        recognized, the tier is derived from ``experience_years``
        instead.

        Args:
            record: The candidate record to classify.

        Returns:
            One of ``"freshers"``, ``"juniors"``, ``"mid_level"``, or
            ``"senior"``.
        """

        normalized_level = record.candidate_level.strip().lower()

        tier = self.CANDIDATE_LEVEL_TIER_ALIASES.get(normalized_level)

        if tier is not None:
            return tier

        return self._classify_experience_tier(record.experience_years)

    def _compute_tier_counts(self) -> dict[str, int]:
        """
        Page through all candidate records and bucket them by
        statistics tier.

        Returns:
            A dictionary with counts for ``freshers``, ``juniors``,
            ``mid_level``, and ``senior``.
        """

        tier_counts = {
            "freshers": 0,
            "juniors": 0,
            "mid_level": 0,
            "senior": 0,
        }

        page = self.DEFAULT_PAGE

        while True:

            try:
                result = self.user_repository.list_users(
                    role=self.CANDIDATE_ROLE,
                    page=page,
                    page_size=self.STATISTICS_PAGE_SIZE,
                )
            except (
                RepositoryInvalidUserError,
                RepositoryUserRepositoryError,
            ) as exc:
                raise self._translate_repository_exception(exc) from exc

            for record in result["items"]:
                tier_counts[self._classify_candidate_tier(record)] += 1

            if not result["has_next"]:
                break

            page += 1

        return tier_counts

    def _record_matches_filters(
        self,
        record: UserRecord,
        *,
        candidate_level: str | None,
        status: str | None,
        experience_years: float | None,
    ) -> bool:
        """
        Check whether a candidate record matches the supplied filters
        that the repository could not apply directly.
        """

        if record.role != self.CANDIDATE_ROLE:
            return False

        if (
            candidate_level is not None
            and record.candidate_level.lower()
            != str(candidate_level).lower()
        ):
            return False

        if (
            status is not None
            and record.status.upper() != str(status).upper()
        ):
            return False

        if (
            experience_years is not None
            and record.experience_years != experience_years
        ):
            return False

        return True

    def _collect_filtered_candidates(
        self,
        *,
        search_term: str | None,
        candidate_level: str | None,
        status: str | None,
        experience_years: float | None,
    ) -> list[UserRecord]:
        """
        Scan repository pages and collect every candidate matching all
        supplied filters, bounded by ``MAX_SEARCH_SCAN_RECORDS``.

        This exists because ``UserRepository`` cannot combine
        free-text search with ``candidate_level``/``status``/
        ``experience_years`` filtering (or even a ``role`` filter) in
        a single query. Scanning and filtering here trades query
        efficiency for pagination metadata that is actually accurate;
        the alternative (filtering only the current page) makes
        ``total``/``total_pages`` misleading. Moving this filtering
        into a repository-level query would be the ideal fix but is
        out of this service's scope.

        Args:
            search_term: Optional free-text term (name or email).
            candidate_level: Optional candidate level filter.
            status: Optional status filter.
            experience_years: Optional exact experience match.

        Returns:
            All matching candidate records, in repository order, up
            to the scan cap.
        """

        collected: list[UserRecord] = []

        page = self.DEFAULT_PAGE

        while True:

            try:
                if search_term:
                    result = self.user_repository.search_users(
                        search_term=search_term,
                        page=page,
                        page_size=self.SEARCH_SCAN_PAGE_SIZE,
                    )
                else:
                    result = self.user_repository.list_users(
                        role=self.CANDIDATE_ROLE,
                        candidate_level=candidate_level,
                        status=status,
                        page=page,
                        page_size=self.SEARCH_SCAN_PAGE_SIZE,
                    )
            except (
                RepositoryInvalidUserError,
                RepositoryUserRepositoryError,
            ) as exc:
                raise self._translate_repository_exception(exc) from exc

            for record in result["items"]:
                if self._record_matches_filters(
                    record,
                    candidate_level=candidate_level,
                    status=status,
                    experience_years=experience_years,
                ):
                    collected.append(record)

            if not result["has_next"]:
                break

            if (
                len(collected) >= self.MAX_SEARCH_SCAN_RECORDS
                or page * self.SEARCH_SCAN_PAGE_SIZE
                >= self.MAX_SEARCH_SCAN_RECORDS
            ):
                self.logger.warning(
                    "Candidate search scan capped at %s records",
                    self.MAX_SEARCH_SCAN_RECORDS,
                )
                break

            page += 1

        return collected

    # =======================================================
    # Register Candidate
    # =======================================================

    def register_candidate(
        self,
        payload: Mapping[str, Any],
        actor: str | None = None,
    ) -> UserRecord:
        """
        Register a new candidate.

        The candidate role is always forced to ``CANDIDATE_ROLE``;
        callers can never register a user with an elevated role
        through this method.

        Args:
            payload: Candidate attributes. Must include ``email``,
                ``first_name``, ``last_name``, ``candidate_level``, and
                ``experience_years``. ``user_id`` is optional and will
                be generated when omitted.
            actor: Identifier of the user performing the registration.
                Defaults to the generated candidate id (self
                registration) when omitted.

        Returns:
            The newly created candidate record.

        Raises:
            UserValidationError: If the payload is invalid.
            UserAlreadyExistsError: If a duplicate email or candidate
                identifier exists.
            UserOperationError: If persistence fails.
        """

        if not isinstance(payload, Mapping):
            raise UserValidationError(
                "payload must be a mapping of candidate attributes"
            )

        self.logger.info("Start register_candidate")

        with self._lock:

            try:
                self._validate_candidate(payload)

                candidate_id = str(
                    payload.get("user_id")
                    or self._generate_candidate_id()
                )

                email = str(payload["email"]).strip()

                if self.candidate_exists(
                    user_id=candidate_id
                ) or self.candidate_exists(email=email):
                    raise UserAlreadyExistsError(
                        f"Candidate already exists: {email}"
                    )

                registration_actor = actor or candidate_id

                create_payload = dict(payload)
                create_payload["user_id"] = candidate_id
                create_payload["role"] = self.CANDIDATE_ROLE
                create_payload["created_by"] = registration_actor
                create_payload.setdefault(
                    "updated_by", registration_actor
                )

                record = self.user_repository.create_user(
                    create_payload
                )
            except (
                UserServiceError,
                RepositoryUserAlreadyExistsError,
                RepositoryInvalidUserError,
                RepositoryUserRepositoryError,
            ) as exc:
                raise self._log_failure(
                    exc,
                    self.ACTION_CANDIDATE_REGISTRATION_FAILED,
                ) from exc

            self._log_event(
                action=self.ACTION_CANDIDATE_REGISTERED,
                description=f"Candidate {record.user_id} registered.",
                user_id=registration_actor,
                entity_id=record.user_id,
            )

        self.logger.info(
            "End register_candidate: candidate=%s",
            record.user_id,
        )

        return record

    # =======================================================
    # Update Candidate
    # =======================================================

    def update_candidate(
        self,
        candidate_id: str,
        updates: Mapping[str, Any],
        actor: str,
    ) -> UserRecord:
        """
        Update allowed profile fields for an existing candidate.

        The candidate's ``role`` can never be modified through this
        method; role changes are out of scope for ``UserService``.

        Args:
            candidate_id: Unique candidate identifier.
            updates: Partial update payload.
            actor: Identifier of the user performing the update.

        Returns:
            The updated candidate record.

        Raises:
            CandidateNotFoundError: If the candidate does not exist.
            UserValidationError: If the update payload is invalid.
            UserAlreadyExistsError: If the update would create a
                duplicate email.
            UserOperationError: If persistence fails.
        """

        if not isinstance(updates, Mapping) or not updates:
            raise UserValidationError(
                "updates must be a non-empty mapping"
            )

        self.logger.info(
            "Start update_candidate: candidate=%s",
            candidate_id,
        )

        with self._lock:

            try:
                self._get_candidate(candidate_id)

                if "role" in updates:
                    raise UserValidationError(
                        "Candidate role cannot be modified through "
                        "update_candidate."
                    )

                if "first_name" in updates:
                    self._validate_name(
                        updates["first_name"], "first_name"
                    )

                if "last_name" in updates:
                    self._validate_name(
                        updates["last_name"], "last_name"
                    )

                if "email" in updates:
                    self._validate_email(updates["email"])

                if "experience_years" in updates:
                    self._validate_experience(
                        updates["experience_years"]
                    )

                record = self.user_repository.update_user(
                    user_id=candidate_id,
                    updates=dict(updates),
                    updated_by=actor,
                )
            except (
                UserServiceError,
                RepositoryUserNotFoundError,
                RepositoryUserAlreadyExistsError,
                RepositoryInvalidUserError,
                RepositoryUserRepositoryError,
            ) as exc:
                raise self._log_failure(
                    exc,
                    self.ACTION_CANDIDATE_UPDATE_FAILED,
                    user_id=actor,
                    entity_id=candidate_id,
                ) from exc

            self._log_event(
                action=self.ACTION_CANDIDATE_UPDATED,
                description=f"Candidate {record.user_id} updated.",
                user_id=actor,
                entity_id=record.user_id,
            )

        self.logger.info(
            "End update_candidate: candidate=%s",
            candidate_id,
        )

        return record

    # =======================================================
    # Delete Candidate
    # =======================================================

    def delete_candidate(
        self,
        candidate_id: str,
        actor: str,
    ) -> UserRecord:
        """
        Soft delete a candidate.

        ``UserRepository.delete_user`` already performs a soft delete
        (marks the record ``INACTIVE`` / ``is_active=False``); this
        method delegates to it directly rather than duplicating that
        logic.

        Args:
            candidate_id: Unique candidate identifier.
            actor: Identifier of the user performing the deletion.

        Returns:
            The updated, now-inactive candidate record.

        Raises:
            CandidateNotFoundError: If the candidate does not exist.
            UserOperationError: If persistence fails.
        """

        self.logger.info(
            "Start delete_candidate: candidate=%s",
            candidate_id,
        )

        with self._lock:

            try:
                self._get_candidate(candidate_id)

                record = self.user_repository.delete_user(
                    user_id=candidate_id,
                    updated_by=actor,
                )
            except (
                UserServiceError,
                RepositoryUserNotFoundError,
                RepositoryInvalidUserError,
                RepositoryUserRepositoryError,
            ) as exc:
                raise self._log_failure(
                    exc,
                    self.ACTION_CANDIDATE_DELETE_FAILED,
                    user_id=actor,
                    entity_id=candidate_id,
                ) from exc

            self.logger.warning(
                "Candidate deleted: %s",
                candidate_id,
            )

            self._log_event(
                action=self.ACTION_CANDIDATE_DELETED,
                description=f"Candidate {record.user_id} deleted.",
                user_id=actor,
                entity_id=record.user_id,
                severity="WARNING",
            )

        self.logger.info(
            "End delete_candidate: candidate=%s",
            candidate_id,
        )

        return record

    # =======================================================
    # Get Candidate
    # =======================================================

    def get_candidate(
        self,
        candidate_id: str,
    ) -> UserRecord:
        """
        Retrieve a candidate by identifier.

        Args:
            candidate_id: Unique candidate identifier.

        Returns:
            The matching candidate record.

        Raises:
            CandidateNotFoundError: If the candidate does not exist.
            UserOperationError: If the lookup fails for another
                reason.
        """

        self.logger.info(
            "Start get_candidate: candidate=%s",
            candidate_id,
        )

        with self._lock:

            try:
                record = self._get_candidate(candidate_id)
            except UserServiceError as exc:
                raise self._log_failure(
                    exc,
                    self.ACTION_CANDIDATE_VIEW_FAILED,
                    entity_id=candidate_id,
                ) from exc

            self._log_event(
                action=self.ACTION_CANDIDATE_VIEWED,
                description=f"Candidate {record.user_id} viewed.",
                entity_id=record.user_id,
            )

        self.logger.info(
            "End get_candidate: candidate=%s",
            candidate_id,
        )

        return record

    # =======================================================
    # Search Candidates
    # =======================================================

    def search_candidates(
        self,
        name: str | None = None,
        email: str | None = None,
        experience_years: float | None = None,
        candidate_level: str | None = None,
        status: str | None = None,
        page: int = DEFAULT_PAGE,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """
        Search candidate records.

        Whenever a filter is requested that ``UserRepository`` cannot
        apply directly in its query (free-text ``name``/``email``
        combined with ``candidate_level``/``status``, an exact
        ``experience_years`` match, or a plain free-text search that
        still needs the ``candidate`` role enforced), this method
        scans matching pages via ``_collect_filtered_candidates`` and
        re-paginates the fully filtered result itself, so the returned
        ``total``/``total_pages`` are always accurate for what's
        actually returned. When only ``candidate_level``/``status``
        are requested, ``UserRepository.list_users`` already filters
        and paginates precisely, so its result is returned unchanged.

        Args:
            name: Optional candidate name search term.
            email: Optional candidate email search term.
            experience_years: Optional exact experience match.
            candidate_level: Optional candidate level filter.
            status: Optional status filter.
            page: Page number starting at 1.
            page_size: Number of records per page.

        Returns:
            Pagination metadata and matching candidate records.

        Raises:
            UserValidationError: If pagination or ``experience_years``
                is invalid.
            UserOperationError: If the search fails.
        """

        self.logger.info(
            "Start search_candidates: page=%s",
            page,
        )

        with self._lock:

            try:
                self._validate_pagination(page, page_size)

                if experience_years is not None:
                    self._validate_experience(experience_years)

                search_term = name or email

                needs_full_scan = (
                    bool(search_term) or experience_years is not None
                )

                if needs_full_scan:

                    matching_candidates = (
                        self._collect_filtered_candidates(
                            search_term=search_term,
                            candidate_level=candidate_level,
                            status=status,
                            experience_years=experience_years,
                        )
                    )

                    total = len(matching_candidates)

                    start = (page - 1) * page_size

                    end = start + page_size

                    filtered_result = {
                        "items": matching_candidates[start:end],
                        "page": page,
                        "page_size": page_size,
                        "total": total,
                        "total_pages": (
                            ceil(total / page_size) if total else 0
                        ),
                        "has_next": end < total,
                        "has_previous": page > 1,
                    }

                else:

                    filtered_result = self.user_repository.list_users(
                        role=self.CANDIDATE_ROLE,
                        candidate_level=candidate_level,
                        status=status,
                        page=page,
                        page_size=page_size,
                    )
            except (
                UserServiceError,
                RepositoryInvalidUserError,
                RepositoryUserRepositoryError,
            ) as exc:
                raise self._log_failure(
                    exc,
                    self.ACTION_CANDIDATE_SEARCH_FAILED,
                ) from exc

            self._log_event(
                action=self.ACTION_CANDIDATE_SEARCHED,
                description=f"Candidate search executed (page={page}).",
            )

        self.logger.info(
            "End search_candidates: page=%s",
            page,
        )

        return filtered_result

    # =======================================================
    # Candidate Exists
    # =======================================================

    def candidate_exists(
        self,
        user_id: str | None = None,
        email: str | None = None,
    ) -> bool:
        """
        Check whether a candidate exists by identifier or email.

        Thin wrapper around ``UserRepository.user_exists``.

        Args:
            user_id: Optional unique candidate identifier.
            email: Optional email address.

        Returns:
            True if a matching record exists, else False.

        Raises:
            UserValidationError: If neither argument is provided.
            UserOperationError: If the check fails.
        """

        with self._lock:
            try:
                return self.user_repository.user_exists(
                    user_id=user_id,
                    email=email,
                )
            except (
                RepositoryInvalidUserError,
                RepositoryUserRepositoryError,
            ) as exc:
                raise self._translate_repository_exception(exc) from exc

    # =======================================================
    # Statistics
    # =======================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Return aggregated candidate statistics.

        Tiering prefers the stored ``candidate_level`` value (see
        ``CANDIDATE_LEVEL_TIER_ALIASES``) and falls back to
        experience-based thresholds for unrecognized level strings.

        Returns:
            A dictionary with ``total_candidates``, ``freshers``,
            ``juniors``, ``mid_level``, ``senior``, ``active``, and
            ``inactive``.

        Raises:
            UserOperationError: If the underlying queries fail.
        """

        self.logger.info("Start get_statistics")

        with self._lock:

            try:
                total_candidates = self.user_repository.count_users(
                    role=self.CANDIDATE_ROLE
                )
                active_candidates = self.user_repository.count_users(
                    role=self.CANDIDATE_ROLE,
                    active=True,
                )
                inactive_candidates = self.user_repository.count_users(
                    role=self.CANDIDATE_ROLE,
                    active=False,
                )

                tier_counts = self._compute_tier_counts()
            except (
                RepositoryInvalidUserError,
                RepositoryUserRepositoryError,
            ) as exc:
                raise self._translate_repository_exception(exc) from exc

            statistics_payload = {
                "total_candidates": total_candidates,
                "freshers": tier_counts["freshers"],
                "juniors": tier_counts["juniors"],
                "mid_level": tier_counts["mid_level"],
                "senior": tier_counts["senior"],
                "active": active_candidates,
                "inactive": inactive_candidates,
            }

        self.logger.info("End get_statistics")

        return statistics_payload