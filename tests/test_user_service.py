from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from repository.audit_repository import AuditRepository
from repository.user_repository import (
    InvalidUserError as RepositoryInvalidUserError,
    UserAlreadyExistsError as RepositoryUserAlreadyExistsError,
    UserNotFoundError as RepositoryUserNotFoundError,
    UserRecord,
    UserRepository,
    UserRepositoryError as RepositoryUserRepositoryError,
)

from services.user_service import (
    CandidateNotFoundError,
    UserAlreadyExistsError,
    UserOperationError,
    UserService,
    UserValidationError,
)


# ============================================================
# Helpers
# ============================================================

from datetime import datetime


def make_user_record(**overrides):
    now = datetime.utcnow()

    defaults = {
        "user_id": "candidate-001",
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "full_name": "John Doe",
        "department": "",
        "designation": "",
        "role": "candidate",
        "candidate_level": "Junior",
        "experience_years": 2.0,
        "status": "ACTIVE",
        "is_active": True,
        "last_login": None,
        "created_at": now,
        "updated_at": now,
        "created_by": "system",
        "updated_by": "system",
    }

    defaults.update(overrides)

    return UserRecord(**defaults)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_user_repository():
    return MagicMock(spec=UserRepository)


@pytest.fixture
def mock_audit_repository():
    return MagicMock(spec=AuditRepository)


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def service(
    mock_user_repository,
    mock_audit_repository,
    mock_logger,
):
    return UserService(
        user_repository=mock_user_repository,
        audit_repository=mock_audit_repository,
        logger=mock_logger,
    )


# ============================================================
# Validation Helpers
# ============================================================


def test_validate_name_success(service):
    service._validate_name("John", "first_name")


def test_validate_name_blank(service):
    with pytest.raises(UserValidationError):
        service._validate_name("", "first_name")


def test_validate_email_success(service):
    service._validate_email("john@example.com")


def test_validate_email_invalid(service):
    with pytest.raises(UserValidationError):
        service._validate_email("invalid-email")


def test_validate_experience_success(service):
    service._validate_experience(2)


def test_validate_experience_negative(service):
    with pytest.raises(UserValidationError):
        service._validate_experience(-1)


def test_validate_pagination_success(service):
    service._validate_pagination(1, 10)


@pytest.mark.parametrize(
    "page,page_size",
    [
        (0, 10),
        (-1, 10),
        (1, 0),
        (1, -1),
    ],
)
def test_validate_pagination_invalid(
    service,
    page,
    page_size,
):
    with pytest.raises(UserValidationError):
        service._validate_pagination(page, page_size)


# ============================================================
# Register Candidate
# ============================================================


def test_register_candidate_success(
    service,
    mock_user_repository,
    mock_audit_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    created = make_user_record()

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.return_value = created

    result = service.register_candidate(payload)

    assert result == created

    mock_user_repository.create_user.assert_called_once()

    mock_audit_repository.log_event.assert_called_once()


def test_register_candidate_actor_supplied(
    service,
    mock_user_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    created = make_user_record()

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.return_value = created

    service.register_candidate(
        payload,
        actor="admin",
    )

    args = mock_user_repository.create_user.call_args.args[0]

    assert args["created_by"] == "admin"
    assert args["updated_by"] == "admin"


def test_register_candidate_generates_candidate_role(
    service,
    mock_user_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    created = make_user_record()

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.return_value = created

    service.register_candidate(payload)

    args = mock_user_repository.create_user.call_args.args[0]

    assert args["role"] == service.CANDIDATE_ROLE


def test_register_candidate_duplicate_email(
    service,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    service.candidate_exists = MagicMock(side_effect=[False, True])

    with pytest.raises(UserAlreadyExistsError):
        service.register_candidate(payload)


def test_register_candidate_duplicate_candidate_id(
    service,
):
    payload = {
        "user_id": "candidate-001",
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    service.candidate_exists = MagicMock(side_effect=[True])

    with pytest.raises(UserAlreadyExistsError):
        service.register_candidate(payload)


def test_register_candidate_invalid_payload_type(
    service,
):
    with pytest.raises(UserValidationError):
        service.register_candidate("invalid")


def test_register_candidate_missing_required_fields(
    service,
):
    payload = {
        "email": "john@example.com",
    }

    with pytest.raises(UserValidationError):
        service.register_candidate(payload)


def test_register_candidate_invalid_email(
    service,
):
    payload = {
        "email": "bad-email",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    with pytest.raises(UserValidationError):
        service.register_candidate(payload)


def test_register_candidate_negative_experience(
    service,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": -5,
    }

    with pytest.raises(UserValidationError):
        service.register_candidate(payload)


def test_register_candidate_blank_first_name(
    service,
):
    payload = {
        "email": "john@example.com",
        "first_name": "",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    with pytest.raises(UserValidationError):
        service.register_candidate(payload)


def test_register_candidate_repository_duplicate(
    service,
    mock_user_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.side_effect = (
        RepositoryUserAlreadyExistsError("duplicate")
    )

    with pytest.raises(UserAlreadyExistsError):
        service.register_candidate(payload)


def test_register_candidate_repository_invalid(
    service,
    mock_user_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.side_effect = (
        RepositoryInvalidUserError("invalid")
    )

    with pytest.raises(UserValidationError):
        service.register_candidate(payload)


def test_register_candidate_repository_failure(
    service,
    mock_user_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.side_effect = (
        RepositoryUserRepositoryError("database down")
    )

    with pytest.raises(UserOperationError):
        service.register_candidate(payload)


def test_register_candidate_audit_failure_ignored(
    service,
    mock_user_repository,
    mock_audit_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    created = make_user_record()

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.return_value = created

    mock_audit_repository.log_event.side_effect = Exception(
        "audit unavailable"
    )

    result = service.register_candidate(payload)

    assert result == created


def test_register_candidate_generated_id(
    service,
    mock_user_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    service.candidate_exists = MagicMock(return_value=False)

    service._generate_candidate_id = MagicMock(
        return_value="generated-id"
    )

    mock_user_repository.create_user.return_value = (
        make_user_record(user_id="generated-id")
    )

    service.register_candidate(payload)

    create_payload = (
        mock_user_repository.create_user.call_args.args[0]
    )

    assert create_payload["user_id"] == "generated-id"


def test_register_candidate_existing_updated_by_not_overwritten(
    service,
    mock_user_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "candidate_level": "Junior",
        "experience_years": 2,
        "updated_by": "already-set",
    }

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.return_value = make_user_record()

    service.register_candidate(payload)

    create_payload = (
        mock_user_repository.create_user.call_args.args[0]
    )

    assert create_payload["updated_by"] == "already-set"


def test_register_candidate_forces_candidate_role(
    service,
    mock_user_repository,
):
    payload = {
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "role": "admin",
        "candidate_level": "Junior",
        "experience_years": 2,
    }

    service.candidate_exists = MagicMock(return_value=False)

    mock_user_repository.create_user.return_value = make_user_record()

    service.register_candidate(payload)

    create_payload = (
        mock_user_repository.create_user.call_args.args[0]
    )

    assert create_payload["role"] == service.CANDIDATE_ROLE


# ============================================================
# Update Candidate
# ============================================================


def test_update_candidate_success(
    service,
    mock_user_repository,
    mock_audit_repository,
):
    existing = make_user_record()

    updated = make_user_record(first_name="Jane")

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    result = service.update_candidate(
        candidate_id=existing.user_id,
        updates={"first_name": "Jane"},
        actor="admin",
    )

    assert result == updated

    mock_user_repository.update_user.assert_called_once_with(
        user_id=existing.user_id,
        updates={"first_name": "Jane"},
        updated_by="admin",
    )

    mock_audit_repository.log_event.assert_called()


def test_update_candidate_multiple_fields(
    service,
    mock_user_repository,
):
    existing = make_user_record()

    updated = make_user_record(
        first_name="Jane",
        last_name="Smith",
        experience_years=5,
    )

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    result = service.update_candidate(
        candidate_id=existing.user_id,
        updates={
            "first_name": "Jane",
            "last_name": "Smith",
            "experience_years": 5,
        },
        actor="admin",
    )

    assert result.first_name == "Jane"
    assert result.last_name == "Smith"
    assert result.experience_years == 5


def test_update_candidate_email(
    service,
    mock_user_repository,
):
    existing = make_user_record()

    updated = make_user_record(email="new@example.com")

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    result = service.update_candidate(
        existing.user_id,
        {"email": "new@example.com"},
        "admin",
    )

    assert result.email == "new@example.com"


def test_update_candidate_experience(
    service,
    mock_user_repository,
):
    existing = make_user_record()

    updated = make_user_record(experience_years=6)

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    result = service.update_candidate(
        existing.user_id,
        {"experience_years": 6},
        "admin",
    )

    assert result.experience_years == 6


def test_update_candidate_empty_updates(
    service,
):
    with pytest.raises(UserValidationError):
        service.update_candidate(
            "candidate-001",
            {},
            "admin",
        )


def test_update_candidate_none_updates(
    service,
):
    with pytest.raises(UserValidationError):
        service.update_candidate(
            "candidate-001",
            None,
            "admin",
        )


def test_update_candidate_invalid_email(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = make_user_record()

    with pytest.raises(UserValidationError):
        service.update_candidate(
            "candidate-001",
            {"email": "bad-email"},
            "admin",
        )


def test_update_candidate_negative_experience(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = make_user_record()

    with pytest.raises(UserValidationError):
        service.update_candidate(
            "candidate-001",
            {"experience_years": -5},
            "admin",
        )


def test_update_candidate_blank_first_name(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = make_user_record()

    with pytest.raises(UserValidationError):
        service.update_candidate(
            "candidate-001",
            {"first_name": ""},
            "admin",
        )


def test_update_candidate_blank_last_name(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = make_user_record()

    with pytest.raises(UserValidationError):
        service.update_candidate(
            "candidate-001",
            {"last_name": ""},
            "admin",
        )


def test_update_candidate_role_not_allowed(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = make_user_record()

    with pytest.raises(UserValidationError):
        service.update_candidate(
            "candidate-001",
            {"role": "admin"},
            "admin",
        )

def test_update_candidate_candidate_not_found(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.side_effect = (
        RepositoryUserNotFoundError("not found")
    )

    with pytest.raises(CandidateNotFoundError):
        service.update_candidate(
            "candidate-001",
            {"first_name": "Jane"},
            "admin",
        )


def test_update_candidate_non_candidate_record(
    service,
    mock_user_repository,
):
    record = make_user_record(role="admin")

    mock_user_repository.get_user.return_value = record

    with pytest.raises(CandidateNotFoundError):
        service.update_candidate(
            record.user_id,
            {"first_name": "Jane"},
            "admin",
        )


def test_update_candidate_duplicate_email(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = make_user_record()

    mock_user_repository.update_user.side_effect = (
        RepositoryUserAlreadyExistsError("duplicate")
    )

    with pytest.raises(UserAlreadyExistsError):
        service.update_candidate(
            "candidate-001",
            {"email": "duplicate@example.com"},
            "admin",
        )


def test_update_candidate_repository_invalid(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = make_user_record()

    mock_user_repository.update_user.side_effect = (
        RepositoryInvalidUserError("invalid")
    )

    with pytest.raises(UserValidationError):
        service.update_candidate(
            "candidate-001",
            {"first_name": "Jane"},
            "admin",
        )


def test_update_candidate_repository_failure(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = make_user_record()

    mock_user_repository.update_user.side_effect = (
        RepositoryUserRepositoryError("database")
    )

    with pytest.raises(UserOperationError):
        service.update_candidate(
            "candidate-001",
            {"first_name": "Jane"},
            "admin",
        )


def test_update_candidate_audit_failure_ignored(
    service,
    mock_user_repository,
    mock_audit_repository,
):
    existing = make_user_record()
    updated = make_user_record(first_name="Jane")

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    mock_audit_repository.log_event.side_effect = Exception(
        "audit down"
    )

    result = service.update_candidate(
        existing.user_id,
        {"first_name": "Jane"},
        "admin",
    )

    assert result == updated


def test_update_candidate_preserves_role(
    service,
    mock_user_repository,
):
    existing = make_user_record(role="candidate")

    updated = make_user_record(role="candidate")

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    result = service.update_candidate(
        existing.user_id,
        {"first_name": "Jane"},
        "admin",
    )

    assert result.role == "candidate"


def test_update_candidate_single_field(
    service,
    mock_user_repository,
):
    existing = make_user_record()

    updated = make_user_record(last_name="Smith")

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    result = service.update_candidate(
        existing.user_id,
        {"last_name": "Smith"},
        "admin",
    )

    assert result.last_name == "Smith"


def test_update_candidate_updated_by_propagated(
    service,
    mock_user_repository,
):
    existing = make_user_record()

    updated = make_user_record()

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    service.update_candidate(
        existing.user_id,
        {"first_name": "Jane"},
        "reviewer-123",
    )

    mock_user_repository.update_user.assert_called_once_with(
        user_id=existing.user_id,
        updates={"first_name": "Jane"},
        updated_by="reviewer-123",
    )


def test_update_candidate_logger_called(
    service,
    mock_user_repository,
    mock_logger,
):
    existing = make_user_record()

    updated = make_user_record(first_name="Jane")

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.update_user.return_value = updated

    service.update_candidate(
        existing.user_id,
        {"first_name": "Jane"},
        "admin",
    )

    assert mock_logger.info.call_count >= 2

# ============================================================
# Delete Candidate
# ============================================================

def test_delete_candidate_success(
    service,
    mock_user_repository,
    mock_audit_repository,
):
    existing = make_user_record()

    deleted = make_user_record(
        status="INACTIVE",
        is_active=False,
    )

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.delete_user.return_value = deleted

    result = service.delete_candidate(
        existing.user_id,
        actor="admin",
    )

    assert result == deleted

    mock_user_repository.delete_user.assert_called_once_with(
        user_id=existing.user_id,
        updated_by="admin",
    )

    mock_audit_repository.log_event.assert_called_once()


def test_delete_candidate_candidate_not_found(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.side_effect = (
        RepositoryUserNotFoundError("not found")
    )

    with pytest.raises(CandidateNotFoundError):
        service.delete_candidate(
            "candidate-001",
            "admin",
        )


def test_delete_candidate_non_candidate(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = (
        make_user_record(role="admin")
    )

    with pytest.raises(CandidateNotFoundError):
        service.delete_candidate(
            "candidate-001",
            "admin",
        )


def test_delete_candidate_repository_failure(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = (
        make_user_record()
    )

    mock_user_repository.delete_user.side_effect = (
        RepositoryUserRepositoryError("database")
    )

    with pytest.raises(UserOperationError):
        service.delete_candidate(
            "candidate-001",
            "admin",
        )


def test_delete_candidate_audit_failure_ignored(
    service,
    mock_user_repository,
    mock_audit_repository,
):
    existing = make_user_record()

    deleted = make_user_record(
        status="INACTIVE",
        is_active=False,
    )

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.delete_user.return_value = deleted

    mock_audit_repository.log_event.side_effect = Exception(
        "audit down"
    )

    result = service.delete_candidate(
        existing.user_id,
        "admin",
    )

    assert result == deleted


def test_delete_candidate_actor_propagated(
    service,
    mock_user_repository,
):
    existing = make_user_record()

    deleted = make_user_record(
        status="INACTIVE",
        is_active=False,
    )

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.delete_user.return_value = deleted

    service.delete_candidate(
        existing.user_id,
        "reviewer",
    )

    mock_user_repository.delete_user.assert_called_once_with(
        user_id=existing.user_id,
        updated_by="reviewer",
    )


def test_delete_candidate_logger_called(
    service,
    mock_user_repository,
    mock_logger,
):
    existing = make_user_record()

    deleted = make_user_record(
        status="INACTIVE",
        is_active=False,
    )

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.delete_user.return_value = deleted

    service.delete_candidate(
        existing.user_id,
        "admin",
    )

    assert mock_logger.info.call_count >= 1
    assert mock_logger.warning.call_count >= 1


def test_delete_candidate_returns_deleted_record(
    service,
    mock_user_repository,
):
    existing = make_user_record()

    deleted = make_user_record(
        status="INACTIVE",
        is_active=False,
    )

    mock_user_repository.get_user.return_value = existing
    mock_user_repository.delete_user.return_value = deleted

    result = service.delete_candidate(
        existing.user_id,
        "admin",
    )

    assert result == deleted
    assert result.status == "INACTIVE"
    assert result.is_active is False

# ============================================================
# Get Candidate
# ============================================================

def test_get_candidate_success(
    service,
    mock_user_repository,
):
    candidate = make_user_record()

    mock_user_repository.get_user.return_value = candidate

    result = service.get_candidate(candidate.user_id)

    assert result == candidate

    mock_user_repository.get_user.assert_called_once_with(
        candidate.user_id
    )


def test_get_candidate_candidate_not_found(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.side_effect = (
        RepositoryUserNotFoundError("not found")
    )

    with pytest.raises(CandidateNotFoundError):
        service.get_candidate("candidate-001")


def test_get_candidate_non_candidate(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.return_value = (
        make_user_record(role="admin")
    )

    with pytest.raises(CandidateNotFoundError):
        service.get_candidate("candidate-001")


def test_get_candidate_repository_failure(
    service,
    mock_user_repository,
):
    mock_user_repository.get_user.side_effect = (
        RepositoryUserRepositoryError("database")
    )

    with pytest.raises(UserOperationError):
        service.get_candidate("candidate-001")


def test_get_candidate_logger_called(
    service,
    mock_user_repository,
    mock_logger,
):
    candidate = make_user_record()

    mock_user_repository.get_user.return_value = candidate

    service.get_candidate(candidate.user_id)

    assert mock_logger.info.call_count >= 1


def test_get_candidate_returns_same_record(
    service,
    mock_user_repository,
):
    candidate = make_user_record(
        first_name="John",
        last_name="Doe",
    )

    mock_user_repository.get_user.return_value = candidate

    result = service.get_candidate(candidate.user_id)

    assert result.user_id == candidate.user_id
    assert result.email == candidate.email
    assert result.first_name == "John"
    assert result.last_name == "Doe"


def test_get_candidate_preserves_candidate_level(
    service,
    mock_user_repository,
):
    candidate = make_user_record(
        candidate_level="Senior"
    )

    mock_user_repository.get_user.return_value = candidate

    result = service.get_candidate(candidate.user_id)

    assert result.candidate_level == "Senior"


def test_get_candidate_preserves_experience(
    service,
    mock_user_repository,
):
    candidate = make_user_record(
        experience_years=8.5
    )

    mock_user_repository.get_user.return_value = candidate

    result = service.get_candidate(candidate.user_id)

    assert result.experience_years == 8.5


def test_get_candidate_inactive_candidate(
    service,
    mock_user_repository,
):
    candidate = make_user_record(
        status="INACTIVE",
        is_active=False,
    )

    mock_user_repository.get_user.return_value = candidate

    result = service.get_candidate(candidate.user_id)

    assert result.status == "INACTIVE"
    assert result.is_active is False


def test_get_candidate_calls_repository_once(
    service,
    mock_user_repository,
):
    candidate = make_user_record()

    mock_user_repository.get_user.return_value = candidate

    service.get_candidate(candidate.user_id)

    assert mock_user_repository.get_user.call_count == 1

# ============================================================
# Candidate Exists
# ============================================================

def test_candidate_exists_by_user_id(
    service,
    mock_user_repository,
):
    mock_user_repository.user_exists.return_value = True

    result = service.candidate_exists(
        user_id="candidate-001"
    )

    assert result is True

    mock_user_repository.user_exists.assert_called_once_with(
        user_id="candidate-001",
        email=None,
    )


def test_candidate_exists_by_email(
    service,
    mock_user_repository,
):
    mock_user_repository.user_exists.return_value = True

    result = service.candidate_exists(
        email="john@test.com"
    )

    assert result is True

    mock_user_repository.user_exists.assert_called_once_with(
        user_id=None,
        email="john@test.com",
    )


def test_candidate_exists_returns_false(
    service,
    mock_user_repository,
):
    mock_user_repository.user_exists.return_value = False

    result = service.candidate_exists(
        user_id="candidate-001"
    )

    assert result is False


def test_candidate_exists_repository_invalid(
    service,
    mock_user_repository,
):
    mock_user_repository.user_exists.side_effect = (
        RepositoryInvalidUserError("invalid")
    )

    with pytest.raises(UserValidationError):
        service.candidate_exists(
            user_id="candidate-001"
        )


def test_candidate_exists_repository_failure(
    service,
    mock_user_repository,
):
    mock_user_repository.user_exists.side_effect = (
        RepositoryUserRepositoryError("database")
    )

    with pytest.raises(UserOperationError):
        service.candidate_exists(
            user_id="candidate-001"
        )


def test_candidate_exists_calls_repository_once(
    service,
    mock_user_repository,
):
    mock_user_repository.user_exists.return_value = True

    service.candidate_exists(
        user_id="candidate-001"
    )

    assert mock_user_repository.user_exists.call_count == 1

# ============================================================
# Get Statistics
# ============================================================

def test_get_statistics_success(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = [
        100,   # total
        80,    # active
        20,    # inactive
    ]

    service._compute_tier_counts = MagicMock(
        return_value={
            "freshers": 10,
            "juniors": 30,
            "mid_level": 40,
            "senior": 20,
        }
    )

    result = service.get_statistics()

    assert result == {
        "total_candidates": 100,
        "freshers": 10,
        "juniors": 30,
        "mid_level": 40,
        "senior": 20,
        "active": 80,
        "inactive": 20,
    }


def test_get_statistics_total_count(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = [
        5,
        4,
        1,
    ]

    service._compute_tier_counts = MagicMock(
        return_value={
            "freshers": 1,
            "juniors": 2,
            "mid_level": 1,
            "senior": 1,
        }
    )

    result = service.get_statistics()

    assert result["total_candidates"] == 5


def test_get_statistics_active_count(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = [
        10,
        8,
        2,
    ]

    service._compute_tier_counts = MagicMock(
        return_value={
            "freshers": 2,
            "juniors": 3,
            "mid_level": 3,
            "senior": 2,
        }
    )

    result = service.get_statistics()

    assert result["active"] == 8


def test_get_statistics_inactive_count(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = [
        10,
        8,
        2,
    ]

    service._compute_tier_counts = MagicMock(
        return_value={
            "freshers": 2,
            "juniors": 3,
            "mid_level": 3,
            "senior": 2,
        }
    )

    result = service.get_statistics()

    assert result["inactive"] == 2


def test_get_statistics_compute_tiers_called(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = [
        1,
        1,
        0,
    ]

    service._compute_tier_counts = MagicMock(
        return_value={
            "freshers": 1,
            "juniors": 0,
            "mid_level": 0,
            "senior": 0,
        }
    )

    service.get_statistics()

    service._compute_tier_counts.assert_called_once()


def test_get_statistics_repository_failure(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = (
        RepositoryUserRepositoryError("database")
    )

    with pytest.raises(UserOperationError):
        service.get_statistics()


def test_get_statistics_invalid_repository(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = (
        RepositoryInvalidUserError("invalid")
    )

    with pytest.raises(UserValidationError):
        service.get_statistics()


def test_get_statistics_logger_called(
    service,
    mock_user_repository,
    mock_logger,
):
    mock_user_repository.count_users.side_effect = [
        1,
        1,
        0,
    ]

    service._compute_tier_counts = MagicMock(
        return_value={
            "freshers": 1,
            "juniors": 0,
            "mid_level": 0,
            "senior": 0,
        }
    )

    service.get_statistics()

    assert mock_logger.info.call_count >= 2


def test_get_statistics_count_users_called_three_times(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = [
        10,
        8,
        2,
    ]

    service._compute_tier_counts = MagicMock(
        return_value={
            "freshers": 2,
            "juniors": 3,
            "mid_level": 3,
            "senior": 2,
        }
    )

    service.get_statistics()

    assert mock_user_repository.count_users.call_count == 3


def test_get_statistics_zero_candidates(
    service,
    mock_user_repository,
):
    mock_user_repository.count_users.side_effect = [
        0,
        0,
        0,
    ]

    service._compute_tier_counts = MagicMock(
        return_value={
            "freshers": 0,
            "juniors": 0,
            "mid_level": 0,
            "senior": 0,
        }
    )

    result = service.get_statistics()

    assert result == {
        "total_candidates": 0,
        "freshers": 0,
        "juniors": 0,
        "mid_level": 0,
        "senior": 0,
        "active": 0,
        "inactive": 0,
    }