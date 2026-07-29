from __future__ import annotations

import logging

import pytest

from utils.retry import RetryConfig, RetryContext, RetryError, execute_with_retry, retry


class TransientError(Exception):
    """Raised for retryable failures during tests."""


class PermanentError(Exception):
    """Raised for non-retryable failures during tests."""


def test_execute_with_retry_returns_value_after_transient_failures() -> None:
    attempts: list[int] = []
    sleep_delays: list[float] = []

    def flaky_operation() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise TransientError("temporary failure")
        return "ok"

    result = execute_with_retry(
        flaky_operation,
        config=RetryConfig(
            max_attempts=4,
            initial_delay=0.1,
            max_delay=1.0,
            jitter_ratio=0.0,
            retryable_exceptions=(TransientError,),
            operation_name="flaky_operation",
        ),
        sleep_func=sleep_delays.append,
    )

    assert result == "ok"
    assert len(attempts) == 3
    assert sleep_delays == [0.1, 0.2]


def test_execute_with_retry_raises_original_exception_for_non_retryable_error() -> None:
    def failing_operation() -> None:
        raise PermanentError("do not retry")

    with pytest.raises(PermanentError):
        execute_with_retry(
            failing_operation,
            config=RetryConfig(
                max_attempts=3,
                retryable_exceptions=(TransientError,),
                operation_name="permanent_operation",
            ),
            sleep_func=lambda _: None,
        )


def test_execute_with_retry_raises_retry_error_after_retry_exhaustion() -> None:
    attempts: list[int] = []

    def always_fail() -> None:
        attempts.append(1)
        raise TransientError("still failing")

    with pytest.raises(RetryError) as exc_info:
        execute_with_retry(
            always_fail,
            config=RetryConfig(
                max_attempts=3,
                initial_delay=0.1,
                max_delay=1.0,
                jitter_ratio=0.0,
                retryable_exceptions=(TransientError,),
                operation_name="always_fail",
            ),
            sleep_func=lambda _: None,
        )

    assert len(attempts) == 3
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, TransientError)


def test_retry_decorator_applies_policy() -> None:
    attempts: list[int] = []

    @retry(
        RetryConfig(
            max_attempts=2,
            initial_delay=0.0,
            max_delay=0.0,
            jitter_ratio=0.0,
            retryable_exceptions=(TransientError,),
            operation_name="decorated_operation",
        ),
        sleep_func=lambda _: None,
    )
    def decorated_operation() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise TransientError("temporary issue")
        return "done"

    assert decorated_operation() == "done"
    assert len(attempts) == 2


def test_execute_with_retry_invokes_callback_and_logs_retry(caplog: pytest.LogCaptureFixture) -> None:
    contexts: list[RetryContext] = []
    logger = logging.getLogger("retry-test")

    def flaky_operation() -> str:
        if not contexts:
            raise TransientError("retry me")
        return "ready"

    with caplog.at_level(logging.WARNING):
        result = execute_with_retry(
            flaky_operation,
            config=RetryConfig(
                max_attempts=2,
                initial_delay=0.25,
                max_delay=1.0,
                jitter_ratio=0.0,
                retryable_exceptions=(TransientError,),
                operation_name="callback_operation",
            ),
            logger=logger,
            sleep_func=lambda _: None,
            on_retry=contexts.append,
        )

    assert result == "ready"
    assert len(contexts) == 1
    assert contexts[0].operation_name == "callback_operation"
    assert "Retrying callback_operation after attempt 1/2" in caplog.text
