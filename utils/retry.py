from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class RetryError(Exception):
    """Raised when an operation exceeds the configured retry attempts."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_exception: BaseException,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


@dataclass(frozen=True)
class RetryContext:
    """Carries runtime information about a retry attempt."""

    operation_name: str
    attempt_number: int
    max_attempts: int
    delay_seconds: float
    elapsed_seconds: float
    exception: BaseException


@dataclass(frozen=True)
class RetryConfig:
    """Defines retry behavior for transient failures."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)
    retryable_predicate: Callable[[BaseException], bool] | None = None
    operation_name: str = "operation"

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be greater than or equal to 0")
        if self.max_delay < 0:
            raise ValueError("max_delay must be greater than or equal to 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")


@dataclass
class RetryExecutor(Generic[T]):
    """Executes callables with exponential backoff and structured logging."""

    config: RetryConfig
    logger: logging.Logger | None = None
    sleep_func: Callable[[float], None] = time.sleep
    time_func: Callable[[], float] = time.monotonic
    random_func: Callable[[float, float], float] = field(
        default_factory=lambda: random.uniform
    )
    on_retry: Callable[[RetryContext], None] | None = None

    def execute(
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute a callable according to the configured retry policy."""
        active_logger = self.logger or logging.getLogger(__name__)
        start_time = self.time_func()
        last_exception: BaseException | None = None

        for attempt_number in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                last_exception = exc
                should_retry = self._should_retry(exc)
                is_final_attempt = attempt_number >= self.config.max_attempts

                if not should_retry or is_final_attempt:
                    if should_retry and is_final_attempt:
                        raise RetryError(
                            (
                                f"{self.config.operation_name} failed after "
                                f"{attempt_number} attempt(s)"
                            ),
                            attempts=attempt_number,
                            last_exception=exc,
                        ) from exc
                    raise

                delay_seconds = self._compute_delay(attempt_number)
                context = RetryContext(
                    operation_name=self.config.operation_name,
                    attempt_number=attempt_number,
                    max_attempts=self.config.max_attempts,
                    delay_seconds=delay_seconds,
                    elapsed_seconds=self.time_func() - start_time,
                    exception=exc,
                )
                self._log_retry(active_logger, context)
                if self.on_retry is not None:
                    self.on_retry(context)
                self.sleep_func(delay_seconds)

        raise RetryError(
            f"{self.config.operation_name} failed after retry exhaustion",
            attempts=self.config.max_attempts,
            last_exception=last_exception or RuntimeError("unknown retry failure"),
        )

    def _should_retry(self, exc: BaseException) -> bool:
        if not isinstance(exc, self.config.retryable_exceptions):
            return False
        if self.config.retryable_predicate is None:
            return True
        return self.config.retryable_predicate(exc)

    def _compute_delay(self, attempt_number: int) -> float:
        base_delay = min(
            self.config.initial_delay
            * (self.config.backoff_multiplier ** (attempt_number - 1)),
            self.config.max_delay,
        )
        if base_delay == 0 or self.config.jitter_ratio == 0:
            return round(base_delay, 4)

        jitter_window = base_delay * self.config.jitter_ratio
        lower_bound = max(0.0, base_delay - jitter_window)
        upper_bound = min(self.config.max_delay, base_delay + jitter_window)
        return round(self.random_func(lower_bound, upper_bound), 4)

    def _log_retry(self, logger: logging.Logger, context: RetryContext) -> None:
        logger.warning(
            "Retrying %s after attempt %d/%d due to %s; next delay %.4fs; elapsed %.4fs",
            context.operation_name,
            context.attempt_number,
            context.max_attempts,
            context.exception.__class__.__name__,
            context.delay_seconds,
            context.elapsed_seconds,
            extra={
                "operation_name": context.operation_name,
                "attempt_number": context.attempt_number,
                "max_attempts": context.max_attempts,
                "delay_seconds": context.delay_seconds,
                "elapsed_seconds": context.elapsed_seconds,
                "exception_type": context.exception.__class__.__name__,
            },
        )


def execute_with_retry(
    func: Callable[P, T],
    *args: P.args,
    config: RetryConfig,
    logger: logging.Logger | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
    time_func: Callable[[], float] = time.monotonic,
    random_func: Callable[[float, float], float] | None = None,
    on_retry: Callable[[RetryContext], None] | None = None,
    **kwargs: P.kwargs,
) -> T:
    """Execute a callable with a reusable retry policy."""
    executor = RetryExecutor[T](
        config=config,
        logger=logger,
        sleep_func=sleep_func,
        time_func=time_func,
        random_func=random_func or random.uniform,
        on_retry=on_retry,
    )
    return executor.execute(func, *args, **kwargs)


def retry(
    config: RetryConfig,
    *,
    logger: logging.Logger | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
    time_func: Callable[[], float] = time.monotonic,
    random_func: Callable[[float, float], float] | None = None,
    on_retry: Callable[[RetryContext], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for applying retry behavior to a callable."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return execute_with_retry(
                func,
                *args,
                config=config,
                logger=logger,
                sleep_func=sleep_func,
                time_func=time_func,
                random_func=random_func,
                on_retry=on_retry,
                **kwargs,
            )

        wrapper.__name__ = getattr(func, "__name__", "wrapped_retry")
        wrapper.__doc__ = getattr(func, "__doc__", None)
        wrapper.__module__ = getattr(func, "__module__", __name__)
        return wrapper

    return decorator
