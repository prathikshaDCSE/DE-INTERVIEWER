"""
exceptions/gemini_exceptions.py

Exception hierarchy for GeminiService.

All exceptions raised by GeminiService are subclasses of
``GeminiServiceError``. Callers should catch the most specific
subclass they intend to handle and let the rest propagate.

Hierarchy
---------
GeminiServiceError                   ← base; catch-all for callers
├── GeminiConfigurationError         ← bad API key, missing model, etc.
├── GeminiTimeoutError               ← request exceeded configured timeout
├── GeminiRateLimitError             ← HTTP 429 / quota exhausted
├── GeminiSafetyError                ← response blocked by safety filters
└── GeminiResponseError              ← bad / unparseable / unexpected response
    └── GeminiJsonParseError         ← generate_json() could not parse JSON

Design notes
------------
* Every exception carries a human-readable ``message`` and an optional
  ``request_id`` so that log correlation is possible at every layer of
  the call stack without requiring callers to hold extra state.
* ``GeminiSafetyError`` carries ``blocked_categories`` so upstream
  services can decide whether to retry with a softer prompt or abort.
* ``GeminiResponseError`` carries ``raw_text`` so callers (and tests)
  can inspect exactly what the SDK returned when something went wrong.
* ``GeminiJsonParseError`` carries both ``raw_text`` (what was
  returned) and ``parse_error`` (the original ``json.JSONDecodeError``)
  so the root cause is never swallowed.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# Base
# ============================================================


class GeminiServiceError(Exception):
    """
    Base exception for all errors raised by ``GeminiService``.

    Every more-specific exception in this module is a subclass of this
    one. Callers that want a single broad catch point should catch
    ``GeminiServiceError``; callers that need fine-grained handling
    should catch the appropriate subclass.

    Attributes:
        message: Human-readable description of the error.
        request_id: Optional correlation ID from the originating
            ``GeminiService`` call. ``None`` when the error occurs
            before a request ID is assigned (e.g. during client
            initialisation or prompt validation).
    """

    def __init__(
        self,
        message: str,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.request_id: str | None = request_id

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Configuration
# ============================================================


class GeminiConfigurationError(GeminiServiceError):
    """
    Raised when ``GeminiService`` cannot be initialised or cannot
    execute a request because of a configuration problem.

    Common causes
    -------------
    * ``GOOGLE_API_KEY`` is missing or empty.
    * ``GEMINI_MODEL`` is not set.
    * An invalid combination of generation parameters was supplied.

    This exception is **not retryable**. The misconfiguration must
    be corrected before any request can succeed.

    Attributes:
        config_key: The name of the configuration field that is
            missing or invalid, when known.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.config_key: str | None = config_key

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"config_key={self.config_key!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Timeout
# ============================================================


class GeminiTimeoutError(GeminiServiceError):
    """
    Raised when a request to the Gemini API exceeds the configured
    timeout.

    Attributes:
        timeout_seconds: The timeout value (in seconds) that was
            exceeded.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: int | float | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.timeout_seconds: int | float | None = timeout_seconds

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Rate limit
# ============================================================


class GeminiRateLimitError(GeminiServiceError):
    """
    Raised when the Gemini API returns an HTTP 429 (Too Many
    Requests) or signals that the project quota has been exhausted.

    ``GeminiService`` retries on rate-limit responses up to
    ``GEMINI_MAX_RETRIES`` times with exponential back-off before
    raising this exception. By the time it reaches the caller, the
    retry budget is spent.

    Attributes:
        retry_count: Number of attempts made before giving up.
    """

    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.retry_count: int = retry_count

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"retry_count={self.retry_count!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Safety
# ============================================================


class GeminiSafetyError(GeminiServiceError):
    """
    Raised when the Gemini API blocks a response because it violates
    one or more of its safety policies.

    This exception is **not retryable** without modifying the prompt.
    Callers must not silently swallow it — a blocked response means
    no usable text was returned and the downstream service must decide
    how to handle the gap.

    Attributes:
        blocked_categories: List of safety category names that
            triggered the block (e.g. ``["HARM_CATEGORY_HATE_SPEECH"]``).
            Empty list when the SDK does not expose category detail.
    """

    def __init__(
        self,
        message: str,
        blocked_categories: list[str] | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.blocked_categories: list[str] = blocked_categories or []

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"blocked_categories={self.blocked_categories!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Response errors
# ============================================================


class GeminiResponseError(GeminiServiceError):
    """
    Raised when the Gemini API returns a response that ``GeminiService``
    cannot parse or that does not match the expected structure.

    Common causes
    -------------
    * The response object is missing ``candidates`` entirely.
    * ``candidates`` is present but empty.
    * The ``content`` block or ``parts`` list is absent or empty.
    * ``finish_reason`` signals an unexpected terminal condition that
      is not a safety block (e.g. ``RECITATION``, ``OTHER``).

    Attributes:
        raw_text: The raw text extracted from the response before
            the parse attempt, or ``None`` if no text could be
            extracted at all. Useful for debugging without needing
            to reconstruct the SDK object.
    """

    def __init__(
        self,
        message: str,
        raw_text: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.raw_text: str | None = raw_text

    def __repr__(self) -> str:  # pragma: no cover
        preview = (
            self.raw_text[:120] + "…"
            if self.raw_text and len(self.raw_text) > 120
            else self.raw_text
        )
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"raw_text={preview!r}, "
            f"request_id={self.request_id!r})"
        )


class GeminiJsonParseError(GeminiResponseError):
    """
    Raised by ``GeminiService.generate_json()`` when the text returned
    by Gemini cannot be parsed as valid JSON, or when the top-level
    JSON value is neither a ``dict`` nor a ``list``.

    ``generate_json()`` never silently returns malformed data. Callers
    that receive this exception should treat the JSON contract as
    broken for this request.

    Attributes:
        raw_text: The full text that failed to parse. Preserved here
            so callers and tests can inspect exactly what was returned.
        parse_error: The original ``json.JSONDecodeError``, or
            ``None`` when the failure was a type check rather than a
            parse error (e.g. the JSON was valid but the top-level
            value was a string or number rather than an object/array).
    """

    def __init__(
        self,
        message: str,
        raw_text: str | None = None,
        parse_error: Exception | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, raw_text=raw_text, request_id=request_id)
        self.parse_error: Exception | None = parse_error

    def __repr__(self) -> str:  # pragma: no cover
        preview = (
            self.raw_text[:120] + "…"
            if self.raw_text and len(self.raw_text) > 120
            else self.raw_text
        )
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"raw_text={preview!r}, "
            f"parse_error={self.parse_error!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Prompt validation
# ============================================================


class GeminiPromptError(GeminiServiceError):
    """
    Raised by ``GeminiService._validate_prompt()`` when the prompt
    string supplied by the caller fails pre-flight validation.

    This exception fires **before** any network call is made, so it
    carries no ``request_id`` (one has not been assigned yet).

    Common causes
    -------------
    * ``prompt`` is not a ``str``.
    * ``prompt`` is empty or contains only whitespace.
    * ``prompt`` exceeds ``GEMINI_MAX_PROMPT_CHARS``.

    This exception is **not retryable**. The caller must fix the
    prompt before trying again.
    """

    # No additional attributes beyond the base class. Kept as a
    # distinct type so callers can distinguish prompt problems from
    # network or API problems with a single isinstance() check.
    pass


# ============================================================
# Public re-exports
# ============================================================

__all__: list[str] = [
    "GeminiServiceError",
    "GeminiConfigurationError",
    "GeminiTimeoutError",
    "GeminiRateLimitError",
    "GeminiSafetyError",
    "GeminiResponseError",
    "GeminiJsonParseError",
    "GeminiPromptError",
]