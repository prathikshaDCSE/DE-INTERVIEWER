"""
exceptions/report_exceptions.py

Exception hierarchy for ReportService.

All exceptions raised by ReportService are subclasses of
``ReportServiceError``. Callers should catch the most specific
subclass they intend to handle and let the rest propagate.

Hierarchy
---------
ReportServiceError                 <- base; catch-all for callers
├── ReportValidationError          <- bad input / invalid aggregated data / invalid output
├── ReportPromptError              <- translated PromptService failure
├── ReportAIError                  <- translated GeminiService failure
└── ReportConfigurationError       <- missing/invalid thresholds.yaml configuration

Design notes
------------
* Every exception carries a human-readable ``message`` and an optional
  ``request_id`` so log correlation is possible at every layer of the
  call stack, mirroring ``exceptions/evaluation_exceptions.py`` and
  ``exceptions/interview_exceptions.py``.
* This module never lets a raw exception from a collaborator
  (``PromptService``, ``GeminiService``) escape to ``ReportService``
  callers -- ``ReportService`` translates each of those into one of
  the types below via its own exception-translation helpers, so
  callers only ever need to know this module's hierarchy.
* Kept deliberately parallel in shape to
  ``exceptions/evaluation_exceptions.py`` so the two read the same way
  across the codebase. ``ReportService`` has no repository-layer
  dependency of its own (it operates entirely on an already-completed
  ``InterviewSession`` and already-produced ``EvaluationResult``
  objects handed to it by ``InterviewService``), so there is no
  ``ReportRepositoryError`` counterpart to
  ``EvaluationRepositoryError``.
"""

from __future__ import annotations


# ============================================================
# Base
# ============================================================


class ReportServiceError(Exception):
    """
    Base exception for all errors raised by ``ReportService``.

    Every more-specific exception in this module is a subclass of this
    one. Callers that want a single broad catch point should catch
    ``ReportServiceError``; callers that need fine-grained handling
    should catch the appropriate subclass.

    Attributes:
        message: Human-readable description of the error.
        request_id: Optional correlation ID from the originating
            ``ReportService`` call. ``None`` when the error occurs
            before a request ID is assigned.
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
# Validation
# ============================================================


class ReportValidationError(ReportServiceError):
    """
    Raised when the input to ``ReportService`` or the data it derives
    from that input fails validation -- for example an
    ``InterviewSession`` with no completed evaluations, an
    out-of-range overall score, a decision value that does not match
    any configured ``decision_thresholds`` band, malformed competency
    score aggregates, or a Gemini-generated report that fails
    structural validation (missing required sections, empty content,
    or a length outside configured bounds).

    Attributes:
        field: The name of the field or section that failed
            validation, when known (e.g. ``"overall_score"``,
            ``"decision"``, ``"report_markdown"``).
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.field: str | None = field

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"field={self.field!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Prompt
# ============================================================


class ReportPromptError(ReportServiceError):
    """
    Raised when ``PromptService`` cannot build the report prompt
    (missing placeholder values, prompt fails validation, or the
    rendered prompt exceeds the configured token budget).

    This is the translated form of any ``PromptServiceError`` (or
    subclass) raised by ``services.prompt_service`` --
    ``ReportService`` never lets a raw ``PromptServiceError`` escape
    to its own callers, and it never builds prompt text itself.
    """


# ============================================================
# AI / Gemini
# ============================================================


class ReportAIError(ReportServiceError):
    """
    Raised when ``GeminiService`` fails to produce a usable report for
    a report-generation request (timeout, rate limit, safety block,
    malformed response, or configuration problem).

    This is the translated form of any ``GeminiServiceError`` (or
    subclass) raised by ``services.gemini_service`` -- ``ReportService``
    never lets a raw ``GeminiServiceError`` escape to its own callers,
    and it never calls the Gemini SDK directly.
    """


# ============================================================
# Configuration
# ============================================================


class ReportConfigurationError(ReportServiceError):
    """
    Raised when required report-generation configuration is missing or
    invalid in ``thresholds.yaml`` -- for example ``decision_thresholds``,
    ``confidence_scoring.confidence_levels``, or
    ``answer_scoring.minimum_score`` / ``maximum_score`` being absent,
    malformed, or internally inconsistent.

    This exception is **not retryable**. The misconfiguration must be
    corrected in the underlying reference configuration before report
    generation can succeed.

    Attributes:
        config_key: Dotted path of the missing/invalid configuration
            key, when known (e.g. ``"decision_thresholds"``).
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
# Public re-exports
# ============================================================

__all__ = [
    "ReportServiceError",
    "ReportValidationError",
    "ReportPromptError",
    "ReportAIError",
    "ReportConfigurationError",
]