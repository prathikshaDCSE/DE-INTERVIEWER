"""
exceptions/interview_exceptions.py

Exception hierarchy for InterviewService.

All exceptions raised by InterviewService are subclasses of
``InterviewServiceError``. Callers should catch the most specific
subclass they intend to handle and let the rest propagate.

Hierarchy
---------
InterviewServiceError                  <- base; catch-all for callers
├── InterviewValidationError           <- bad candidate input / invalid args
├── InterviewSessionError              <- session lookup / lifecycle failure
├── InterviewConfigurationError        <- missing/invalid stages.yaml config
├── InterviewQuestionError             <- translated QuestionRepository failure
├── InterviewPromptError               <- translated PromptService failure
├── InterviewEvaluationError           <- translated EvaluationService failure
└── InterviewReportError               <- translated ReportService failure

Design notes
------------
* Every exception carries a human-readable ``message`` and an optional
  ``request_id`` / ``session_id`` so log correlation is possible at
  every layer of the call stack, mirroring
  ``exceptions/gemini_exceptions.py`` and
  ``exceptions/evaluation_exceptions.py``.
* This module never lets a raw exception from a collaborator
  (``QuestionRepository``, ``PromptService``, ``GeminiService``,
  ``EvaluationService``, ``ReportService``) escape to
  ``InterviewService`` callers -- ``InterviewService`` translates each
  of those into one of the types below, so callers only ever need to
  know this module's hierarchy.
* Kept deliberately parallel in shape to
  ``exceptions/evaluation_exceptions.py`` so the two read the same way
  across the codebase.
"""

from __future__ import annotations


# ============================================================
# Base
# ============================================================


class InterviewServiceError(Exception):
    """
    Base exception for all errors raised by ``InterviewService``.

    Every more-specific exception in this module is a subclass of this
    one. Callers that want a single broad catch point should catch
    ``InterviewServiceError``; callers that need fine-grained handling
    should catch the appropriate subclass.

    Attributes:
        message: Human-readable description of the error.
        session_id: Optional interview session ID the error pertains
            to. ``None`` when the error occurs before a session exists
            (e.g. during ``start_interview`` input validation).
        request_id: Optional correlation ID for the specific
            operation that failed. ``None`` when the error occurs
            before a request ID is assigned.
    """

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.session_id: str | None = session_id
        self.request_id: str | None = request_id

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"session_id={self.session_id!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Validation
# ============================================================


class InterviewValidationError(InterviewServiceError):
    """
    Raised when candidate-supplied inputs, method arguments, or an
    attempted state transition fail validation -- for example a
    missing candidate name, an out-of-range follow-up count, an
    unknown stage/competency/difficulty value, or a call made in an
    order the interview lifecycle does not allow (e.g. submitting an
    answer before a question has been asked).

    Attributes:
        field: The name of the field or argument that failed
            validation, when known (e.g. ``"candidate"``,
            ``"answer_text"``).
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, session_id=session_id, request_id=request_id)
        self.field: str | None = field

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"field={self.field!r}, "
            f"session_id={self.session_id!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Session lifecycle
# ============================================================


class InterviewSessionError(InterviewServiceError):
    """
    Raised when an interview session cannot be found, has already
    ended, is in a state that does not permit the requested
    operation, or otherwise fails a lifecycle check.

    Common causes
    -------------
    * ``session_id`` does not correspond to any in-memory session.
    * An operation was attempted on a session whose ``status`` is
      already ``COMPLETED`` or ``TERMINATED``.
    * ``get_current_question`` / ``submit_answer`` was called with no
      question currently active.

    This exception is generally **not retryable** with the same
    session -- the caller must start a new interview or query session
    state before retrying.
    """


# ============================================================
# Configuration
# ============================================================


class InterviewConfigurationError(InterviewServiceError):
    """
    Raised when required interview workflow configuration (as already
    loaded by ``QuestionRepository`` / ``PromptService`` from
    ``stages.yaml`` / ``thresholds.yaml``) is missing, malformed, or
    internally inconsistent -- for example a stage with no
    ``next_stage``, a stage referencing an unknown competency, or a
    missing ``completion_condition``.

    This exception is **not retryable**. The misconfiguration must be
    corrected in the underlying reference configuration before the
    interview can proceed.

    Attributes:
        config_key: Dotted path or stage id of the missing/invalid
            configuration, when known (e.g. ``"S2.next_stage"``).
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, session_id=session_id, request_id=request_id)
        self.config_key: str | None = config_key

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"config_key={self.config_key!r}, "
            f"session_id={self.session_id!r}, "
            f"request_id={self.request_id!r})"
        )


# ============================================================
# Question selection
# ============================================================


class InterviewQuestionError(InterviewServiceError):
    """
    Raised when ``QuestionRepository`` cannot supply a question needed
    to continue the interview (no matching question found for the
    current stage/competency/difficulty, repository not loaded, or
    any other repository-layer failure).

    This is the translated form of any exception raised by
    ``repository.question_repository`` -- ``InterviewService`` never
    lets a raw repository exception escape to its own callers.
    """


# ============================================================
# Prompt
# ============================================================


class InterviewPromptError(InterviewServiceError):
    """
    Raised when ``PromptService`` cannot build a prompt needed to
    continue the interview (missing placeholder values, prompt fails
    validation, or the rendered prompt exceeds the configured token
    budget).

    This is the translated form of any ``PromptServiceError`` (or
    subclass) raised by ``services.prompt_service`` --
    ``InterviewService`` never lets a raw ``PromptServiceError``
    escape to its own callers.
    """


# ============================================================
# Evaluation
# ============================================================


class InterviewEvaluationError(InterviewServiceError):
    """
    Raised when ``EvaluationService`` fails to produce a usable
    evaluation for a submitted answer (validation failure, repository
    failure, AI failure, prompt failure, or configuration failure at
    the evaluation layer).

    This is the translated form of any ``EvaluationServiceError`` (or
    subclass) raised by ``services.evaluation_service`` --
    ``InterviewService`` never lets a raw ``EvaluationServiceError``
    escape to its own callers.
    """


# ============================================================
# Report
# ============================================================


class InterviewReportError(InterviewServiceError):
    """
    Raised when ``ReportService`` fails to generate the final
    interview report/summary once an interview completes.

    This is the translated form of any exception raised by the
    report-generation layer -- ``InterviewService`` never lets a raw
    report-service exception escape to its own callers, and it never
    builds the report itself.
    """


# ============================================================
# Public re-exports
# ============================================================

__all__ = [
    "InterviewServiceError",
    "InterviewValidationError",
    "InterviewSessionError",
    "InterviewConfigurationError",
    "InterviewQuestionError",
    "InterviewPromptError",
    "InterviewEvaluationError",
    "InterviewReportError",
]