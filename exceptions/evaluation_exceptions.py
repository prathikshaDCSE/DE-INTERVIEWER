"""
exceptions/evaluation_exceptions.py

Exception hierarchy for EvaluationService.

All exceptions raised by EvaluationService are subclasses of
``EvaluationServiceError``. Callers should catch the most specific
subclass they intend to handle and let the rest propagate.

Hierarchy
---------
EvaluationServiceError                 <- base; catch-all for callers
├── EvaluationValidationError          <- bad input / bad AI output shape
├── EvaluationRepositoryError          <- repository lookup/retrieval failure
├── EvaluationAIError                  <- GeminiService failure
├── EvaluationPromptError              <- PromptService failure
└── EvaluationConfigurationError       <- missing/invalid reference configuration

Design notes
------------
* Every exception carries a human-readable ``message`` and an optional
  ``request_id`` so log correlation is possible at every layer of the
  call stack, mirroring ``exceptions/gemini_exceptions.py``.
* This module never raises a raw exception from a collaborator
  (``QuestionRepository``, ``PromptService``, ``GeminiService``)
  directly to EvaluationService callers -- ``EvaluationService``
  translates each of those into one of the types below via
  ``_translate_repository_exception`` / ``_translate_gemini_exception``
  / prompt-exception handling, so callers only ever need to know this
  module's hierarchy.
"""

from __future__ import annotations


# ============================================================
# Base
# ============================================================


class EvaluationServiceError(Exception):
    """
    Base exception for all errors raised by ``EvaluationService``.

    Every more-specific exception in this module is a subclass of this
    one. Callers that want a single broad catch point should catch
    ``EvaluationServiceError``; callers that need fine-grained
    handling should catch the appropriate subclass.

    Attributes:
        message: Human-readable description of the error.
        request_id: Optional correlation ID from the originating
            ``EvaluationService`` call. ``None`` when the error occurs
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


class EvaluationValidationError(EvaluationServiceError):
    """
    Raised when candidate-supplied inputs (question_id, candidate
    answer, competency, stage, difficulty) or a Gemini-produced
    evaluation payload fail validation.

    Attributes:
        field: The name of the field that failed validation, when
            known (e.g. ``"score"``, ``"candidate_answer"``).
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
# Repository
# ============================================================


class EvaluationRepositoryError(EvaluationServiceError):
    """
    Raised when the question repository cannot complete the requested
    question lookup or retrieval operation (not found, bank not
    loaded, invalid reference value, connection/timeout failure,
    missing table, invalid schema, or repository unavailable).

    This is the translated form of repository-layer exceptions raised
    by ``QuestionRepository`` -- ``EvaluationService`` never lets a raw
    repository exception escape to its own callers. Keeping this
    docstring generic (rather than naming a specific repository
    exception type) means the repository's own exception hierarchy can
    evolve or be renamed without requiring a change here.
    """


# ============================================================
# AI / Gemini
# ============================================================


class EvaluationAIError(EvaluationServiceError):
    """
    Raised when ``GeminiService`` fails to produce a usable response
    for an evaluation request (timeout, rate limit, safety block,
    malformed response, configuration problem, or invalid JSON).

    This is the translated form of any ``GeminiServiceError`` (or
    subclass) raised by ``services.gemini_service`` --
    ``EvaluationService`` never lets a raw ``GeminiServiceError``
    escape to its own callers.
    """


# ============================================================
# Prompt
# ============================================================


class EvaluationPromptError(EvaluationServiceError):
    """
    Raised when ``PromptService`` cannot build the evaluation prompt
    (missing placeholder values, prompt fails validation, or the
    rendered prompt exceeds the configured token budget).

    This is the translated form of any ``PromptServiceError`` (or
    subclass) raised by ``services.prompt_service`` --
    ``EvaluationService`` never lets a raw ``PromptServiceError``
    escape to its own callers.
    """


# ============================================================
# Configuration
# ============================================================


class EvaluationConfigurationError(EvaluationServiceError):
    """
    Raised when required evaluation configuration is missing or
    invalid (e.g. in ``thresholds.yaml``, ``competencies.yaml``, or
    other evaluation reference configuration) -- for example
    ``answer_scoring.score_scale``, ``answer_scoring.minimum_score`` /
    ``maximum_score``, or ``adaptive_questioning`` thresholds that are
    absent, malformed, or internally inconsistent (e.g. minimum >
    maximum).

    This exception is **not retryable**. The misconfiguration must be
    corrected in the underlying reference configuration before
    evaluation can succeed.

    Attributes:
        config_key: Dotted path of the missing/invalid configuration
            key, when known (e.g.
            ``"adaptive_questioning.maximum_followups"``).
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
    "EvaluationServiceError",
    "EvaluationValidationError",
    "EvaluationRepositoryError",
    "EvaluationAIError",
    "EvaluationPromptError",
    "EvaluationConfigurationError",
]