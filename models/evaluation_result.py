"""
models/evaluation_result.py

Structured, immutable representation of a single completed answer
evaluation. ``EvaluationService`` is the only thing that should
construct this object -- callers just read it.

Design notes
------------
* Kept as a plain ``@dataclass(frozen=True)`` rather than a pydantic
  model, mirroring ``models/gemini_response.py``: this is an internal,
  in-process value object produced entirely from data already
  validated by ``EvaluationService._validate_evaluation``, so no
  further validation is needed on the read path.
* This model represents exactly one thing: the outcome of evaluating
  one candidate answer to one question, including the adaptive
  follow-up decision for that answer. Raw Gemini JSON is deliberately
  never smuggled into this class -- ``EvaluationService`` always
  returns this normalized shape, never the underlying dict.
* Mutable inputs (lists) are converted to tuples in ``__post_init__``
  so the dataclass is genuinely immutable/hashable-safe, not merely
  frozen at the attribute level.
* ``timestamp`` is kept as a ``datetime`` object rather than a
  pre-formatted string -- sorting, comparison, filtering, and any
  future BigQuery/serialization use are all cleaner against a real
  ``datetime``. It is converted to ISO-8601 only at the ``to_dict()``
  boundary, mirroring how any future serialization of
  ``GeminiResponse`` should also work.
* ``next_action`` is typed as a ``Literal`` of the three legal values
  rather than a bare ``str``, so a typo like ``"CONTINEU"`` is caught
  by static analysis instead of silently becoming a fourth, unhandled
  state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

NextAction = Literal["FOLLOWUP", "ADVANCE", "CONTINUE"]


@dataclass(frozen=True)
class EvaluationResult:
    """
    Result of evaluating a single candidate answer.

    Attributes:
        score: Integer score assigned to the answer, within the range
            configured by the evaluation reference configuration
            (``answer_scoring``).
        confidence: Evaluator's confidence in this score, an integer
            0-100.
        strengths: What the candidate's answer did well.
        weaknesses: What the candidate's answer got wrong or omitted.
        missing_concepts: Specific expected concepts the answer did
            not address.
        evidence: Short excerpts/paraphrases from the candidate's
            answer supporting the score.
        recommendation: Free-text evaluator recommendation.
        needs_followup: Whether an adaptive follow-up question should
            be asked next, per ``determine_followup()``.
        next_action: One of ``"FOLLOWUP"``, ``"ADVANCE"``, or
            ``"CONTINUE"`` (see
            ``EvaluationService.determine_followup``).
        missing_concept: The single concept a follow-up question
            should target, or ``None`` when ``needs_followup`` is
            ``False``.
        followup_reason: Short explanation of why a follow-up is (or
            is not) needed, or ``None`` when no reason was determined.
        request_id: Correlation ID assigned by ``EvaluationService`` at
            the start of the call, threaded through logs and any
            exception raised for this request.
        timestamp: UTC ``datetime`` of when this result was
            constructed.
        metadata: Small, known set of contextual fields about how this
            result was produced (e.g. ``question_id``, ``competency``,
            ``stage``, ``difficulty``) -- never a dumping ground for
            raw Gemini/prompt output. Excluded from ``repr()`` to keep
            log lines short.
    """

    score: int
    confidence: int
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    missing_concepts: tuple[str, ...]
    evidence: tuple[str, ...]
    recommendation: str
    needs_followup: bool
    next_action: NextAction
    missing_concept: str | None
    followup_reason: str | None
    request_id: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass __setattr__ to coerce any
        # list-typed inputs into tuples so instances stay immutable
        # even if a caller passed lists.
        object.__setattr__(self, "strengths", tuple(self.strengths))
        object.__setattr__(self, "weaknesses", tuple(self.weaknesses))
        object.__setattr__(self, "missing_concepts", tuple(self.missing_concepts))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EvaluationResult("
            f"score={self.score!r}, "
            f"confidence={self.confidence!r}, "
            f"needs_followup={self.needs_followup!r}, "
            f"next_action={self.next_action!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the evaluation result."""
        return {
            "score": self.score,
            "confidence": self.confidence,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "missing_concepts": list(self.missing_concepts),
            "evidence": list(self.evidence),
            "recommendation": self.recommendation,
            "needs_followup": self.needs_followup,
            "next_action": self.next_action,
            "missing_concept": self.missing_concept,
            "followup_reason": self.followup_reason,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
        }