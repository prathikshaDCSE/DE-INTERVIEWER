"""
models/interview_session.py

Structured, mutable representation of a single in-progress or
completed interview session.

This is process-local, in-memory state owned exclusively by
``InterviewService`` -- no other component reads or writes it
directly, and it is never persisted to BigQuery, Redis, or any other
store. Per the current project requirement, session state exists only
for the lifetime of the active interview.

Design notes
------------
* Unlike ``PromptResult`` / ``EvaluationResult`` / ``GeminiResponse``
  (each an immutable snapshot of one outcome), ``InterviewSession``
  represents accumulating state across an entire interview. It is a
  plain, mutable dataclass, deliberately *not* frozen, and is mutated
  in place by ``InterviewService`` only while holding the service's
  session lock.
* ``evaluation_history`` stores ``EvaluationResult`` objects directly
  rather than raw dicts, keeping the architecture strongly typed end
  to end (per the project's own recommendation).
* ``AskedQuestion`` is a small nested record rather than reusing the
  raw ``QuestionRepository`` dict everywhere, so ``InterviewService``
  always has a stable, typed handle on "what was asked, when, and how
  many follow-ups it has had" without re-deriving that from the
  question bank row each time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from models.evaluation_result import EvaluationResult


class InterviewStatus(str, Enum):
    """Lifecycle status of an interview session."""

    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"


@dataclass
class AskedQuestion:
    """
    Record of a single question instance asked during the interview.

    Attributes:
        question_id: The underlying question bank identifier.
        competency: Competency id the question belongs to.
        stage: Stage id the question was asked in.
        difficulty: Difficulty label the question was asked at.
        question_record: The full question dictionary as returned by
            ``QuestionRepository``, retained so later steps (building
            follow-up prompts, scoring, summaries) never need to
            re-query the repository for the same question.
        asked_at: UTC timestamp when this question was presented.
        followup_count: Number of follow-up questions asked so far for
            this specific question instance.
    """

    question_id: str
    competency: str
    stage: str
    difficulty: str
    question_record: dict[str, Any]
    asked_at: datetime
    followup_count: int = 0
    question_source: str = "REPOSITORY"
    is_dynamic: bool = False
    fallback_reason: str | None = None
    generation_attempt: int = 0
    prompt_version: str = "1.0"
    model: str = "gemini-3.6-flash"
    temperature: float = 0.0
    generated_at: str | None = None


@dataclass
class InterviewSession:
    """
    In-memory state for a single interview.

    Attributes:
        session_id: Unique identifier for this interview session.
        candidate: Candidate attributes as supplied to
            ``start_interview`` (e.g. ``name``, ``experience_years``,
            ``target_role``, ``candidate_level``).
        started_at: UTC timestamp when the session was created.
        ended_at: UTC timestamp when the session finished, or ``None``
            while still in progress.
        current_stage: Current stage id (e.g. ``"S1"``), or ``None``
            before the session starts / after it ends.
        current_competency: Competency id currently being assessed, or
            ``None`` when not mid-question (e.g. during a stage
            transition).
        current_question: The ``AskedQuestion`` currently awaiting an
            answer, or ``None`` when no question is active.
        pending_followup_prompt_metadata: Metadata about an
            outstanding follow-up question that has been generated but
            not yet answered (``missing_concept``, ``followup_level``),
            or ``None``. The current question being answered is still
            ``current_question`` -- this field only distinguishes "the
            candidate is now answering a follow-up to that question"
            from "the candidate is answering it for the first time".
        asked_questions: Every question instance asked so far, in
            order.
        answers: Raw candidate answer text keyed by a stable answer
            key (``question_id`` for the original answer,
            ``question_id::followup::N`` for follow-up answers),
            preserved for the final summary.
        evaluation_history: Every ``EvaluationResult`` produced so
            far, in order.
        followup_counts: Number of follow-ups asked so far, keyed by
            the original ``question_id``.
        competency_scores: Running average score (0-5 scale) per
            competency, recomputed as evaluations arrive.
        covered_competencies: Competency ids that have met their
            configured ``minimum_questions`` requirement.
        remaining_competencies: Competency ids in the current stage
            not yet covered.
        questions_per_competency: Count of questions asked per
            competency id.
        followups_per_competency: Count of follow-ups asked per
            competency id.
        completed_stages: Stage ids that have satisfied their exit
            conditions and been advanced past.
        questions_asked_in_stage: Count of scored questions asked in
            ``current_stage`` so far (resets on stage advance).
        status: Current lifecycle status.
        metadata: Free-form, non-sensitive bookkeeping (e.g. adaptive
            difficulty tracking per competency, last request id).
            Excluded from ``repr()`` to keep logs short.
        generated_questions: Dynamic questions generated during this session.
        generated_followups: Dynamic follow-ups generated during this session.
        generated_question_count: Total dynamic primary questions generated.
        generated_followup_count: Total dynamic follow-ups generated.
        failed_generation_attempts: Count of failed generation retries.
    """

    session_id: str
    candidate: dict[str, Any]
    started_at: datetime
    ended_at: datetime | None = None

    current_stage: str | None = None
    current_competency: str | None = None
    current_question: AskedQuestion | None = None
    question_source: str | None = None
    pending_followup_prompt_metadata: dict[str, Any] | None = None

    asked_questions: list[AskedQuestion] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)
    evaluation_history: list[EvaluationResult] = field(default_factory=list)
    followup_counts: dict[str, int] = field(default_factory=dict)

    competency_scores: dict[str, float] = field(default_factory=dict)
    covered_competencies: set[str] = field(default_factory=set)
    remaining_competencies: set[str] = field(default_factory=set)
    questions_per_competency: dict[str, int] = field(default_factory=dict)
    followups_per_competency: dict[str, int] = field(default_factory=dict)

    completed_stages: list[str] = field(default_factory=list)
    questions_asked_in_stage: int = 0

    status: InterviewStatus = InterviewStatus.IN_PROGRESS

    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    generated_questions: list[dict[str, Any]] = field(default_factory=list)
    generated_followups: list[dict[str, Any]] = field(default_factory=list)
    generated_question_count: int = 0
    generated_followup_count: int = 0
    failed_generation_attempts: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"InterviewSession(session_id={self.session_id!r}, "
            f"status={self.status!r}, "
            f"current_stage={self.current_stage!r}, "
            f"questions_asked={len(self.asked_questions)!r})"
        )

    # =======================================================
    # Derived helpers
    # =======================================================

    def duration_seconds(self, now: datetime | None = None) -> float:
        """
        Return elapsed seconds since ``started_at``.

        Args:
            now: Current UTC time to measure against while the session
                is still in progress. Ignored once ``ended_at`` is set.

        Returns:
            Non-negative elapsed seconds.
        """
        end = self.ended_at or now or self.started_at
        return max(0.0, (end - self.started_at).total_seconds())

    def total_followups(self) -> int:
        """Return the total number of follow-up questions asked so far."""
        return sum(self.followup_counts.values())

    def average_score(self) -> float | None:
        """
        Return the mean score across all completed evaluations, or
        ``None`` if no evaluations have been recorded yet.
        """
        if not self.evaluation_history:
            return None
        return sum(result.score for result in self.evaluation_history) / len(
            self.evaluation_history
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "session_id": self.session_id,
            "candidate": dict(self.candidate),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "current_stage": self.current_stage,
            "current_competency": self.current_competency,
            "current_question_id": (
                self.current_question.question_id
                if self.current_question
                else None
            ),
            "questions_asked": len(self.asked_questions),
            "answers_submitted": len(self.answers),
            "evaluations_completed": len(self.evaluation_history),
            "followup_counts": dict(self.followup_counts),
            "competency_scores": dict(self.competency_scores),
            "covered_competencies": sorted(self.covered_competencies),
            "remaining_competencies": sorted(self.remaining_competencies),
            "questions_per_competency": dict(self.questions_per_competency),
            "followups_per_competency": dict(self.followups_per_competency),
            "completed_stages": list(self.completed_stages),
            "status": self.status.value,
        }