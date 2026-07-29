from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from repository.audit_repository import AuditRepository
from repository.question_repository import (
    QuestionBankError,
    QuestionNotFoundError,
    QuestionRepository,
)
from repository.user_repository import (
    InvalidUserError,
    UserNotFoundError,
    UserRepository,
)


# ============================================================
# Exceptions
# ============================================================


class QuestionServiceError(Exception):
    """Base exception raised by QuestionService."""


class InterviewSessionError(QuestionServiceError):
    """Raised when an interview session is invalid."""


class CandidateValidationError(QuestionServiceError):
    """Raised when candidate validation fails."""


class QuestionSelectionError(QuestionServiceError):
    """Raised when a question cannot be selected."""


class InterviewCompletedError(QuestionServiceError):
    """Raised when interview has already completed."""


# ============================================================
# Interview Session
# ============================================================


@dataclass
class InterviewSession:
    """
    Represents a single interview session.

    Sessions are maintained only in memory.
    Persistence belongs to ReportRepository.
    """

    session_id: str

    candidate_id: str

    competency: str

    stage: str

    candidate_level: str

    experience: float

    current_difficulty: str = "Easy"

    current_question_id: str | None = None

    asked_questions: list[str] = field(default_factory=list)

    score_history: list[int] = field(default_factory=list)

    total_questions: int = 0

    completed_questions: int = 0

    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    completed_at: datetime | None = None

    active: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Question Service
# ============================================================


class QuestionService:
    """
    Business layer responsible for interview orchestration.

    Responsibilities

    • Interview lifecycle
    • Question selection
    • Adaptive difficulty
    • Session management
    • Progress tracking
    • Audit logging

    This service never communicates with BigQuery directly.
    """

    DIFFICULTY_ORDER = (
        "Easy",
        "Medium",
        "Hard",
        "Expert",
    )

    PROMOTE_SCORE = 4

    DEMOTE_SCORE = 2

    def __init__(
        self,
        question_repository: QuestionRepository,
        user_repository: UserRepository,
        audit_repository: AuditRepository,
        logger: logging.Logger | None = None,
    ) -> None:

        self.question_repository = question_repository

        self.user_repository = user_repository

        self.audit_repository = audit_repository

        self.logger = logger or logging.getLogger(
            self.__class__.__name__
        )

        self._sessions: dict[str, InterviewSession] = {}

        self._lock = threading.RLock()

    # =======================================================
    # Internal Helpers
    # =======================================================

    def _generate_session_id(self) -> str:
        """
        Generate a globally unique interview session id.
        """
        return uuid.uuid4().hex

    def _get_session(
        self,
        session_id: str,
    ) -> InterviewSession:

        session = self._sessions.get(session_id)

        if session is None:
            raise InterviewSessionError(
                f"Session not found: {session_id}"
            )

        return session

    def _validate_session_active(
        self,
        session: InterviewSession,
    ) -> None:

        if not session.active:
            raise InterviewCompletedError(
                "Interview session is already closed."
            )

    def _validate_candidate(
        self,
        candidate_id: str,
    ) -> None:
        """
        Validate candidate exists.

        Delegates persistence to UserRepository.
        """

        try:

            if not self.user_repository.user_exists(candidate_id):

                raise CandidateValidationError(
                    f"Candidate does not exist: {candidate_id}"
                )

        except UserNotFoundError as exc:

            raise CandidateValidationError(
                str(exc)
            ) from exc

        except InvalidUserError as exc:

            raise CandidateValidationError(
                str(exc)
            ) from exc

    def _log_event(
        self,
        action: str,
        description: str,
        session: InterviewSession,
    ) -> None:
        """
        Send an audit event.

        Failures here should never stop interview flow.
        """

        try:

            self.audit_repository.log_event(
                {
                    "user_id": session.candidate_id,
                    "action": action,
                    "entity": "InterviewSession",
                    "entity_id": session.session_id,
                    "description": description,
                    "status": "SUCCESS",
                    "severity": "INFO",
                }
            )

        except Exception:

            self.logger.exception(
                "Failed to write audit log."
            )

    def start_interview(self, candidate_id: str, competency: str, stage: str,
                        candidate_level: str, experience: float,
                        difficulty: str = "Easy") -> InterviewSession:
        """Start a new interview session."""

        with self._lock:
            self._validate_candidate(candidate_id)

            for session in self._sessions.values():
                if session.candidate_id == candidate_id and session.active:
                    raise InterviewSessionError(
                        f"Candidate '{candidate_id}' already has an active interview."
                    )

            session = InterviewSession(
                session_id=self._generate_session_id(),
                candidate_id=candidate_id,
                competency=competency,
                stage=stage,
                candidate_level=candidate_level,
                experience=experience,
                current_difficulty=difficulty,
            )

            self._sessions[session.session_id] = session

            self.logger.info("Interview started for candidate %s", candidate_id)

            self._log_event(
                action="INTERVIEW_STARTED",
                description="Interview session created.",
                session=session,
            )

            return session

    def resume_interview(
        self,
        session_id: str,
    ) -> InterviewSession:
        """
        Resume an existing interview.
        """

        with self._lock:

            session = self._get_session(session_id)

            self._validate_session_active(session)

            self.logger.info(
                "Interview resumed: %s",
                session.session_id,
            )

            return session

    def end_interview(
        self,
        session_id: str,
    ) -> None:
        """
        Mark an interview as completed.
        """

        with self._lock:

            session = self._get_session(session_id)

            self._validate_session_active(session)

            session.active = False
            session.completed_at = datetime.now(timezone.utc)

            self.logger.info(
                "Interview ended: %s",
                session.session_id,
            )

            self._log_event(
                action="INTERVIEW_COMPLETED",
                description="Interview completed.",
                session=session,
            )

    def cancel_interview(
        self,
        session_id: str,
    ) -> None:
        """
        Cancel an interview session.
        """

        with self._lock:

            session = self._get_session(session_id)

            session.active = False

            self.logger.warning(
                "Interview cancelled: %s",
                session.session_id,
            )

            self._log_event(
                action="INTERVIEW_CANCELLED",
                description="Interview cancelled.",
                session=session,
            )

    def _is_question_used(
        self,
        session: InterviewSession,
        question_id: str,
    ) -> bool:
        """
        Check whether a question has already been asked.
        """
        return question_id in session.asked_questions

    def mark_question_used(
        self,
        session: InterviewSession,
        question_id: str,
    ) -> None:
        """
        Mark a question as asked during the interview.
        """
        if not self._is_question_used(session, question_id):
            session.asked_questions.append(question_id)
            session.completed_questions += 1

    def get_next_question(
        self,
        session_id: str,
    ):
        """
        Return the next question for the interview.

        Repository-layer exceptions (``QuestionBankError``,
        ``QuestionNotFoundError``) are translated into
        ``QuestionServiceError`` subclasses via
        ``_translate_repository_exception`` and never leak past this
        service.
        """

        with self._lock:

            session = self._get_session(session_id)

            self._validate_session_active(session)

            try:
                question = self.question_repository.get_question(
                    competency=session.competency,
                    stage=session.stage,
                    difficulty=session.current_difficulty,
                    candidate_level=session.candidate_level,
                    experience=session.experience,
                    exclude_used=True,
                )
            except (QuestionBankError, QuestionNotFoundError) as exc:
                raise self._translate_repository_exception(exc) from exc

            if question is None:
                raise QuestionSelectionError(
                    "No suitable question available."
                )

            question_id = question["question_id"]

            try:
                self.question_repository.mark_question_used(question_id)
            except (QuestionBankError, QuestionNotFoundError) as exc:
                raise self._translate_repository_exception(exc) from exc

            self.mark_question_used(
                session,
                question_id,
            )

            session.current_question_id = question_id
            session.total_questions += 1

            self._log_event(
                action="QUESTION_SELECTED",
                description=f"Question {question_id} selected.",
                session=session,
            )

            return question

    def get_followup_question(
        self,
        session_id: str,
        followup_number: int = 1,
    ):
        """
        Return a follow-up question for the current question.
        """
        with self._lock:

            session = self._get_session(session_id)

            if session.current_question_id is None:
                raise QuestionSelectionError("No current question exists.")

            current_question_id = session.current_question_id

        try:
            return self.question_repository.get_followup_question(
                current_question_id,
                followup_number,
            )
        except (QuestionBankError, QuestionNotFoundError) as exc:
            raise self._translate_repository_exception(exc) from exc

    # =======================================================
    # Class Constants (Business Rules)
    # =======================================================

    MIN_SCORE = 0

    MAX_SCORE = 5

    REMAIN_SCORE = 3

    DEFAULT_TARGET_QUESTIONS = 10

    RECOMMENDATION_STRONG_HIRE_THRESHOLD = 4.5

    RECOMMENDATION_HIRE_THRESHOLD = 3.5

    RECOMMENDATION_BORDERLINE_THRESHOLD = 2.5

    RECOMMENDATION_TRAINING_REQUIRED_THRESHOLD = 1.5

    RECOMMENDATION_STRONG_HIRE = "Strong Hire"

    RECOMMENDATION_HIRE = "Hire"

    RECOMMENDATION_BORDERLINE = "Borderline"

    RECOMMENDATION_TRAINING_REQUIRED = "Training Required"

    RECOMMENDATION_REJECT = "Reject"

    # =======================================================
    # Repository Exception Translation
    # =======================================================

    def _translate_repository_exception(
        self,
        exc: Exception,
    ) -> QuestionServiceError:
        """
        Translate a repository-layer exception into a service-layer
        exception.

        Repository exceptions (``QuestionBankError``,
        ``QuestionNotFoundError``, ``UserNotFoundError``,
        ``InvalidUserError``) must never leak past this service.

        Args:
            exc: The exception raised by a repository.

        Returns:
            The corresponding ``QuestionServiceError`` subclass, ready
            to be raised by the caller.
        """

        if isinstance(exc, QuestionNotFoundError):
            return QuestionSelectionError(str(exc))

        if isinstance(exc, QuestionBankError):
            return QuestionSelectionError(str(exc))

        if isinstance(exc, (UserNotFoundError, InvalidUserError)):
            return CandidateValidationError(str(exc))

        return QuestionServiceError(str(exc))

    # =======================================================
    # Score Validation Helper
    # =======================================================

    def _validate_score(
        self,
        score: int,
    ) -> None:
        """
        Validate that a score is an integer within the allowed range.

        Args:
            score: Candidate response score.

        Raises:
            QuestionServiceError: If the score is not an integer
                between ``MIN_SCORE`` and ``MAX_SCORE`` inclusive.
        """

        if isinstance(score, bool) or not isinstance(score, int):
            raise QuestionServiceError(
                f"Score must be an integer, got: {type(score).__name__}"
            )

        if score < self.MIN_SCORE or score > self.MAX_SCORE:
            raise QuestionServiceError(
                f"Score must be between {self.MIN_SCORE} and "
                f"{self.MAX_SCORE}, got: {score}"
            )

    # =======================================================
    # Part 2: Score Management
    # =======================================================

    def add_score(
        self,
        session_id: str,
        score: int,
    ) -> None:
        """
        Record a validated score against the session's score history.

        Args:
            session_id: Identifier of the interview session.
            score: Score between ``MIN_SCORE`` and ``MAX_SCORE``.

        Raises:
            QuestionServiceError: If the score is invalid.
        """

        self.logger.info("Start add_score: session=%s", session_id)

        with self._lock:

            session = self._get_session(session_id)

            self._validate_session_active(session)

            self._validate_score(score)

            session.score_history.append(score)

            self._log_event(
                action="SCORE_RECORDED",
                description=f"Score {score} recorded.",
                session=session,
            )

        self.logger.info("End add_score: session=%s", session_id)

    def get_total_score(
        self,
        session_id: str,
    ) -> int:
        """
        Return the sum of all recorded scores for a session.
        """

        with self._lock:

            session = self._get_session(session_id)

            return sum(session.score_history)

    def get_average_score(
        self,
        session_id: str,
    ) -> float:
        """
        Return the average recorded score for a session.

        Returns:
            The average score, or ``0.0`` when no scores exist.
        """

        with self._lock:

            session = self._get_session(session_id)

            if not session.score_history:
                return 0.0

            return sum(session.score_history) / len(session.score_history)

    def get_highest_score(
        self,
        session_id: str,
    ) -> int | None:
        """
        Return the highest recorded score for a session.

        Returns:
            The highest score, or ``None`` when no scores exist.
        """

        with self._lock:

            session = self._get_session(session_id)

            if not session.score_history:
                return None

            return max(session.score_history)

    def get_lowest_score(
        self,
        session_id: str,
    ) -> int | None:
        """
        Return the lowest recorded score for a session.

        Returns:
            The lowest score, or ``None`` when no scores exist.
        """

        with self._lock:

            session = self._get_session(session_id)

            if not session.score_history:
                return None

            return min(session.score_history)

    def clear_scores(
        self,
        session_id: str,
    ) -> None:
        """
        Clear all recorded scores for a session.
        """

        self.logger.info("Start clear_scores: session=%s", session_id)

        with self._lock:

            session = self._get_session(session_id)

            session.score_history.clear()

            self._log_event(
                action="SCORES_CLEARED",
                description="Score history cleared.",
                session=session,
            )

        self.logger.info("End clear_scores: session=%s", session_id)

    # =======================================================
    # Part 1: Difficulty Engine
    # =======================================================

    def calculate_next_difficulty(
        self,
        current_difficulty: str,
        score: int,
    ) -> str:
        """
        Determine the next difficulty level for a given score.

        Business rules:
            * Score 0-2   -> demote one level
            * Score 3     -> remain at current level
            * Score 4-5   -> promote one level

        Difficulty never exceeds ``Expert`` and never drops below
        ``Easy``.

        Args:
            current_difficulty: The candidate's current difficulty.
            score: The score just recorded.

        Returns:
            The resulting difficulty level.

        Raises:
            QuestionServiceError: If ``current_difficulty`` is not a
                recognized difficulty level.
        """

        self._validate_score(score)

        if current_difficulty not in self.DIFFICULTY_ORDER:
            raise QuestionServiceError(
                f"Unknown difficulty level: {current_difficulty}"
            )

        current_index = self.DIFFICULTY_ORDER.index(current_difficulty)

        if score <= self.DEMOTE_SCORE:
            next_index = max(current_index - 1, 0)
        elif score >= self.PROMOTE_SCORE:
            next_index = min(
                current_index + 1,
                len(self.DIFFICULTY_ORDER) - 1,
            )
        else:
            next_index = current_index

        return self.DIFFICULTY_ORDER[next_index]

    def _apply_difficulty_change(
        self,
        session: InterviewSession,
        new_difficulty: str,
        reason: str,
    ) -> None:
        """
        Apply and audit a difficulty transition for a session.

        Caller must already hold ``self._lock``.
        """

        previous_difficulty = session.current_difficulty

        if previous_difficulty == new_difficulty:
            return

        session.current_difficulty = new_difficulty

        self.logger.info(
            "Difficulty changed: session=%s %s -> %s (%s)",
            session.session_id,
            previous_difficulty,
            new_difficulty,
            reason,
        )

        self._log_event(
            action="DIFFICULTY_CHANGED",
            description=(
                f"Difficulty changed from {previous_difficulty} to "
                f"{new_difficulty} ({reason})."
            ),
            session=session,
        )

    def promote_difficulty(
        self,
        session_id: str,
    ) -> str:
        """
        Promote a session's difficulty by one level, capped at Expert.

        Returns:
            The resulting difficulty level.
        """

        with self._lock:

            session = self._get_session(session_id)

            current_index = self.DIFFICULTY_ORDER.index(
                session.current_difficulty
            )

            next_index = min(
                current_index + 1,
                len(self.DIFFICULTY_ORDER) - 1,
            )

            new_difficulty = self.DIFFICULTY_ORDER[next_index]

            self._apply_difficulty_change(
                session=session,
                new_difficulty=new_difficulty,
                reason="promotion",
            )

            return session.current_difficulty

    def demote_difficulty(
        self,
        session_id: str,
    ) -> str:
        """
        Demote a session's difficulty by one level, floored at Easy.

        Returns:
            The resulting difficulty level.
        """

        with self._lock:

            session = self._get_session(session_id)

            current_index = self.DIFFICULTY_ORDER.index(
                session.current_difficulty
            )

            next_index = max(current_index - 1, 0)

            new_difficulty = self.DIFFICULTY_ORDER[next_index]

            self._apply_difficulty_change(
                session=session,
                new_difficulty=new_difficulty,
                reason="demotion",
            )

            return session.current_difficulty

    def update_difficulty(
        self,
        session_id: str,
        score: int,
    ) -> str:
        """
        Recalculate and apply the next difficulty level based on score.

        Args:
            session_id: Identifier of the interview session.
            score: The score just recorded.

        Returns:
            The resulting difficulty level.
        """

        with self._lock:

            session = self._get_session(session_id)

            self._validate_session_active(session)

            new_difficulty = self.calculate_next_difficulty(
                current_difficulty=session.current_difficulty,
                score=score,
            )

            self._apply_difficulty_change(
                session=session,
                new_difficulty=new_difficulty,
                reason=f"score={score}",
            )

            return session.current_difficulty

    def record_score(
        self,
        session_id: str,
        score: int,
    ) -> str:
        """
        Record a score and update the session's difficulty accordingly.

        This combines ``add_score`` and ``update_difficulty`` into a
        single atomic, thread-safe operation. Both calls happen inside
        the same lock acquisition (``self._lock`` is reentrant), so no
        other thread can mutate the session between recording the
        score and applying the resulting difficulty change.

        Score-recording logic lives only in ``add_score``; this method
        does not duplicate it.

        Args:
            session_id: Identifier of the interview session.
            score: Score between ``MIN_SCORE`` and ``MAX_SCORE``.

        Returns:
            The resulting difficulty level.

        Raises:
            QuestionServiceError: If the score is invalid.
        """

        self.logger.info("Start record_score: session=%s", session_id)

        with self._lock:

            self.add_score(session_id, score)

            result_difficulty = self.update_difficulty(session_id, score)

        self.logger.info("End record_score: session=%s", session_id)

        return result_difficulty

    # =======================================================
    # Part 3: Interview Progress
    # =======================================================

    def _get_target_question_count(
        self,
        session: InterviewSession,
    ) -> int:
        """
        Return the planned total number of questions for a session.

        Falls back to ``DEFAULT_TARGET_QUESTIONS`` unless the session
        metadata specifies an override under ``target_question_count``.
        """

        return int(
            session.metadata.get(
                "target_question_count",
                self.DEFAULT_TARGET_QUESTIONS,
            )
        )

    def get_answered_questions(
        self,
        session_id: str,
    ) -> int:
        """
        Return the number of unique questions answered so far.
        """

        with self._lock:

            session = self._get_session(session_id)

            return session.completed_questions

    def get_remaining_questions(
        self,
        session_id: str,
    ) -> int:
        """
        Return the number of questions remaining in the interview.
        """

        with self._lock:

            session = self._get_session(session_id)

            target = self._get_target_question_count(session)

            return max(target - session.completed_questions, 0)

    def get_completion_percentage(
        self,
        session_id: str,
    ) -> float:
        """
        Return the interview completion percentage (0-100).
        """

        with self._lock:

            session = self._get_session(session_id)

            target = self._get_target_question_count(session)

            if target <= 0:
                return 0.0

            percentage = (session.completed_questions / target) * 100

            return min(percentage, 100.0)

    def get_progress(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Return a structured snapshot of interview progress.

        All sub-reads are taken under a single lock acquisition so the
        returned snapshot is internally consistent even if another
        thread mutates the session concurrently.

        Returns:
            A dictionary with ``total_questions``,
            ``answered_questions``, ``remaining_questions``,
            ``completion_percentage``, and ``difficulty``.
        """

        self.logger.info("Start get_progress: session=%s", session_id)

        with self._lock:

            session = self._get_session(session_id)

            target = self._get_target_question_count(session)

            progress = {
                "total_questions": target,
                "answered_questions": session.completed_questions,
                "remaining_questions": self.get_remaining_questions(
                    session_id
                ),
                "completion_percentage": self.get_completion_percentage(
                    session_id
                ),
                "difficulty": session.current_difficulty,
            }

        self.logger.info("End get_progress: session=%s", session_id)

        return progress

    # =======================================================
    # Part 4: Interview Statistics
    # =======================================================

    def get_statistics(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Return aggregate statistics for an interview session.

        All sub-reads are taken under a single lock acquisition so the
        returned snapshot is internally consistent even if another
        thread mutates the session concurrently.

        Returns:
            A dictionary with ``questions_asked``,
            ``questions_completed``, ``average_score``,
            ``highest_score``, ``lowest_score``, ``current_difficulty``,
            ``duration_seconds``, and ``active``.
        """

        with self._lock:

            session = self._get_session(session_id)

            end_time = session.completed_at or datetime.now(timezone.utc)

            duration_seconds = (
                end_time - session.started_at
            ).total_seconds()

            statistics_payload = {
                "questions_asked": session.total_questions,
                "questions_completed": session.completed_questions,
                "average_score": self.get_average_score(session_id),
                "highest_score": self.get_highest_score(session_id),
                "lowest_score": self.get_lowest_score(session_id),
                "current_difficulty": session.current_difficulty,
                "duration_seconds": duration_seconds,
                "active": session.active,
            }

        self.logger.info(
            "Statistics generated: session=%s",
            session_id,
        )

        return statistics_payload

    # =======================================================
    # Part 5: Interview Summary
    # =======================================================

    def generate_summary(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Generate an in-memory summary of the interview.

        This method never writes to ``ReportRepository``; it only
        composes and returns a summary payload. All sub-reads are
        taken under a single lock acquisition so the returned summary
        is internally consistent even if another thread mutates the
        session concurrently.

        Returns:
            A dictionary describing the candidate, scores, difficulty,
            and timing of the interview.
        """

        with self._lock:

            session = self._get_session(session_id)

            end_time = session.completed_at or datetime.now(timezone.utc)

            duration_seconds = (
                end_time - session.started_at
            ).total_seconds()

            summary = {
                "candidate_id": session.candidate_id,
                "session_id": session.session_id,
                "average_score": self.get_average_score(session_id),
                "highest_score": self.get_highest_score(session_id),
                "lowest_score": self.get_lowest_score(session_id),
                "questions_answered": session.completed_questions,
                "difficulty": session.current_difficulty,
                "started_at": session.started_at,
                "completed_at": session.completed_at,
                "duration_seconds": duration_seconds,
            }

            self.logger.info(
                "Summary generated: session=%s",
                session_id,
            )

            self._log_event(
                action="SUMMARY_GENERATED",
                description="Interview summary generated.",
                session=session,
            )

        return summary

    # =======================================================
    # Part 6: Interview Recommendation
    # =======================================================

    def get_recommendation(
        self,
        session_id: str,
    ) -> str:
        """
        Return a hiring recommendation based on the average score.

        Business rules:
            * Average >= 4.5  -> Strong Hire
            * Average >= 3.5  -> Hire
            * Average >= 2.5  -> Borderline
            * Average >= 1.5  -> Training Required
            * Otherwise       -> Reject

        Returns:
            The recommendation label.
        """

        with self._lock:

            session = self._get_session(session_id)

            average_score = self.get_average_score(session_id)

            if average_score >= self.RECOMMENDATION_STRONG_HIRE_THRESHOLD:
                recommendation = self.RECOMMENDATION_STRONG_HIRE
            elif average_score >= self.RECOMMENDATION_HIRE_THRESHOLD:
                recommendation = self.RECOMMENDATION_HIRE
            elif average_score >= self.RECOMMENDATION_BORDERLINE_THRESHOLD:
                recommendation = self.RECOMMENDATION_BORDERLINE
            elif (
                average_score
                >= self.RECOMMENDATION_TRAINING_REQUIRED_THRESHOLD
            ):
                recommendation = self.RECOMMENDATION_TRAINING_REQUIRED
            else:
                recommendation = self.RECOMMENDATION_REJECT

            self._log_event(
                action="RECOMMENDATION_GENERATED",
                description=f"Recommendation: {recommendation}.",
                session=session,
            )

        self.logger.info(
            "Recommendation generated: session=%s recommendation=%s",
            session_id,
            recommendation,
        )

        return recommendation