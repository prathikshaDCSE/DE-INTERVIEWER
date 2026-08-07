"""
services/interview_service.py

InterviewService -- the orchestration layer that coordinates the
entire interview lifecycle for the DE-INTERVIEWER platform.

Position in the architecture::

    FastAPI -> InterviewService -> QuestionRepository
                                 -> PromptService
                                 -> GeminiService (indirectly, via
                                    PromptService/EvaluationService)
                                 -> EvaluationService
                                 -> ReportService

``InterviewService`` is a pure orchestration layer. It never talks to
Gemini directly, never renders prompt text itself, never scores an
answer itself, and never builds the final report itself. Its only
responsibilities are:

* Own and mutate ``InterviewSession`` state for every active
  interview (in-memory only -- no BigQuery, Redis, or other
  persistence of any kind).
* Read stage/competency structure from the reference configuration
  already loaded by ``QuestionRepository`` (``stages.yaml`` /
  ``competencies.yaml``) -- never hardcoded, never re-parsed here.
* Select questions via ``QuestionRepository``, build prompts via
  ``PromptService``, and delegate answer scoring to
  ``EvaluationService``.
* Interpret ``EvaluationResult.next_action`` (``FOLLOWUP`` /
  ``ADVANCE`` / ``CONTINUE``) to decide what happens next -- ask a
  follow-up, move to the next question, or advance the stage.
* Advance stages strictly according to ``stages.yaml`` exit
  conditions, never according to hardcoded thresholds.
* Delegate final report generation to ``ReportService`` once an
  interview completes.
* Track thread-safe, in-memory usage statistics.

Callers should catch ``InterviewServiceError`` (or a specific
subclass from ``exceptions.interview_exceptions``) -- this module
never lets a raw exception from ``QuestionRepository``,
``PromptService``, ``GeminiService``, ``EvaluationService``, or
``ReportService`` escape to its own callers.

A note on ``ReportService``: it is not defined in this module. This
service assumes an injected collaborator exposing a
``generate_report(candidate, competency_summaries, overall_score,
decision, confidence_level, confidence_percentage, strengths,
weaknesses)`` method, mirroring the aggregated-summary-only contract
``PromptService.build_report_prompt`` already establishes. If that
signature differs once ``ReportService`` is implemented, only
``_generate_report`` in this file needs to change.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from config.settings import settings as default_settings
from exceptions.evaluation_exceptions import EvaluationServiceError
from exceptions.gemini_exceptions import GeminiServiceError
from exceptions.interview_exceptions import (
    InterviewAIError,
    InterviewConfigurationError,
    InterviewEvaluationError,
    InterviewPromptError,
    InterviewQuestionError,
    InterviewReportError,
    InterviewServiceError,
    InterviewSessionError,
    InterviewValidationError,
)
from models.evaluation_result import EvaluationResult
from models.interview_session import AskedQuestion, InterviewSession, InterviewStatus
from repository.question_repository import QuestionBankError, QuestionRepository
from services.evaluation_service import EvaluationService
from services.gemini_service import GeminiService
from services.prompt_service import PromptService, PromptServiceError

# ============================================================
# Module-level constants
# ============================================================

_STAGE_END = "END"

_DEFAULT_CANDIDATE_LEVEL = "Mid"

_EXPERIENCE_LEVEL_BUCKETS: tuple[tuple[float, str], ...] = (
    (2.0, "Junior"),
    (5.0, "Mid"),
    (float("inf"), "Senior"),
)

# Fallback difficulty used when a competency's configured
# difficulty_levels don't include "Medium" (e.g. C7, which only
# defines Easy/Medium) -- the first configured level is always a
# safe starting point.
_PREFERRED_STARTING_DIFFICULTY = "Medium"


class InterviewService:
    """
    Business orchestration service responsible for running an entire
    interview session from start to finish.

    Thread safety: one instance is safe to share across concurrently
    running interviews. All mutable state -- the session registry and
    every individual ``InterviewSession`` -- is protected by a single
    ``threading.RLock``. This trades finer-grained per-session locking
    for simplicity and correctness: interview traffic is
    request/response and human-paced (seconds between turns), so
    serializing session mutations behind one lock is not a realistic
    throughput bottleneck for this platform. In-memory statistics are
    protected by a separate ``threading.RLock``.
    """

    # =======================================================
    # Construction
    # =======================================================

    def __init__(
        self,
        question_repository: QuestionRepository,
        prompt_service: PromptService,
        gemini_service: GeminiService,
        evaluation_service: EvaluationService,
        report_service: Any,
        settings: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the service.

        Args:
            question_repository: Already-loaded ``QuestionRepository``
                instance. Its validated competency/stage/difficulty
                sets and parsed ``stages.yaml`` / ``competencies.yaml``
                reference data are reused directly.
            prompt_service: Injected ``PromptService`` used to build
                every prompt sent downstream. Never bypassed.
            gemini_service: Injected ``GeminiService``. Not called
                directly by this service today, but accepted per the
                architecture contract and exposed to ``health_check``.
            evaluation_service: Injected ``EvaluationService`` used to
                score every candidate answer.
            report_service: Injected report-generation collaborator
                used once an interview completes. See module
                docstring for the assumed interface.
            settings: Optional injected settings object (defaults to
                the project-wide ``config.settings.settings``
                singleton).
            logger: Optional injected logger.

        Raises:
            InterviewConfigurationError: If ``stages.yaml`` or
                ``competencies.yaml`` reference data (as already
                loaded by ``question_repository``) is missing
                required structure.
        """
        self.question_repository = question_repository
        self.prompt_service = prompt_service
        self.gemini_service = gemini_service
        self.evaluation_service = evaluation_service
        self.report_service = report_service
        self.settings = settings if settings is not None else default_settings
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._session_lock = threading.RLock()
        self._sessions: dict[str, InterviewSession] = {}

        self._config_lock = threading.RLock()
        self._stages: dict[str, dict[str, Any]] = {}
        self._stage_sequence: list[str] = []
        self._competencies: dict[str, dict[str, Any]] = {}

        self._stats_lock = threading.RLock()
        self._statistics: dict[str, Any] = {
            "interviews_started": 0,
            "interviews_completed": 0,
            "interviews_terminated": 0,
            "questions_asked": 0,
            "answers_evaluated": 0,
            "followups_asked": 0,
            "average_duration_seconds": 0.0,
            "average_score": 0.0,
        }

        self._refresh_configuration_cache()

    # =======================================================
    # Configuration
    # =======================================================

    def _refresh_configuration_cache(self) -> None:
        """
        Populate the stage/competency configuration cache from
        ``question_repository.reference_data`` without re-parsing any
        YAML file.

        Raises:
            InterviewConfigurationError: Required structure is
                missing from the loaded ``stages.yaml`` /
                ``competencies.yaml`` reference data.
        """
        stages_data = self.question_repository.reference_data.get("stages", {})
        raw_stages = stages_data.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise InterviewConfigurationError(
                "stages.yaml has no 'stages' list configured",
                config_key="stages",
            )

        stages: dict[str, dict[str, Any]] = {}
        for entry in raw_stages:
            if not isinstance(entry, Mapping) or not entry.get("id"):
                raise InterviewConfigurationError(
                    "stages.yaml contains a stage entry with no 'id'",
                    config_key="stages",
                )
            stage_id = str(entry["id"])
            if "next_stage" not in entry:
                raise InterviewConfigurationError(
                    f"stage {stage_id} is missing 'next_stage'",
                    config_key=f"stages.{stage_id}.next_stage",
                )
            stages[stage_id] = dict(entry)

        stage_sequence = [
            stage_id
            for stage_id, _ in sorted(
                stages.items(),
                key=lambda item: item[1].get("display_order", 0),
            )
        ]

        competencies_data = self.question_repository.reference_data.get(
            "competencies", {}
        )
        raw_competencies = competencies_data.get("competencies")
        if not isinstance(raw_competencies, list) or not raw_competencies:
            raise InterviewConfigurationError(
                "competencies.yaml has no 'competencies' list configured",
                config_key="competencies",
            )

        competencies: dict[str, dict[str, Any]] = {
            str(entry["id"]): dict(entry)
            for entry in raw_competencies
            if isinstance(entry, Mapping) and entry.get("id")
        }
        if not competencies:
            raise InterviewConfigurationError(
                "competencies.yaml produced no usable competency entries",
                config_key="competencies",
            )

        with self._config_lock:
            self._stages = stages
            self._stage_sequence = stage_sequence
            self._competencies = competencies

    def reload_configuration(self) -> None:
        """
        Reload the question bank and all downstream reference
        configuration (stages, competencies, thresholds, prompt
        templates), propagating the reload to every collaborator that
        caches configuration of its own.

        Does not affect any currently in-progress ``InterviewSession``
        -- only newly selected questions and newly built prompts pick
        up the reloaded configuration.

        Raises:
            InterviewConfigurationError: The reload failed at any
                layer.
        """
        self.logger.info("Reloading InterviewService configuration")
        try:
            self.prompt_service.reload_configuration()
            self.evaluation_service.reload_configuration()
        except (PromptServiceError, EvaluationServiceError) as exc:
            raise InterviewConfigurationError(
                f"Failed to reload downstream configuration: {exc}"
            ) from exc

        self._refresh_configuration_cache()

    def _get_stage(self, stage_id: str) -> dict[str, Any]:
        with self._config_lock:
            stage = self._stages.get(stage_id)
        if stage is None:
            raise InterviewConfigurationError(
                f"Unknown stage id: {stage_id}", config_key=f"stages.{stage_id}"
            )
        return stage

    def _first_stage_id(self) -> str:
        with self._config_lock:
            if not self._stage_sequence:
                raise InterviewConfigurationError(
                    "No stages configured", config_key="stages"
                )
            return self._stage_sequence[0]

    def _get_competency(self, competency_id: str) -> dict[str, Any]:
        with self._config_lock:
            competency = self._competencies.get(competency_id)
        if competency is None:
            raise InterviewConfigurationError(
                f"Unknown competency id: {competency_id}",
                config_key=f"competencies.{competency_id}",
            )
        return competency

    # =======================================================
    # Public API -- Lifecycle
    # =======================================================

    def start_interview(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """
        Start a new interview session for a candidate.

        Creates the session, loads the first configured stage, and
        auto-advances through any non-scored stage (e.g. an intake
        stage with zero configured questions) until a scored stage
        with questions is reached, then selects and builds the first
        question.

        Args:
            candidate: Candidate attributes. Must include ``name``.
                May include ``experience_years``, ``target_role``, and
                ``candidate_level`` (derived from
                ``experience_years`` when omitted).

        Returns:
            A dictionary with ``session_id``, ``stage``,
            ``competency``, ``question`` (the rendered interview
            prompt text), ``question_id``, and ``progress``.

        Raises:
            InterviewValidationError: ``candidate`` fails validation.
            InterviewConfigurationError: Stage configuration is
                unusable.
            InterviewQuestionError: No question could be selected for
                the first scored stage.
            InterviewPromptError: The first question's prompt could
                not be built.
        """
        self._validate_candidate(candidate)
        session = self._create_session(candidate)

        with self._session_lock:
            self._sessions[session.session_id] = session
            self._load_stage(session, self._first_stage_id())
            while self._should_advance_stage(session) and session.status == (
                InterviewStatus.IN_PROGRESS
            ):
                self._advance_stage_locked(session)
                if session.status != InterviewStatus.IN_PROGRESS:
                    break

            question_payload: dict[str, Any] | None = None
            if session.status == InterviewStatus.IN_PROGRESS:
                req_id = self._make_request_id()
                self._select_question(session, request_id=req_id)
                question_payload = self._build_question(session)

        with self._stats_lock:
            self._statistics["interviews_started"] += 1

        self.logger.info(
            "interview.started",
            extra={"session_id": session.session_id, "stage": session.current_stage},
        )

        return {
            "session_id": session.session_id,
            "stage": session.current_stage,
            "competency": session.current_competency,
            "question": question_payload.get("prompt") if question_payload else None,
            "question_id": (
                session.current_question.question_id
                if session.current_question
                else None
            ),
            "progress": self.get_progress(session.session_id),
        }

    def submit_answer(
        self,
        session_id: str,
        answer_text: str,
        **gemini_overrides: Any,
    ) -> dict[str, Any]:
        """
        Submit the candidate's answer to the currently active question
        (or active follow-up), evaluate it, and advance the interview
        accordingly.

        Args:
            session_id: The interview session id.
            answer_text: The candidate's raw answer text.
            **gemini_overrides: Forwarded to
                ``EvaluationService.evaluate_answer()``.

        Returns:
            A dictionary describing what happens next. Always
            contains ``next_action`` (``"FOLLOWUP"``, ``"NEXT_QUESTION"``,
            ``"STAGE_ADVANCED"``, or ``"INTERVIEW_COMPLETE"``),
            ``evaluation`` (the evaluation result as a dict), and
            ``progress``. When ``next_action`` is ``"FOLLOWUP"`` or
            ``"NEXT_QUESTION"``, also contains ``question`` (rendered
            prompt text).

        Raises:
            InterviewSessionError: The session does not exist, has
                already ended, or has no active question.
            InterviewValidationError: ``answer_text`` fails validation.
            InterviewEvaluationError: ``EvaluationService`` failed.
            InterviewPromptError: A follow-up/next-question prompt
                could not be built.
            InterviewQuestionError: No next question could be selected.
        """
        request_id = self._make_request_id()
        session = self._get_active_session(session_id)

        with self._session_lock:
            if session.current_question is None:
                raise InterviewSessionError(
                    "No active question to answer",
                    session_id=session_id,
                    request_id=request_id,
                )

            answer_key = self._current_answer_key(session)
            self._validate_answer_text(answer_text)
            session.answers[answer_key] = answer_text

            evaluation = self._evaluate_answer(
                session, answer_text, request_id=request_id, **gemini_overrides
            )
            session.evaluation_history.append(evaluation)

            self._update_progress(session, evaluation)

            result = self._handle_followup(session, evaluation, request_id=request_id)

        with self._stats_lock:
            self._statistics["answers_evaluated"] += 1
            if evaluation.needs_followup:
                self._statistics["followups_asked"] += 1

        self.logger.info(
            "interview.answer_submitted",
            extra={
                "session_id": session_id,
                "request_id": request_id,
                "next_action": result["next_action"],
                "score": evaluation.score,
            },
        )
        return result

    def get_current_question(self, session_id: str) -> dict[str, Any] | None:
        """
        Return the currently active question's rendered prompt,
        rebuilt fresh from stored session state.

        Args:
            session_id: The interview session id.

        Returns:
            A dictionary with the rendered ``prompt`` and metadata, or
            ``None`` if no question is currently active.

        Raises:
            InterviewSessionError: The session does not exist.
            InterviewPromptError: The prompt could not be rebuilt.
        """
        session = self._get_session(session_id)
        with self._session_lock:
            if session.current_question is None:
                return None
            return self._build_question(session)

    def get_next_question(self, session_id: str) -> dict[str, Any] | None:
        """
        Select and build the next question for the session's current
        stage/competency, replacing any current question.

        Intended for callers that want to explicitly skip ahead rather
        than go through ``submit_answer``'s natural progression.

        Args:
            session_id: The interview session id.

        Returns:
            The rendered next question payload, or ``None`` if the
            interview has no more questions to ask (stage/interview
            already complete).

        Raises:
            InterviewSessionError: The session does not exist or has
                already ended.
            InterviewQuestionError: No matching question is available.
            InterviewPromptError: The prompt could not be built.
        """
        session = self._get_active_session(session_id)
        with self._session_lock:
            if self._should_advance_stage(session):
                self._advance_stage_locked(session)
            if session.status != InterviewStatus.IN_PROGRESS:
                return None
            req_id = self._make_request_id()
            self._select_question(session, request_id=req_id)
            return self._build_question(session)

    def advance_stage(self, session_id: str) -> dict[str, Any]:
        """
        Force an advance out of the current stage, following
        ``stages.yaml``'s configured ``next_stage`` regardless of
        whether the stage's own exit conditions are met.

        Intended for administrative/override use; normal interview
        flow advances stages automatically via ``submit_answer``.

        Args:
            session_id: The interview session id.

        Returns:
            The updated progress dictionary (see ``get_progress``).

        Raises:
            InterviewSessionError: The session does not exist or has
                already ended.
            InterviewConfigurationError: The current stage has no
                usable ``next_stage``.
        """
        session = self._get_active_session(session_id)
        with self._session_lock:
            self._advance_stage_locked(session)
        return self.get_progress(session_id)

    def is_stage_complete(self, session_id: str) -> bool:
        """
        Return whether the session's current stage has satisfied its
        configured exit conditions.

        Args:
            session_id: The interview session id.

        Returns:
            ``True`` if the current stage is complete (or the
            interview has already ended), ``False`` otherwise.

        Raises:
            InterviewSessionError: The session does not exist.
        """
        session = self._get_session(session_id)
        with self._session_lock:
            if session.status != InterviewStatus.IN_PROGRESS:
                return True
            return self._should_advance_stage(session)

    def is_interview_complete(self, session_id: str) -> bool:
        """
        Return whether the interview has reached its completion
        condition (the terminal ``END`` stage, per ``stages.yaml``).

        Args:
            session_id: The interview session id.

        Returns:
            ``True`` if the session status is ``COMPLETED`` or
            ``TERMINATED``, ``False`` if still in progress.

        Raises:
            InterviewSessionError: The session does not exist.
        """
        session = self._get_session(session_id)
        return session.status != InterviewStatus.IN_PROGRESS

    def generate_summary(self, session_id: str) -> Any:
        """
        Generate the final interview report for a completed session by
        delegating to ``ReportService``.

        This method never builds report content itself: it aggregates
        already-computed competency scores, strengths, weaknesses, an
        overall score, and a decision, then hands that summary to
        ``ReportService``.

        Args:
            session_id: The interview session id.

        Returns:
            Whatever ``ReportService.generate_report()`` returns.

        Raises:
            InterviewSessionError: The session does not exist or has
                not yet completed.
            InterviewConfigurationError: ``decision_thresholds`` are
                not usable from configuration.
            InterviewReportError: ``ReportService`` failed.
        """
        session = self._get_session(session_id)
        if session.status == InterviewStatus.IN_PROGRESS:
            raise InterviewSessionError(
                "Cannot generate a summary for an interview still in progress",
                session_id=session_id,
            )

        with self._session_lock:
            competency_summaries = self._build_competency_summaries(session)
            overall_score = self._compute_overall_score(session)
            decision = self._decide_outcome(overall_score)
            confidence_level, confidence_percentage = self._compute_confidence(
                session
            )
            strengths = self._collect_top_points(
                session, attr="strengths", limit=10
            )
            weaknesses = self._collect_top_points(
                session, attr="weaknesses", limit=10
            )

        try:
            return self.report_service.generate_report(
                candidate=dict(session.candidate),
                competency_summaries=competency_summaries,
                overall_score=overall_score,
                decision=decision,
                confidence_level=confidence_level,
                confidence_percentage=confidence_percentage,
                strengths=strengths,
                weaknesses=weaknesses,
            )
        except Exception as exc:  # noqa: BLE001 - translate any report failure
            raise InterviewReportError(
                f"Report generation failed: {exc}", session_id=session_id
            ) from exc

    def get_progress(self, session_id: str) -> dict[str, Any]:
        """
        Return a snapshot of interview progress.

        Args:
            session_id: The interview session id.

        Returns:
            A dictionary with ``stage``, ``completed_stages``,
            ``questions_asked``, ``covered_competencies``,
            ``remaining_competencies``, ``followups_asked``,
            ``status``, and ``progress_percentage`` (0-100, based on
            completed stages relative to total configured stages).

        Raises:
            InterviewSessionError: The session does not exist.
        """
        session = self._get_session(session_id)
        with self._config_lock:
            total_stages = len(self._stage_sequence)

        with self._session_lock:
            completed = len(session.completed_stages)
            percentage = (
                round((completed / total_stages) * 100, 2) if total_stages else 0.0
            )
            return {
                "session_id": session.session_id,
                "stage": session.current_stage,
                "completed_stages": list(session.completed_stages),
                "questions_asked": len(session.asked_questions),
                "covered_competencies": sorted(session.covered_competencies),
                "remaining_competencies": sorted(session.remaining_competencies),
                "followups_asked": session.total_followups(),
                "status": session.status.value,
                "progress_percentage": percentage,
            }

    # =======================================================
    # Public API -- Statistics / Health
    # =======================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Return aggregated ``InterviewService`` usage statistics.

        Returns:
            A dictionary with interview counts, question/answer/
            follow-up counts, average interview duration in seconds,
            and average overall score across completed interviews.
        """
        with self._stats_lock:
            return dict(self._statistics)

    def health_check(self) -> bool:
        """
        Lightweight liveness check for every collaborator this service
        depends on.

        Returns:
            ``True`` if the question repository has a loaded question
            bank and ``EvaluationService`` reports itself healthy.
            ``False`` if any check raises or reports unhealthy -- this
            method never lets an exception propagate to a caller
            polling it.
        """
        try:
            self.question_repository.get_total_questions()
            return bool(self.evaluation_service.health_check())
        except Exception as exc:  # noqa: BLE001 - liveness probe, never raises
            self.logger.warning(
                "interview.health_check.failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return False

    # =======================================================
    # Internal Helpers -- Validation
    # =======================================================

    def _validate_candidate(self, candidate: Mapping[str, Any]) -> None:
        """
        Raises:
            InterviewValidationError: ``candidate`` is not a mapping,
                or is missing a non-empty ``name``.
        """
        if not isinstance(candidate, Mapping) or not str(
            candidate.get("name", "")
        ).strip():
            raise InterviewValidationError(
                "candidate must be a mapping including a non-empty 'name'",
                field="candidate",
            )

    def _validate_answer_text(self, answer_text: str) -> None:
        """
        Raises:
            InterviewValidationError: ``answer_text`` is not a string.
                Blank answers are permitted here and are a business
                decision left to ``EvaluationService``.
        """
        if not isinstance(answer_text, str):
            raise InterviewValidationError(
                "answer_text must be a string", field="answer_text"
            )

    # =======================================================
    # Internal Helpers -- Session lifecycle
    # =======================================================

    def _create_session(self, candidate: Mapping[str, Any]) -> InterviewSession:
        """Build a new, empty ``InterviewSession`` for ``candidate``."""
        candidate_dict = dict(candidate)
        candidate_dict.setdefault(
            "candidate_level", self._infer_candidate_level(candidate_dict)
        )
        return InterviewSession(
            session_id=uuid.uuid4().hex,
            candidate=candidate_dict,
            started_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _infer_candidate_level(candidate: Mapping[str, Any]) -> str:
        experience = candidate.get("experience_years")
        if not isinstance(experience, (int, float)):
            return _DEFAULT_CANDIDATE_LEVEL
        for ceiling, level in _EXPERIENCE_LEVEL_BUCKETS:
            if experience <= ceiling:
                return level
        return _DEFAULT_CANDIDATE_LEVEL

    def _get_session(self, session_id: str) -> InterviewSession:
        """
        Raises:
            InterviewSessionError: No session exists for ``session_id``.
        """
        with self._session_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise InterviewSessionError(
                f"No interview session found: {session_id}", session_id=session_id
            )
        return session

    def _get_active_session(self, session_id: str) -> InterviewSession:
        """
        Raises:
            InterviewSessionError: No session exists, or it has
                already ended.
        """
        session = self._get_session(session_id)
        if session.status != InterviewStatus.IN_PROGRESS:
            raise InterviewSessionError(
                f"Interview session {session_id} is already {session.status.value}",
                session_id=session_id,
            )
        return session

    @staticmethod
    def _current_answer_key(session: InterviewSession) -> str:
        assert session.current_question is not None  # guarded by caller
        base = session.current_question.question_id
        followup_index = session.current_question.followup_count
        if followup_index == 0:
            return base
        return f"{base}::followup::{followup_index}"

    # =======================================================
    # Internal Helpers -- Stage management
    # =======================================================

    def _load_stage(self, session: InterviewSession, stage_id: str) -> None:
        """
        Enter ``stage_id``: set it as current, and (re)populate
        remaining-competency tracking from its configuration.

        Must be called while holding ``self._session_lock``.
        """
        if stage_id == _STAGE_END:
            session.status = InterviewStatus.COMPLETED
            session.ended_at = datetime.now(timezone.utc)
            session.current_stage = None
            session.current_competency = None
            session.current_question = None
            return

        stage_cfg = self._get_stage(stage_id)
        session.current_stage = stage_id
        session.questions_asked_in_stage = 0
        session.remaining_competencies = {
            str(c) for c in stage_cfg.get("competencies", [])
        } - session.covered_competencies
        session.current_competency = (
            next(iter(sorted(session.remaining_competencies)), None)
        )

    def _should_advance_stage(self, session: InterviewSession) -> bool:
        """
        Decide whether the current stage's exit conditions are met.

        Called while holding ``self._session_lock``.
        """
        if session.status != InterviewStatus.IN_PROGRESS or session.current_stage is None:
            return False

        stage_cfg = self._get_stage(session.current_stage)
        competencies = stage_cfg.get("competencies", [])
        maximum_questions = int(stage_cfg.get("maximum_questions", 0))
        minimum_questions = int(stage_cfg.get("minimum_questions", 0))

        # Non-scored stages (e.g. intake / wrap-up) with no configured
        # questions exit immediately once loaded.
        if not competencies and maximum_questions == 0:
            return True

        if session.questions_asked_in_stage >= maximum_questions:
            return True

        if (
            session.questions_asked_in_stage >= minimum_questions
            and not session.remaining_competencies
        ):
            return True

        return False

    def _advance_stage_locked(self, session: InterviewSession) -> None:
        """
        Move the session out of its current stage into
        ``next_stage``, marking the old stage completed.

        Must be called while holding ``self._session_lock``.

        Raises:
            InterviewConfigurationError: The current stage has no
                usable ``next_stage`` configured.
        """
        if session.current_stage is None:
            return

        stage_cfg = self._get_stage(session.current_stage)
        next_stage = stage_cfg.get("next_stage")
        if not next_stage:
            raise InterviewConfigurationError(
                f"stage {session.current_stage} has no 'next_stage' configured",
                config_key=f"stages.{session.current_stage}.next_stage",
            )

        session.completed_stages.append(session.current_stage)
        self._load_stage(session, str(next_stage))

        if session.status == InterviewStatus.COMPLETED:
            self.logger.info(
                "interview.completed", extra={"session_id": session.session_id}
            )
            with self._stats_lock:
                self._statistics["interviews_completed"] += 1
                self._record_duration_statistics(session)
                self._record_score_statistics(session)

    # =======================================================
    # Internal Helpers -- Question selection / building
    # =======================================================

    def _get_dynamic_gen_config(self) -> dict[str, Any]:
        with self._config_lock:
            ref_data = getattr(self.question_repository, "reference_data", {}) or {}
        thresholds = ref_data.get("thresholds", {}) or {}
        return thresholds.get("dynamic_generation", {}) or {}

    def _get_difficulty_progression_config(self) -> dict[str, Any]:
        with self._config_lock:
            ref_data = getattr(self.question_repository, "reference_data", {}) or {}
        thresholds = ref_data.get("thresholds", {}) or {}
        return thresholds.get("difficulty_progression", {}) or {}

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by converting to lowercase, removing punctuation, and collapsing whitespace."""
        import string
        if not text:
            return ""
        text_clean = text.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(text_clean.split())

    def _select_question(
        self, session: InterviewSession, request_id: str | None = None
    ) -> None:
        """
        Select the next question for ``session.current_competency`` in
        ``session.current_stage``, avoiding duplicates, and set it as
        ``session.current_question``.

        If QuestionRepository has a matching question, use it (source=REPOSITORY).
        If QuestionRepository returns None, generate one via PromptService and
        GeminiService (source=GEMINI) subject to session limits.

        Must be called while holding ``self._session_lock``.

        Raises:
            InterviewQuestionError: No competency available to select a question for,
                or repository failed.
            InterviewPromptError: Dynamic question prompt could not be built.
            InterviewAIError: AI question generation failed validation or retries.
        """
        if session.current_competency is None:
            raise InterviewQuestionError(
                "No competency available to select a question for",
                session_id=session.session_id,
            )

        req_id = request_id or self._make_request_id()
        competency = session.current_competency
        difficulty = self._current_difficulty(session, competency)

        try:
            question_record = self.question_repository.get_question(
                competency=competency,
                stage=session.current_stage,
                difficulty=difficulty,
                candidate_level=session.candidate.get(
                    "candidate_level", _DEFAULT_CANDIDATE_LEVEL
                ),
                experience=session.candidate.get("experience_years", 0) or 0,
                exclude_used=True,
            )
        except QuestionBankError as exc:
            raise InterviewQuestionError(
                f"Question repository failed: {exc}", session_id=session.session_id
            ) from exc

        if question_record is not None:
            self.question_repository.mark_question_used(question_record["question_id"])
            question_source = "REPOSITORY"
            is_dynamic = False
            fallback_reason = None
            generation_attempt = 0
            self.logger.info(
                "repository_question_used",
                extra={
                    "request_id": req_id,
                    "session_id": session.session_id,
                    "competency": competency,
                    "stage": session.current_stage,
                    "difficulty": difficulty,
                },
            )
        else:
            dyn_cfg = self._get_dynamic_gen_config()
            max_gen = dyn_cfg.get("max_generated_questions_per_session", 5)
            stop_failures = dyn_cfg.get("stop_generation_after_failures", 3)

            if (
                session.generated_question_count >= max_gen
                or session.failed_generation_attempts >= stop_failures
            ):
                fallback_reason = "GENERATION_LIMIT_REACHED"
                self.logger.warning(
                    "Dynamic question generation limit reached; attempting repository reuse or safe fallback",
                    extra={"session_id": session.session_id, "request_id": req_id},
                )
                try:
                    question_record = self.question_repository.get_question(
                        competency=competency,
                        stage=session.current_stage,
                        difficulty=difficulty,
                        candidate_level=session.candidate.get(
                            "candidate_level", _DEFAULT_CANDIDATE_LEVEL
                        ),
                        experience=session.candidate.get("experience_years", 0) or 0,
                        exclude_used=False,
                    )
                except QuestionBankError:
                    question_record = None

                if question_record is None:
                    generated_id = f"FALLBACK_Q_{uuid.uuid4().hex[:8]}"
                    question_record = {
                        "question_id": generated_id,
                        "competency": competency,
                        "stage": session.current_stage,
                        "difficulty": difficulty,
                        "question": f"Could you describe your core technical experience and key architectural principles related to {competency}?",
                        "learning_objective": "General competency assessment",
                        "business_context": "Safe fallback assessment",
                        "expected_concepts": ["core concepts", "experience", "trade-offs"],
                        "evaluator_notes": "Evaluate general clarity and technical foundation.",
                        "positive_indicators": ["Structured answer", "Clear examples"],
                        "negative_indicators": ["Vague response"],
                        "estimated_time": 5,
                        "is_ai_generated": False,
                    }
                question_source = "FALLBACK"
                is_dynamic = False
                generation_attempt = 0
            else:
                comp_cfg = self._get_competency(competency)
                stage_cfg = self._get_stage(session.current_stage) if session.current_stage else {}
                already_asked_topics = [
                    q.question_record.get("question")
                    for q in session.asked_questions
                    if q.question_record and q.question_record.get("question")
                ]
                question_history = [q.question_id for q in session.asked_questions]

                try:
                    prompt_result = self.prompt_service.build_dynamic_question_prompt(
                        competency=competency,
                        competency_description=comp_cfg.get("description") or comp_cfg.get("name"),
                        stage=session.current_stage,
                        stage_objective=stage_cfg.get("objective") or stage_cfg.get("name"),
                        difficulty=difficulty,
                        candidate_experience=session.candidate.get("experience_years", "Not specified"),
                        candidate_role=session.candidate.get("target_role", "Not specified"),
                        already_asked_topics=already_asked_topics,
                        question_history=question_history,
                    )
                except PromptServiceError as exc:
                    raise InterviewPromptError(
                        f"Failed to build dynamic question prompt: {exc}",
                        session_id=session.session_id,
                        request_id=req_id,
                    ) from exc

                validated_ai, generation_attempt = self._generate_and_validate_question(
                    session=session,
                    competency=competency,
                    stage=session.current_stage,
                    difficulty=difficulty,
                    prompt_text=prompt_result.prompt,
                    request_id=req_id,
                )
                generated_id = f"AI_Q_{uuid.uuid4().hex[:8]}"
                question_record = {
                    "question_id": generated_id,
                    "competency": competency,
                    "stage": session.current_stage,
                    "difficulty": difficulty,
                    "question": validated_ai["question"],
                    "learning_objective": validated_ai.get("learning_objective", ""),
                    "business_context": validated_ai.get("business_context", ""),
                    "expected_concepts": validated_ai.get("expected_concepts", []),
                    "evaluator_notes": validated_ai.get("evaluator_notes", ""),
                    "positive_indicators": validated_ai.get("positive_indicators", []),
                    "negative_indicators": validated_ai.get("negative_indicators", []),
                    "estimated_time": validated_ai.get("estimated_time", 5),
                    "is_ai_generated": True,
                }
                session.generated_questions.append(question_record)
                session.generated_question_count += 1

                question_source = "AI_GENERATED"
                is_dynamic = True
                fallback_reason = "REPOSITORY_EMPTY"
                self.logger.info(
                    "generated_question_used",
                    extra={
                        "request_id": req_id,
                        "session_id": session.session_id,
                        "competency": competency,
                        "stage": session.current_stage,
                        "difficulty": difficulty,
                        "attempt": generation_attempt,
                    },
                )

        asked = AskedQuestion(
            question_id=question_record["question_id"],
            competency=competency,
            stage=session.current_stage,
            difficulty=difficulty,
            question_record=question_record,
            asked_at=datetime.now(timezone.utc),
            question_source=question_source,
            is_dynamic=is_dynamic,
            fallback_reason=fallback_reason,
            generation_attempt=generation_attempt,
            prompt_version=PromptService.TEMPLATE_VERSION,
            model=getattr(self.gemini_service, "model", "gemini-3.6-flash"),
            temperature=float(self._get_dynamic_gen_config().get("temperature", 0.1)),
            generated_at=datetime.now(timezone.utc).isoformat() if is_dynamic else None,
        )
        session.current_question = asked
        session.question_source = question_source
        session.pending_followup_prompt_metadata = None
        session.asked_questions.append(asked)
        session.questions_asked_in_stage += 1
        session.questions_per_competency[competency] = (
            session.questions_per_competency.get(competency, 0) + 1
        )

        with self._stats_lock:
            self._statistics["questions_asked"] += 1

    def _current_difficulty(self, session: InterviewSession, competency: str) -> str:
        """Look up (or initialize) the adaptive difficulty for a competency."""
        difficulty_map: dict[str, str] = session.metadata.setdefault(
            "difficulty_by_competency", {}
        )
        if competency in difficulty_map:
            return difficulty_map[competency]

        levels = self._get_competency(competency).get("difficulty_levels", [])
        starting = (
            _PREFERRED_STARTING_DIFFICULTY
            if _PREFERRED_STARTING_DIFFICULTY in levels
            else (levels[0] if levels else _PREFERRED_STARTING_DIFFICULTY)
        )
        difficulty_map[competency] = starting
        return starting

    def _adapt_difficulty(
        self, session: InterviewSession, question_record: Mapping[str, Any], score: int
    ) -> None:
        """Adjust the next difficulty for a competency based on score, checking
        thresholds.yaml difficulty_progression configuration or question record guidance."""
        competency = question_record.get("competency")
        if not competency:
            return

        diff_cfg = self._get_difficulty_progression_config()
        increase_thresh = diff_cfg.get("increase_threshold", 4)
        decrease_thresh = diff_cfg.get("decrease_threshold", 2)

        levels = self._get_competency(competency).get("difficulty_levels", [])
        next_difficulty: str | None = None

        if score >= increase_thresh:
            next_difficulty = question_record.get("next_difficulty_if_score_ge_4")
            if not next_difficulty and levels:
                curr = self._current_difficulty(session, competency)
                if curr == "Easy" and "Medium" in levels:
                    next_difficulty = "Medium"
                elif curr == "Medium" and "Hard" in levels:
                    next_difficulty = "Hard"
        elif score <= decrease_thresh:
            next_difficulty = question_record.get("next_difficulty_if_score_le_2")
            if not next_difficulty and levels:
                curr = self._current_difficulty(session, competency)
                if curr == "Hard" and "Medium" in levels:
                    next_difficulty = "Medium"
                elif curr == "Medium" and "Easy" in levels:
                    next_difficulty = "Easy"

        if next_difficulty and next_difficulty in levels:
            session.metadata.setdefault("difficulty_by_competency", {})[
                competency
            ] = next_difficulty

    def _build_question(self, session: InterviewSession) -> dict[str, Any]:
        """
        Render the interview prompt for ``session.current_question`` via
        ``PromptService``.

        Must be called while holding ``self._session_lock``.

        Raises:
            InterviewPromptError: The prompt could not be built.
        """
        asked = session.current_question
        assert asked is not None  # guarded by callers

        previous_context = self._summarize_previous_context(session)

        try:
            prompt_result = self.prompt_service.build_interview_prompt(
                candidate=session.candidate,
                question_record=asked.question_record,
                stage=asked.stage,
                competency=asked.competency,
                difficulty=asked.difficulty,
                previous_context=previous_context,
            )
        except PromptServiceError as exc:
            raise InterviewPromptError(
                f"Failed to build interview prompt: {exc}",
                session_id=session.session_id,
            ) from exc

        return {
            "prompt": prompt_result.prompt,
            "question_id": asked.question_id,
            "competency": asked.competency,
            "stage": asked.stage,
            "difficulty": asked.difficulty,
            "question_source": asked.question_source,
            "is_dynamic": asked.is_dynamic,
            "fallback_reason": asked.fallback_reason,
            "generation_attempt": asked.generation_attempt,
        }

    @staticmethod
    def _summarize_previous_context(session: InterviewSession) -> str | None:
        """A short, non-transcript summary of recent progress, passed to
        ``PromptService`` as ``previous_context``. Never the raw transcript."""
        if not session.evaluation_history:
            return None
        completed = len(session.evaluation_history)
        covered = ", ".join(sorted(session.covered_competencies)) or "none yet"
        return (
            f"{completed} question(s) answered so far this interview. "
            f"Competencies covered so far: {covered}."
        )

    # =======================================================
    # Internal Helpers -- Evaluation / follow-up
    # =======================================================

    def _evaluate_answer(
        self,
        session: InterviewSession,
        answer_text: str,
        request_id: str,
        **gemini_overrides: Any,
    ) -> EvaluationResult:
        """
        Delegate scoring of the current question/follow-up to
        ``EvaluationService``.

        Must be called while holding ``self._session_lock``.

        Raises:
            InterviewEvaluationError: ``EvaluationService`` failed.
        """
        asked = session.current_question
        assert asked is not None  # guarded by callers

        followup_count = session.followup_counts.get(asked.question_id, 0)

        try:
            return self.evaluation_service.evaluate_answer(
                question_id=asked.question_id,
                candidate_answer=answer_text,
                competency=asked.competency,
                stage=asked.stage,
                difficulty=asked.difficulty,
                current_followup_count=followup_count,
                question_record=asked.question_record,
                **gemini_overrides,
            )
        except EvaluationServiceError as exc:
            raise InterviewEvaluationError(
                f"Answer evaluation failed: {exc}",
                session_id=session.session_id,
                request_id=request_id,
            ) from exc

    def _handle_followup(
        self,
        session: InterviewSession,
        evaluation: EvaluationResult,
        request_id: str,
    ) -> dict[str, Any]:
        """
        Act on ``evaluation.next_action``: ask a follow-up, move to the
        next question, or advance the stage.

        Must be called while holding ``self._session_lock``.

        Raises:
            InterviewPromptError: A follow-up prompt could not be built.
            InterviewQuestionError: No next question could be selected.
        """
        asked = session.current_question
        assert asked is not None  # guarded by callers

        if evaluation.next_action == "FOLLOWUP":
            followup_level = session.followup_counts.get(asked.question_id, 0) + 1
            session.followup_counts[asked.question_id] = followup_level
            asked.followup_count = followup_level
            session.followups_per_competency[asked.competency] = (
                session.followups_per_competency.get(asked.competency, 0) + 1
            )

            missing_concept = evaluation.missing_concept or "the missing concept"
            reason = evaluation.followup_reason or "additional depth is needed"

            previous_answer = session.answers.get(
                asked.question_id
                if followup_level == 1
                else f"{asked.question_id}::followup::{followup_level - 1}",
                "",
            )

            repo_followup = self.question_repository.get_followup_question(
                asked.question_id, followup_level
            )

            if repo_followup is not None and str(repo_followup).strip() != "":
                question_source = "REPOSITORY"
                is_dynamic = False
                fallback_reason = None
                generation_attempt = 0
                self.logger.info(
                    "repository_followup_used",
                    extra={
                        "request_id": request_id,
                        "session_id": session.session_id,
                        "competency": asked.competency,
                        "stage": asked.stage,
                        "difficulty": asked.difficulty,
                    },
                )
                try:
                    prompt_result = self.prompt_service.build_followup_prompt(
                        previous_answer=previous_answer,
                        question_record=asked.question_record,
                        followup_level=followup_level,
                        missing_concept=missing_concept,
                        followup_reason=reason,
                    )
                except PromptServiceError as exc:
                    raise InterviewPromptError(
                        f"Failed to build follow-up prompt: {exc}",
                        session_id=session.session_id,
                        request_id=request_id,
                    ) from exc
                followup_prompt_text = prompt_result.prompt
            else:
                dyn_cfg = self._get_dynamic_gen_config()
                max_f = dyn_cfg.get("max_generated_followups_per_session", 10)
                stop_failures = dyn_cfg.get("stop_generation_after_failures", 3)
                default_fallback = dyn_cfg.get("validation", {}).get(
                    "default_fallback_followup"
                ) or "Could you elaborate further on the technical details and trade-offs of your approach?"

                if (
                    session.generated_followup_count >= max_f
                    or session.failed_generation_attempts >= stop_failures
                ):
                    followup_prompt_text = default_fallback
                    question_source = "FALLBACK"
                    is_dynamic = False
                    fallback_reason = "GENERATION_LIMIT_REACHED"
                    generation_attempt = 0
                else:
                    try:
                        prompt_result = self.prompt_service.build_dynamic_followup_prompt(
                            original_question=asked.question_record.get("question", ""),
                            candidate_answer=previous_answer,
                            missing_concepts=evaluation.missing_concepts or [missing_concept],
                            weaknesses=evaluation.weaknesses,
                            competency=asked.competency,
                            stage=asked.stage,
                            difficulty=asked.difficulty,
                            followup_count=followup_level,
                        )
                    except PromptServiceError as exc:
                        raise InterviewPromptError(
                            f"Failed to build dynamic follow-up prompt: {exc}",
                            session_id=session.session_id,
                            request_id=request_id,
                        ) from exc

                    validated_ai_followup, generation_attempt = self._generate_and_validate_followup(
                        session=session,
                        asked=asked,
                        followup_level=followup_level,
                        missing_concept=missing_concept,
                        prompt_text=prompt_result.prompt,
                        request_id=request_id,
                    )
                    followup_prompt_text = validated_ai_followup["question"]
                    session.generated_followups.append({
                        "followup_id": f"AI_F_{uuid.uuid4().hex[:8]}",
                        "question_id": asked.question_id,
                        "followup_level": followup_level,
                        "question": followup_prompt_text,
                        "missing_concept": missing_concept,
                        "is_ai_generated": True,
                    })
                    session.generated_followup_count += 1
                    question_source = "AI_GENERATED"
                    is_dynamic = True
                    fallback_reason = "NO_FOLLOWUP"
                    self.logger.info(
                        "generated_followup_used",
                        extra={
                            "request_id": request_id,
                            "session_id": session.session_id,
                            "competency": asked.competency,
                            "stage": asked.stage,
                            "difficulty": asked.difficulty,
                        },
                    )

            session.question_source = question_source
            session.pending_followup_prompt_metadata = {
                "followup_level": followup_level,
                "missing_concept": missing_concept,
                "question_source": question_source,
                "is_dynamic": is_dynamic,
                "fallback_reason": fallback_reason,
                "generation_attempt": generation_attempt,
            }
            session.metadata.setdefault("previous_followup_prompts", []).append(followup_prompt_text)

            return {
                "next_action": "FOLLOWUP",
                "question": followup_prompt_text,
                "question_source": question_source,
                "is_dynamic": is_dynamic,
                "fallback_reason": fallback_reason,
                "generation_attempt": generation_attempt,
                "evaluation": evaluation.to_dict(),
                "progress": self.get_progress_locked(session),
            }

        # ADVANCE or CONTINUE: this question is done.
        self._adapt_difficulty(session, asked.question_record, evaluation.score)
        self._mark_competency(session, asked.competency)
        session.current_question = None
        session.pending_followup_prompt_metadata = None

        if self._should_advance_stage(session):
            self._advance_stage_locked(session)
            while (
                session.status == InterviewStatus.IN_PROGRESS
                and self._should_advance_stage(session)
            ):
                self._advance_stage_locked(session)
            if session.status != InterviewStatus.IN_PROGRESS:
                return {
                    "next_action": "INTERVIEW_COMPLETE",
                    "evaluation": evaluation.to_dict(),
                    "progress": self.get_progress_locked(session),
                }
            next_action_label = "STAGE_ADVANCED"
        else:
            next_action_label = "NEXT_QUESTION"

        self._select_question(session, request_id=request_id)
        question_payload = self._build_question(session)

        return {
            "next_action": next_action_label,
            "question": question_payload["prompt"],
            "question_source": question_payload.get("question_source"),
            "is_dynamic": question_payload.get("is_dynamic", False),
            "fallback_reason": question_payload.get("fallback_reason"),
            "generation_attempt": question_payload.get("generation_attempt", 0),
            "evaluation": evaluation.to_dict(),
            "progress": self.get_progress_locked(session),
        }

    # =======================================================
    # Internal Helpers -- AI Generation & Validation
    # =======================================================

    def _generate_and_validate_question(
        self,
        session: InterviewSession,
        competency: str,
        stage: str,
        difficulty: str,
        prompt_text: str,
        request_id: str,
    ) -> tuple[dict[str, Any], int]:
        dyn_cfg = self._get_dynamic_gen_config()
        max_retries = dyn_cfg.get("max_retry_attempts", getattr(self.gemini_service, "_max_retries", 3))
        total_attempts = max(1, max_retries + 1)
        max_len = dyn_cfg.get("validation", {}).get("max_word_count", 500) * 10

        last_exc: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                raw_json = self.gemini_service.generate_json(prompt_text)
                validated = self._validate_generated_question(
                    data=raw_json,
                    requested_competency=competency,
                    requested_difficulty=difficulty,
                    asked_questions=session.asked_questions,
                    max_length=max_len,
                )
                return validated, attempt
            except (GeminiServiceError, InterviewValidationError, ValueError, TypeError) as exc:
                last_exc = exc
                self.logger.warning(
                    "AI question generation attempt %d/%d failed: %s",
                    attempt,
                    total_attempts,
                    exc,
                    extra={
                        "request_id": request_id,
                        "session_id": session.session_id,
                        "competency": competency,
                        "stage": stage,
                        "difficulty": difficulty,
                    },
                )

        raise InterviewAIError(
            f"AI question generation failed after {total_attempts} attempts: {last_exc}",
            session_id=session.session_id,
            request_id=request_id,
        ) from last_exc

    def _validate_generated_question(
        self,
        data: Any,
        requested_competency: str,
        requested_difficulty: str,
        asked_questions: Sequence[AskedQuestion],
        max_length: int = 1000,
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise InterviewValidationError("Generated question output must be a JSON object")

        required_fields = ["question", "competency", "difficulty", "estimated_time"]
        for key in required_fields:
            if key not in data or data[key] is None or (isinstance(data[key], str) and not data[key].strip()):
                raise InterviewValidationError(f"Missing required field in generated question: {key}")

        question = str(data["question"]).strip()
        if not question:
            raise InterviewValidationError("Question text must not be empty")

        if len(question) > max_length:
            raise InterviewValidationError(
                f"Question length ({len(question)} chars) exceeds maximum configured length ({max_length})"
            )

        evaluator_notes = str(data.get("evaluator_notes", "")).strip().lower()
        if evaluator_notes and len(evaluator_notes) > 5 and evaluator_notes in question.lower():
            raise InterviewValidationError("Generated question text must not contain the answer")

        norm_question = self._normalize_text(question)
        for asked in asked_questions:
            prev_q = asked.question_record.get("question") if asked.question_record else None
            if prev_q and norm_question == self._normalize_text(str(prev_q)):
                raise InterviewValidationError("Generated question duplicates an already asked question")

        ai_competency = str(data["competency"]).strip()
        if ai_competency != requested_competency:
            raise InterviewValidationError(
                f"Generated competency '{ai_competency}' does not match requested '{requested_competency}'"
            )

        ai_difficulty = str(data["difficulty"]).strip()
        if ai_difficulty != requested_difficulty:
            raise InterviewValidationError(
                f"Generated difficulty '{ai_difficulty}' does not match requested '{requested_difficulty}'"
            )

        return data

    def _generate_and_validate_followup(
        self,
        session: InterviewSession,
        asked: AskedQuestion,
        followup_level: int,
        missing_concept: str,
        prompt_text: str,
        request_id: str,
    ) -> tuple[dict[str, Any], int]:
        dyn_cfg = self._get_dynamic_gen_config()
        max_retries = dyn_cfg.get("max_retry_attempts", getattr(self.gemini_service, "_max_retries", 3))
        total_attempts = max(1, max_retries + 1)

        previous_followups: list[str] = session.metadata.get("previous_followup_prompts", [])

        last_exc: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                raw_json = self.gemini_service.generate_json(prompt_text)
                validated = self._validate_generated_followup(
                    data=raw_json,
                    target_missing_concept=missing_concept,
                    previous_followups=previous_followups,
                )
                return validated, attempt
            except (GeminiServiceError, InterviewValidationError, ValueError, TypeError) as exc:
                last_exc = exc
                self.logger.warning(
                    "AI follow-up generation attempt %d/%d failed: %s",
                    attempt,
                    total_attempts,
                    exc,
                    extra={
                        "request_id": request_id,
                        "session_id": session.session_id,
                        "competency": asked.competency,
                        "stage": asked.stage,
                        "difficulty": asked.difficulty,
                    },
                )

        raise InterviewAIError(
            f"AI follow-up generation failed after {total_attempts} attempts: {last_exc}",
            session_id=session.session_id,
            request_id=request_id,
        ) from last_exc

    def _validate_generated_followup(
        self,
        data: Any,
        target_missing_concept: str,
        previous_followups: Sequence[str],
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise InterviewValidationError("Generated follow-up output must be a JSON object")

        if not data.get("question") or not str(data["question"]).strip():
            raise InterviewValidationError("Follow-up question text must not be empty")

        missing_concept = str(
            data.get("missing_concept")
            or data.get("missing_concepts")
            or target_missing_concept
            or ""
        ).strip()
        if not missing_concept:
            raise InterviewValidationError("Follow-up missing concept must not be empty")
        data["missing_concept"] = missing_concept

        followup_reason = str(
            data.get("followup_reason") or data.get("reason") or "followup needed"
        ).strip()
        if not followup_reason:
            raise InterviewValidationError("Follow-up reason must not be empty")
        data["followup_reason"] = followup_reason

        question = str(data["question"]).strip()
        norm_question = self._normalize_text(question)
        for prev in previous_followups:
            if norm_question == self._normalize_text(str(prev)):
                raise InterviewValidationError("Generated follow-up repeats a previous follow-up question")

        return data

    def _update_progress(
        self, session: InterviewSession, evaluation: EvaluationResult
    ) -> None:
        """
        Update running competency-score tracking after a new
        evaluation. Must be called while holding ``self._session_lock``.
        """
        asked = session.current_question
        assert asked is not None  # guarded by callers
        competency = asked.competency

        scores_for_competency = [
            result.score
            for result in session.evaluation_history
            if result.metadata.get("competency") == competency
        ]
        if scores_for_competency:
            session.competency_scores[competency] = sum(
                scores_for_competency
            ) / len(scores_for_competency)

    def _mark_competency(self, session: InterviewSession, competency: str) -> None:
        """
        Mark ``competency`` as covered once it has met its configured
        ``minimum_questions``. Must be called while holding
        ``self._session_lock``.
        """
        required = int(self._get_competency(competency).get("minimum_questions", 1))
        asked_count = session.questions_per_competency.get(competency, 0)
        if asked_count >= required:
            session.covered_competencies.add(competency)
            session.remaining_competencies.discard(competency)
            if session.current_competency == competency:
                session.current_competency = next(
                    iter(sorted(session.remaining_competencies)), None
                )
        elif session.remaining_competencies:
            # Still needs more questions in this competency; keep it
            # (or another remaining one) as the active competency.
            session.current_competency = (
                competency
                if competency in session.remaining_competencies
                else next(iter(sorted(session.remaining_competencies)), None)
            )

    def get_progress_locked(self, session: InterviewSession) -> dict[str, Any]:
        """Same as ``get_progress`` but for callers already holding
        ``self._session_lock`` and already holding the session object,
        avoiding lock re-acquisition and a second dict lookup."""
        with self._config_lock:
            total_stages = len(self._stage_sequence)
        completed = len(session.completed_stages)
        percentage = (
            round((completed / total_stages) * 100, 2) if total_stages else 0.0
        )
        return {
            "session_id": session.session_id,
            "stage": session.current_stage,
            "completed_stages": list(session.completed_stages),
            "questions_asked": len(session.asked_questions),
            "covered_competencies": sorted(session.covered_competencies),
            "remaining_competencies": sorted(session.remaining_competencies),
            "followups_asked": session.total_followups(),
            "status": session.status.value,
            "progress_percentage": percentage,
        }

    # =======================================================
    # Internal Helpers -- Summary / report aggregation
    # =======================================================

    def _build_competency_summaries(
        self, session: InterviewSession
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for competency, average_score in sorted(session.competency_scores.items()):
            competency_cfg = self._get_competency(competency)
            summaries.append(
                {
                    "competency": competency_cfg.get("name", competency),
                    "competency_percentage": round((average_score / 5.0) * 100, 2),
                    "weight_pct": competency_cfg.get("evaluation_weight"),
                    "question_count": session.questions_per_competency.get(
                        competency, 0
                    ),
                }
            )
        return summaries

    def _compute_overall_score(self, session: InterviewSession) -> float:
        """Weighted overall score (0-100) across every scored competency,
        weighted by each competency's configured ``evaluation_weight``,
        renormalized to the competencies actually covered."""
        if not session.competency_scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for competency, average_score in session.competency_scores.items():
            weight = float(self._get_competency(competency).get(
                "evaluation_weight", 0
            ))
            percentage = (average_score / 5.0) * 100
            weighted_sum += percentage * weight
            total_weight += weight

        if total_weight == 0:
            return round(
                sum(
                    (score / 5.0) * 100
                    for score in session.competency_scores.values()
                )
                / len(session.competency_scores),
                2,
            )
        return round(weighted_sum / total_weight, 2)

    def _decide_outcome(self, overall_score: float) -> str:
        """
        Raises:
            InterviewConfigurationError: ``decision_thresholds`` is not
                configured in ``thresholds.yaml``.
        """
        thresholds = self.question_repository.reference_data.get(
            "thresholds", {}
        ).get("decision_thresholds", {})
        if not thresholds:
            raise InterviewConfigurationError(
                "thresholds.yaml is missing 'decision_thresholds'",
                config_key="decision_thresholds",
            )

        for decision, bounds in thresholds.items():
            minimum = bounds.get("minimum_score", 0)
            maximum = bounds.get("maximum_score", 100)
            if minimum <= overall_score <= maximum:
                return str(decision)

        # Fall back to the lowest-scoring configured band if score
        # falls outside every configured range (e.g. malformed config).
        return str(min(thresholds, key=lambda d: thresholds[d].get("minimum_score", 0)))

    def _compute_confidence(
        self, session: InterviewSession
    ) -> tuple[str, float]:
        """Confidence percentage is the mean of every evaluation's own
        ``confidence`` field; the level is looked up from
        ``thresholds.yaml`` ``confidence_scoring.confidence_levels``."""
        if not session.evaluation_history:
            confidence_percentage = 0.0
        else:
            confidence_percentage = round(
                sum(result.confidence for result in session.evaluation_history)
                / len(session.evaluation_history),
                2,
            )

        levels = (
            self.question_repository.reference_data.get("thresholds", {})
            .get("confidence_scoring", {})
            .get("confidence_levels", {})
        )
        confidence_level = "LOW"
        best_minimum = -1.0
        for label, bounds in levels.items():
            minimum = float(bounds.get("minimum_percentage", 0))
            if confidence_percentage >= minimum and minimum >= best_minimum:
                confidence_level = str(label)
                best_minimum = minimum

        return confidence_level, confidence_percentage

    @staticmethod
    def _collect_top_points(
        session: InterviewSession, attr: str, limit: int
    ) -> list[str]:
        points: list[str] = []
        for result in session.evaluation_history:
            points.extend(getattr(result, attr))
        # Preserve order, drop exact duplicates.
        seen: set[str] = set()
        deduped: list[str] = []
        for point in points:
            if point not in seen:
                seen.add(point)
                deduped.append(point)
        return deduped[:limit]

    # =======================================================
    # Internal Helpers -- Statistics
    # =======================================================

    def _record_duration_statistics(self, session: InterviewSession) -> None:
        """Must be called while holding ``self._stats_lock``."""
        duration = session.duration_seconds()
        completed_before = max(0, self._statistics["interviews_completed"] - 1)
        previous_average = self._statistics["average_duration_seconds"]
        self._statistics["average_duration_seconds"] = (
            (previous_average * completed_before) + duration
        ) / self._statistics["interviews_completed"]

    def _record_score_statistics(self, session: InterviewSession) -> None:
        """Must be called while holding ``self._stats_lock``."""
        average = session.average_score()
        if average is None:
            return
        completed_before = max(0, self._statistics["interviews_completed"] - 1)
        previous_average = self._statistics["average_score"]
        self._statistics["average_score"] = (
            (previous_average * completed_before) + average
        ) / self._statistics["interviews_completed"]

    def _record_statistics(self, **kwargs: Any) -> None:
        """
        Generic statistics recording hook, kept separate from the
        duration/score helpers above so future counters can be added
        without threading new parameters through every call site.
        """
        with self._stats_lock:
            self._statistics.update(kwargs)

    # =======================================================
    # Internal Helpers -- Misc
    # =======================================================

    @staticmethod
    def _make_request_id() -> str:
        """Generate a short correlation id for a single orchestration call."""
        return uuid.uuid4().hex[:12]