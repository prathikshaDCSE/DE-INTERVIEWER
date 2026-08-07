"""
services/evaluation_service.py

EvaluationService -- the business layer responsible for evaluating a
candidate's answer to a single interview question.

Position in the architecture::

    InterviewService
            |
            v
    EvaluationService
            |
            +-- QuestionRepository
            +-- PromptService
            +-- GeminiService
            +-- thresholds.yaml (via QuestionRepository.reference_data)

Responsibilities (and nothing more)
------------------------------------
* Validate evaluation inputs (question id, candidate answer,
  competency, stage, difficulty).
* Load the question being answered from ``QuestionRepository``.
* Ask ``PromptService`` to build the evaluation prompt -- this
  service never builds or renders prompts itself.
* Ask ``GeminiService`` to score the answer and return structured
  JSON -- this service never talks to Gemini, or any Gemini SDK type,
  directly.
* Validate and normalize the AI's JSON output into an
  ``EvaluationResult``.
* Decide, using only ``thresholds.yaml`` (never hardcoded values),
  whether an adaptive follow-up question is needed next.
* Track basic in-memory usage statistics.

Explicitly OUT of scope
------------------------
* Talking to the UI (``EvaluationService`` returns domain models only;
  presentation is the caller's job).
* Building or rendering prompts (that's ``PromptService``).
* Any direct Gemini SDK usage (that's ``GeminiService``).
* Selecting which question to ask next, or deciding interview flow
  beyond "does this specific answer need a follow-up" (that's
  ``InterviewService``).

Callers should catch ``EvaluationServiceError`` (or a specific
subclass from ``exceptions.evaluation_exceptions``) -- this module
never lets a raw exception from ``QuestionRepository``,
``PromptService``, or ``GeminiService`` escape to its own callers.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from config.settings import settings as default_settings
from exceptions.evaluation_exceptions import (
    EvaluationAIError,
    EvaluationConfigurationError,
    EvaluationPromptError,
    EvaluationRepositoryError,
    EvaluationServiceError,
    EvaluationValidationError,
)
from exceptions.gemini_exceptions import GeminiServiceError
from models.evaluation_result import EvaluationResult, NextAction
from repository.question_repository import (
    QuestionBankError,
    QuestionRepository,
)
from services.gemini_service import GeminiService
from services.prompt_service import PromptService, PromptServiceError

# ============================================================
# Module-level constants
# ============================================================

_NEXT_ACTION_FOLLOWUP: NextAction = "FOLLOWUP"
_NEXT_ACTION_ADVANCE: NextAction = "ADVANCE"
_NEXT_ACTION_CONTINUE: NextAction = "CONTINUE"

_VALID_NEXT_ACTIONS = frozenset(
    {_NEXT_ACTION_FOLLOWUP, _NEXT_ACTION_ADVANCE, _NEXT_ACTION_CONTINUE}
)

# Fields every Gemini evaluation payload must contain. Anything
# beyond these is ignored rather than rejected, so the AI can return
# extra diagnostic fields without breaking validation.
_REQUIRED_EVALUATION_FIELDS = (
    "score",
    "confidence",
    "strengths",
    "weaknesses",
    "missing_concepts",
    "evidence",
    "recommendation",
)

_LIST_EVALUATION_FIELDS = (
    "strengths",
    "weaknesses",
    "missing_concepts",
    "evidence",
)


class EvaluationService:
    """
    Business service responsible for evaluating candidate answers and
    deciding whether an adaptive follow-up is required next.

    Thread safety: one instance is safe to share across concurrent
    requests. In-memory statistics, the thresholds configuration
    cache, and the reserved-for-future-use evaluation cache are each
    protected by their own ``threading.RLock``; no mutable state is
    read or written outside those locks.
    """

    # =======================================================
    # Construction
    # =======================================================

    def __init__(
        self,
        question_repository: QuestionRepository,
        prompt_service: PromptService,
        gemini_service: GeminiService,
        settings: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the service.

        Args:
            question_repository: Already-loaded ``QuestionRepository``
                instance. Its validated competency/stage/difficulty
                sets and already-parsed ``thresholds.yaml`` reference
                data are reused directly rather than re-parsed, the
                same pattern ``PromptService`` uses -- avoiding
                duplicated configuration and drift between services.
            prompt_service: Injected ``PromptService`` used to build
                the evaluation prompt. ``EvaluationService`` never
                renders prompt text itself.
            gemini_service: Injected ``GeminiService`` used to call
                the AI model. ``EvaluationService`` never touches the
                Gemini SDK directly.
            settings: Optional injected settings object (defaults to
                the project-wide ``config.settings.settings``
                singleton). Reserved for configuration that is not
                sourced from ``thresholds.yaml`` via
                ``QuestionRepository`` (e.g. future feature flags);
                not required for current functionality.
            logger: Optional injected logger.

        Raises:
            EvaluationConfigurationError: If the evaluation reference
                configuration (as already loaded by
                ``question_repository`` from ``thresholds.yaml``) is
                missing the ``answer_scoring`` or
                ``adaptive_questioning`` sections required by this
                service.
        """
        self.question_repository = question_repository
        self.prompt_service = prompt_service
        self.gemini_service = gemini_service
        self.settings = settings if settings is not None else default_settings
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._cache_lock = threading.RLock()
        self._configuration_cache: dict[str, Any] = {}

        self._stats_lock = threading.RLock()
        self._statistics: dict[str, Any] = {
            "total_evaluations": 0,
            "successful": 0,
            "failed": 0,
            "followups_triggered": 0,
            "average_score": 0.0,
            "average_latency": 0.0,
        }

        self._refresh_configuration_cache()

    # =======================================================
    # Configuration
    # =======================================================

    def _refresh_configuration_cache(self) -> None:
        """
        Populate the configuration cache from
        ``question_repository.reference_data["thresholds"]`` without
        re-parsing ``thresholds.yaml``.

        Raises:
            EvaluationConfigurationError: If required sections are
                missing from the loaded threshold configuration.
        """
        thresholds = dict(
            self.question_repository.reference_data.get("thresholds", {})
        )

        answer_scoring = thresholds.get("answer_scoring")
        if not isinstance(answer_scoring, Mapping):
            raise EvaluationConfigurationError(
                "thresholds.yaml is missing the 'answer_scoring' section",
                config_key="answer_scoring",
            )

        adaptive_questioning = thresholds.get("adaptive_questioning")
        if not isinstance(adaptive_questioning, Mapping):
            raise EvaluationConfigurationError(
                "thresholds.yaml is missing the 'adaptive_questioning' section",
                config_key="adaptive_questioning",
            )

        with self._cache_lock:
            self._configuration_cache = {"thresholds": thresholds}

    def reload_configuration(self) -> None:
        """
        Reload threshold configuration from the question bank's
        reference data.

        Does not reload the question bank itself -- callers that need
        that should reload ``question_repository`` directly (e.g. via
        ``PromptService.reload_configuration`` or by calling
        ``question_repository.load_question_bank()``), then call this
        method to pick up any resulting change to ``thresholds.yaml``.

        Raises:
            EvaluationConfigurationError: If required sections are
                missing from the reloaded threshold configuration.
        """
        self.logger.info("Reloading EvaluationService configuration")
        self._refresh_configuration_cache()

    def _get_thresholds(self) -> Mapping[str, Any]:
        with self._cache_lock:
            return dict(self._configuration_cache.get("thresholds", {}))

    def _get_answer_scoring(self) -> Mapping[str, Any]:
        return self._get_thresholds().get("answer_scoring", {})

    def _get_adaptive_questioning(self) -> Mapping[str, Any]:
        return self._get_thresholds().get("adaptive_questioning", {})

    def _get_score_bounds(self) -> tuple[int, int]:
        answer_scoring = self._get_answer_scoring()
        minimum = answer_scoring.get("minimum_score")
        maximum = answer_scoring.get("maximum_score")
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise EvaluationConfigurationError(
                "thresholds.yaml answer_scoring.minimum_score / "
                "maximum_score must both be configured integers",
                config_key="answer_scoring.minimum_score/maximum_score",
            )
        if minimum > maximum:
            raise EvaluationConfigurationError(
                "thresholds.yaml answer_scoring.minimum_score exceeds "
                "maximum_score",
                config_key="answer_scoring.minimum_score/maximum_score",
            )
        return minimum, maximum

    # =======================================================
    # Public API -- Evaluation
    # =======================================================

    def evaluate_answer(
        self,
        question_id: str,
        candidate_answer: str,
        competency: str,
        stage: str,
        difficulty: str,
        current_followup_count: int = 0,
        question_record: Mapping[str, Any] | None = None,
        **gemini_overrides: Any,
    ) -> EvaluationResult:
        """
        Evaluate a candidate's answer to a single question.

        Orchestrates the full evaluation flow: validates inputs, loads
        the question from ``QuestionRepository``, builds the
        evaluation prompt via ``PromptService``, calls
        ``GeminiService.generate_json()``, validates and normalizes
        the structured response, and decides whether a follow-up is
        needed next.

        Args:
            question_id: Identifier of the question being answered.
            candidate_answer: The candidate's raw answer text. An
                empty/blank answer is rejected here (unlike
                ``PromptService.build_evaluation_prompt``, which
                tolerates blank answers) because scoring a "no
                answer" is a business decision this service must make
                deliberately, not a formatting concern.
            competency: Competency id the question belongs to (e.g.
                ``"C1"``); validated against
                ``question_repository.valid_competencies``.
            stage: Interview stage id the question belongs to (e.g.
                ``"S1"``); validated against
                ``question_repository.valid_stages``.
            difficulty: Difficulty label the question belongs to;
                validated against
                ``question_repository.valid_difficulties``.
            current_followup_count: Number of follow-ups already asked
                for this question/topic so far, used by
                ``determine_followup`` to enforce
                ``adaptive_questioning.maximum_followups``. Defaults
                to ``0`` (first answer, no follow-ups yet).
            question_record: Optional question dictionary to evaluate
                against directly (e.g. for dynamic or AI questions not in
                ``QuestionRepository``).
            **gemini_overrides: Forwarded to
                ``GeminiService.generate_json()`` (e.g.
                ``temperature=0``) to override per-call generation
                defaults.

        Returns:
            A populated ``EvaluationResult``.

        Raises:
            EvaluationValidationError: Inputs failed validation, or
                the AI's JSON payload failed evaluation validation.
            EvaluationRepositoryError: The question could not be
                loaded from ``QuestionRepository``.
            EvaluationPromptError: The evaluation prompt could not be
                built.
            EvaluationAIError: ``GeminiService`` failed to produce a
                usable response.
        """
        request_id = self._make_request_id()
        start = time.monotonic()

        try:
            self._validate_inputs(
                question_id=question_id,
                candidate_answer=candidate_answer,
                competency=competency,
                stage=stage,
                difficulty=difficulty,
            )

            resolved_question_record = self._load_question(
                question_id=question_id,
                competency=competency,
                stage=stage,
                difficulty=difficulty,
                question_record=question_record,
            )

            prompt_result = self._build_prompt(resolved_question_record, candidate_answer)

            raw_evaluation = self._call_gemini(
                prompt_result.prompt, request_id=request_id, **gemini_overrides
            )

            result = self._normalize_response(
                raw_evaluation,
                question_record=resolved_question_record,
                current_followup_count=current_followup_count,
                request_id=request_id,
            )

        except EvaluationServiceError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_statistics(
                success=False, score=None, latency_ms=latency_ms, needs_followup=False
            )
            self.logger.warning(
                "evaluation.failed",
                extra={
                    "request_id": request_id,
                    "question_id": question_id,
                    "competency": competency,
                    "stage": stage,
                    "difficulty": difficulty,
                    "error_type": type(exc).__name__,
                    "error": exc.message,
                    "latency_ms": latency_ms,
                },
            )
            raise

        latency_ms = (time.monotonic() - start) * 1000
        self._record_statistics(
            success=True,
            score=result.score,
            latency_ms=latency_ms,
            needs_followup=result.needs_followup,
        )
        self.logger.info(
            "evaluation.success",
            extra={
                "request_id": request_id,
                "question_id": question_id,
                "competency": competency,
                "stage": stage,
                "difficulty": difficulty,
                "score": result.score,
                "latency_ms": latency_ms,
                "needs_followup": result.needs_followup,
                "next_action": result.next_action,
            },
        )
        return result

    def determine_followup(
        self,
        evaluation_data: Mapping[str, Any],
        current_followup_count: int = 0,
    ) -> dict[str, Any]:
        """
        Decide whether an adaptive follow-up question is needed next.

        Rules come entirely from ``thresholds.yaml``
        ``adaptive_questioning`` (``followup_threshold``,
        ``advance_threshold``, ``maximum_followups``) -- never
        hardcoded.

        Args:
            evaluation_data: A mapping containing at least ``score``
                (and, optionally, ``missing_concepts`` /
                ``weaknesses`` used to explain the decision). This
                may be a raw (already-validated) Gemini payload or an
                ``EvaluationResult.to_dict()``.
            current_followup_count: Number of follow-ups already asked
                for this question/topic so far.

        Returns:
            A dictionary with ``needs_followup`` (bool),
            ``next_action`` (one of ``"FOLLOWUP"``, ``"ADVANCE"``,
            ``"CONTINUE"``), ``missing_concept`` (``str | None``), and
            ``reason`` (``str``).

        Raises:
            EvaluationValidationError: ``evaluation_data['score']`` is
                missing or not an integer, or
                ``current_followup_count`` is negative.
        """
        return self._determine_followup(evaluation_data, current_followup_count)

    def validate_evaluation(self, evaluation_data: Mapping[str, Any]) -> bool:
        """
        Validate a Gemini-produced (or otherwise supplied) evaluation
        payload against the fields and bounds required to build an
        ``EvaluationResult``.

        Args:
            evaluation_data: Mapping expected to contain ``score``,
                ``confidence``, ``strengths``, ``weaknesses``,
                ``missing_concepts``, ``evidence``, and
                ``recommendation``.

        Returns:
            ``True`` if the payload passes every check.

        Raises:
            EvaluationValidationError: If any required field is
                missing or fails its validation rule (see module
                docstring / class responsibilities for the exact
                rules per field).
        """
        self._validate_evaluation(evaluation_data)
        return True

    def normalize_response(
        self,
        raw_evaluation: Mapping[str, Any],
        question_record: Mapping[str, Any],
        current_followup_count: int = 0,
        request_id: str | None = None,
    ) -> EvaluationResult:
        """
        Convert a raw Gemini evaluation payload into an
        ``EvaluationResult``. Never returns raw Gemini JSON.

        Args:
            raw_evaluation: Parsed JSON dict from
                ``GeminiService.generate_json()``.
            question_record: The question dictionary the answer was
                scored against, as returned by ``QuestionRepository``.
            current_followup_count: Number of follow-ups already asked
                for this question/topic so far.
            request_id: Correlation id to attach to the result. A new
                one is generated when not supplied, e.g. when this
                method is called directly rather than via
                ``evaluate_answer``.

        Returns:
            A populated ``EvaluationResult``.

        Raises:
            EvaluationValidationError: ``raw_evaluation`` fails
                ``validate_evaluation``.
        """
        return self._normalize_response(
            raw_evaluation,
            question_record=question_record,
            current_followup_count=current_followup_count,
            request_id=request_id or self._make_request_id(),
        )

    def get_score_level(self, score: int) -> str:
        """
        Return the configured label for a given score.

        Reads ``thresholds.yaml`` ``answer_scoring.score_scale``
        exclusively -- the set of levels (e.g. "No Answer" through
        "Exceptional") and their meaning is entirely defined by
        configuration, never hardcoded here.

        Args:
            score: The score to look up.

        Returns:
            The configured label string for ``score`` (e.g.
            ``"Meets Expectations"``).

        Raises:
            EvaluationValidationError: ``score`` is not an integer
                within the configured score bounds.
            EvaluationConfigurationError: No ``score_scale`` entry is
                configured for ``score``.
        """
        minimum, maximum = self._get_score_bounds()
        if not isinstance(score, int) or isinstance(score, bool):
            raise EvaluationValidationError(
                "score must be an integer", field="score"
            )
        if score < minimum or score > maximum:
            raise EvaluationValidationError(
                f"score must be between {minimum} and {maximum}, got {score}",
                field="score",
            )

        score_scale = self._get_answer_scoring().get("score_scale", {})
        entry = score_scale.get(str(score))
        if not isinstance(entry, Mapping) or not entry.get("label"):
            raise EvaluationConfigurationError(
                f"No score_scale entry configured for score={score}",
                config_key="answer_scoring.score_scale",
            )
        return str(entry["label"])

    # =======================================================
    # Internal Helpers -- Validation
    # =======================================================

    def _validate_inputs(
        self,
        question_id: str,
        candidate_answer: str,
        competency: str,
        stage: str,
        difficulty: str,
    ) -> None:
        """
        Validate the raw inputs to ``evaluate_answer``.

        Raises:
            EvaluationValidationError: Any input is missing, blank,
                the wrong type, or not a recognized reference value.
        """
        if not isinstance(question_id, str) or not question_id.strip():
            raise EvaluationValidationError(
                "question_id must be a non-empty string", field="question_id"
            )

        self._validate_candidate_answer(candidate_answer)

        if competency not in self.question_repository.valid_competencies:
            raise EvaluationValidationError(
                f"Unknown competency value: {competency}", field="competency"
            )
        if stage not in self.question_repository.valid_stages:
            raise EvaluationValidationError(
                f"Unknown stage value: {stage}", field="stage"
            )
        if difficulty not in self.question_repository.valid_difficulties:
            raise EvaluationValidationError(
                f"Unknown difficulty value: {difficulty}", field="difficulty"
            )

    def _validate_candidate_answer(self, candidate_answer: str) -> None:
        """
        Raises:
            EvaluationValidationError: ``candidate_answer`` is
                ``None``, not a string, empty, or blank
                (whitespace-only).
        """
        if candidate_answer is None:
            raise EvaluationValidationError(
                "candidate_answer must not be None", field="candidate_answer"
            )
        if not isinstance(candidate_answer, str):
            raise EvaluationValidationError(
                "candidate_answer must be a string", field="candidate_answer"
            )
        if not candidate_answer.strip():
            raise EvaluationValidationError(
                "candidate_answer must not be empty or blank",
                field="candidate_answer",
            )

    def _validate_evaluation(self, evaluation_data: Mapping[str, Any]) -> None:
        """
        Raises:
            EvaluationValidationError: Any required field is missing
                or invalid.
        """
        if not isinstance(evaluation_data, Mapping):
            raise EvaluationValidationError(
                "evaluation_data must be a mapping", field="evaluation_data"
            )

        missing_fields = [
            name for name in _REQUIRED_EVALUATION_FIELDS if name not in evaluation_data
        ]
        if missing_fields:
            raise EvaluationValidationError(
                "Evaluation payload is missing required field(s): "
                + ", ".join(missing_fields)
            )

        minimum_score, maximum_score = self._get_score_bounds()
        score = evaluation_data["score"]
        if not isinstance(score, int) or isinstance(score, bool):
            raise EvaluationValidationError(
                "score must be an integer", field="score"
            )
        if score < minimum_score or score > maximum_score:
            raise EvaluationValidationError(
                f"score must be between {minimum_score} and {maximum_score}, "
                f"got {score}",
                field="score",
            )

        confidence = evaluation_data["confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise EvaluationValidationError(
                "confidence must be numeric", field="confidence"
            )
        if confidence < 0 or confidence > 100:
            raise EvaluationValidationError(
                f"confidence must be between 0 and 100, got {confidence}",
                field="confidence",
            )

        for field_name in _LIST_EVALUATION_FIELDS:
            value = evaluation_data[field_name]
            if not isinstance(value, list):
                raise EvaluationValidationError(
                    f"{field_name} must be a list", field=field_name
                )
            if not all(isinstance(item, str) for item in value):
                raise EvaluationValidationError(
                    f"{field_name} must be a list of strings", field=field_name
                )

        recommendation = evaluation_data["recommendation"]
        if not isinstance(recommendation, str) or not recommendation.strip():
            raise EvaluationValidationError(
                "recommendation must be a non-empty string",
                field="recommendation",
            )

    # =======================================================
    # Internal Helpers -- Collaborators
    # =======================================================

    def _load_question(
        self,
        question_id: str,
        competency: str,
        stage: str,
        difficulty: str,
        question_record: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """
        Load and cross-validate the question being answered.

        ``QuestionRepository`` does not expose a direct
        "get question by id" lookup, so this loads the stage's active
        question set (the narrowest existing filtered accessor) and
        matches on ``question_id`` client-side, then confirms the
        record's own competency/stage/difficulty agree with what the
        caller supplied -- catching a stale or mismatched question_id
        before it reaches the AI.

        Raises:
            EvaluationRepositoryError: ``QuestionRepository`` failed,
                or no question with ``question_id`` exists in
                ``stage``.
            EvaluationValidationError: A question was found but its
                competency or difficulty does not match the supplied
                values.
        """
        if question_record is not None:
            if question_record.get("competency") != competency:
                raise EvaluationValidationError(
                    "competency mismatch: requested="
                    f"{competency} question_record="
                    f"{question_record.get('competency')}",
                    field="competency",
                )
            if question_record.get("difficulty") != difficulty:
                raise EvaluationValidationError(
                    "difficulty mismatch: requested="
                    f"{difficulty} question_record="
                    f"{question_record.get('difficulty')}",
                    field="difficulty",
                )
            return question_record

        try:
            stage_questions = self.question_repository.get_questions_by_stage(stage)
        except QuestionBankError as exc:
            raise self._translate_repository_exception(exc) from exc

        found_record = next(
            (q for q in stage_questions if q.get("question_id") == question_id),
            None,
        )
        if found_record is None:
            raise EvaluationRepositoryError(
                f"Question not found: question_id={question_id} stage={stage}"
            )

        if found_record.get("competency") != competency:
            raise EvaluationValidationError(
                "competency mismatch: requested="
                f"{competency} question_record="
                f"{found_record.get('competency')}",
                field="competency",
            )
        if found_record.get("difficulty") != difficulty:
            raise EvaluationValidationError(
                "difficulty mismatch: requested="
                f"{difficulty} question_record="
                f"{found_record.get('difficulty')}",
                field="difficulty",
            )

        return found_record

    def _build_prompt(
        self,
        question_record: Mapping[str, Any],
        candidate_answer: str,
    ) -> Any:
        """
        Build the evaluation prompt via ``PromptService``.

        Raises:
            EvaluationPromptError: ``PromptService`` failed to build
                the prompt.
        """
        try:
            return self.prompt_service.build_evaluation_prompt(
                question_record, candidate_answer
            )
        except PromptServiceError as exc:
            raise EvaluationPromptError(
                f"Failed to build evaluation prompt: {exc}"
            ) from exc

    def _call_gemini(
        self,
        prompt: str,
        request_id: str,
        **generation_kwargs: Any,
    ) -> Any:
        """
        Call ``GeminiService.generate_json()`` and return the parsed
        payload.

        Raises:
            EvaluationAIError: ``GeminiService`` failed for any reason
                (timeout, rate limit, safety block, malformed
                response, JSON parse failure, or configuration
                problem).
        """
        try:
            return self.gemini_service.generate_json(prompt, **generation_kwargs)
        except GeminiServiceError as exc:
            raise self._translate_gemini_exception(exc, request_id) from exc

    # =======================================================
    # Internal Helpers -- Normalization / Follow-up
    # =======================================================

    def _normalize_response(
        self,
        raw_evaluation: Mapping[str, Any],
        question_record: Mapping[str, Any],
        current_followup_count: int,
        request_id: str,
    ) -> EvaluationResult:
        """
        Validate ``raw_evaluation`` and convert it into an
        ``EvaluationResult``, including the follow-up decision.

        Raises:
            EvaluationValidationError: ``raw_evaluation`` fails
                ``_validate_evaluation``.
        """
        self._validate_evaluation(raw_evaluation)

        followup_decision = self._determine_followup(
            raw_evaluation, current_followup_count
        )

        timestamp = datetime.now(timezone.utc)

        # confidence is validated as numeric (int or float) in the
        # 0-100 range by _validate_evaluation; it is normalized to an
        # int here (rounding rather than truncating) since
        # EvaluationResult.confidence is an integer field even though
        # Gemini may return a JSON float (e.g. 87.5).
        confidence = int(round(float(raw_evaluation["confidence"])))

        return EvaluationResult(
            score=int(raw_evaluation["score"]),
            confidence=confidence,
            strengths=list(raw_evaluation["strengths"]),
            weaknesses=list(raw_evaluation["weaknesses"]),
            missing_concepts=list(raw_evaluation["missing_concepts"]),
            evidence=list(raw_evaluation["evidence"]),
            recommendation=str(raw_evaluation["recommendation"]),
            needs_followup=followup_decision["needs_followup"],
            next_action=followup_decision["next_action"],
            missing_concept=followup_decision["missing_concept"],
            followup_reason=followup_decision["reason"],
            request_id=request_id,
            timestamp=timestamp,
            metadata={
                "question_id": question_record.get("question_id"),
                "competency": question_record.get("competency"),
                "stage": question_record.get("stage"),
                "difficulty": question_record.get("difficulty"),
            },
        )

    def _determine_followup(
        self,
        evaluation_data: Mapping[str, Any],
        current_followup_count: int,
    ) -> dict[str, Any]:
        """
        Core follow-up decision logic, driven entirely by
        ``adaptive_questioning`` in ``thresholds.yaml``.

        Decision order:
            1. If ``current_followup_count`` has already reached
               ``maximum_followups``, always advance -- the follow-up
               budget for this topic is spent regardless of score.
            2. Else if ``score <= followup_threshold``, a follow-up is
               needed.
            3. Else if ``score >= advance_threshold``, advance.
            4. Otherwise (a score between the two thresholds), the
               answer is adequate but does not warrant advancing
               early -- continue the interview without a follow-up.

        Raises:
            EvaluationValidationError: ``evaluation_data`` has no
                usable integer ``score``, or ``current_followup_count``
                is negative.
            EvaluationConfigurationError: The required
                ``adaptive_questioning`` threshold keys are missing.
        """
        if (
            not isinstance(current_followup_count, int)
            or isinstance(current_followup_count, bool)
            or current_followup_count < 0
        ):
            raise EvaluationValidationError(
                "current_followup_count must be a non-negative integer",
                field="current_followup_count",
            )

        score = evaluation_data.get("score")
        if not isinstance(score, int) or isinstance(score, bool):
            raise EvaluationValidationError(
                "evaluation_data['score'] must be an integer to determine "
                "a follow-up",
                field="score",
            )

        adaptive = self._get_adaptive_questioning()
        try:
            followup_threshold = int(
                adaptive["followup_threshold"]["score_less_than_or_equal"]
            )
            advance_threshold = int(
                adaptive["advance_threshold"]["score_greater_than_or_equal"]
            )
            maximum_followups = int(adaptive["maximum_followups"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationConfigurationError(
                "thresholds.yaml adaptive_questioning is missing or has "
                f"malformed threshold keys: {exc}",
                config_key="adaptive_questioning",
            ) from exc

        missing_concepts = list(evaluation_data.get("missing_concepts") or [])
        weaknesses = list(evaluation_data.get("weaknesses") or [])
        missing_concept = missing_concepts[0] if missing_concepts else None

        if current_followup_count >= maximum_followups:
            return {
                "needs_followup": False,
                "next_action": _NEXT_ACTION_ADVANCE,
                "missing_concept": None,
                "reason": (
                    f"Maximum follow-ups ({maximum_followups}) already "
                    "reached for this topic; advancing regardless of score."
                ),
            }

        if score <= followup_threshold:
            reason = (
                f"Score {score} is at or below the follow-up threshold "
                f"({followup_threshold})."
            )
            if weaknesses:
                reason += f" Primary weakness: {weaknesses[0]}"
            return {
                "needs_followup": True,
                "next_action": _NEXT_ACTION_FOLLOWUP,
                "missing_concept": missing_concept,
                "reason": reason,
            }

        if score >= advance_threshold:
            return {
                "needs_followup": False,
                "next_action": _NEXT_ACTION_ADVANCE,
                "missing_concept": None,
                "reason": (
                    f"Score {score} meets or exceeds the advance threshold "
                    f"({advance_threshold})."
                ),
            }

        return {
            "needs_followup": False,
            "next_action": _NEXT_ACTION_CONTINUE,
            "missing_concept": None,
            "reason": (
                f"Score {score} is between the follow-up threshold "
                f"({followup_threshold}) and the advance threshold "
                f"({advance_threshold}); continuing without a follow-up."
            ),
        }

    # =======================================================
    # Internal Helpers -- Exception Translation
    # =======================================================

    @staticmethod
    def _translate_gemini_exception(
        exc: GeminiServiceError, request_id: str | None = None
    ) -> EvaluationAIError:
        """Translate any ``GeminiServiceError`` into ``EvaluationAIError``."""
        return EvaluationAIError(
            f"Gemini evaluation call failed: {exc}",
            request_id=request_id or getattr(exc, "request_id", None),
        )

    @staticmethod
    def _translate_repository_exception(
        exc: QuestionBankError,
    ) -> EvaluationRepositoryError:
        """Translate any ``QuestionBankError`` into ``EvaluationRepositoryError``."""
        return EvaluationRepositoryError(f"Question repository failed: {exc}")

    # =======================================================
    # Internal Helpers -- Misc
    # =======================================================

    @staticmethod
    def _make_request_id() -> str:
        """Generate a short correlation id for a single evaluation call."""
        return uuid.uuid4().hex[:12]

    # =======================================================
    # Statistics
    # =======================================================

    def _record_statistics(
        self,
        success: bool,
        score: int | None,
        latency_ms: float,
        needs_followup: bool,
    ) -> None:
        """Update running statistics under ``self._stats_lock``."""
        with self._stats_lock:
            total_before = self._statistics["total_evaluations"]
            self._statistics["total_evaluations"] = total_before + 1

            if success:
                self._statistics["successful"] += 1
                if score is not None:
                    successful_before = self._statistics["successful"] - 1
                    previous_average = self._statistics["average_score"]
                    self._statistics["average_score"] = (
                        (previous_average * successful_before) + score
                    ) / self._statistics["successful"]
                if needs_followup:
                    self._statistics["followups_triggered"] += 1
            else:
                self._statistics["failed"] += 1

            previous_latency_average = self._statistics["average_latency"]
            self._statistics["average_latency"] = (
                (previous_latency_average * total_before) + latency_ms
            ) / self._statistics["total_evaluations"]

    def get_statistics(self) -> dict[str, Any]:
        """
        Return aggregated ``EvaluationService`` usage statistics.

        Returns:
            A dictionary with ``total_evaluations``, ``successful``,
            ``failed``, ``followups_triggered``, ``average_score``
            (mean of successful evaluations' scores), and
            ``average_latency`` (mean latency in milliseconds across
            all evaluations, successful or not).
        """
        with self._stats_lock:
            return dict(self._statistics)

    # =======================================================
    # Health Check
    # =======================================================

    def health_check(self) -> bool:
        """
        Lightweight liveness check for the collaborators
        ``evaluate_answer`` depends on.

        Verifies that the question repository has a loaded question
        bank and that ``GeminiService`` reports itself healthy. This
        does not call ``PromptService`` directly since it has no
        liveness probe of its own and performs no I/O beyond local
        template rendering.

        Returns:
            ``True`` if every checked collaborator is usable. ``False``
            if any check raises or reports unhealthy -- mirroring
            ``GeminiService.health_check()``, this method never lets an
            exception propagate to a caller polling it.
        """
        try:
            self.question_repository.get_total_questions()
            return bool(self.gemini_service.health_check())
        except Exception as exc:  # noqa: BLE001 - liveness probe, never raises
            self.logger.warning(
                "evaluation.health_check.failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return False