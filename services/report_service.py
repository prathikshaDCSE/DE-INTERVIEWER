"""
services/report_service.py

ReportService -- the reporting layer responsible for producing the
final hiring report once an interview has completed.

Position in the architecture::

    QuestionRepository -> PromptService -> GeminiService
                                                  |
                                                  v
                        EvaluationService -> InterviewService
                                                  |
                                                  v
                                            ReportService
                                                  |
                                                  v
                                        LangGraph (future)

``ReportService`` is a pure reporting layer. It never asks interview
questions, never drives interview flow, never evaluates candidate
answers, never touches the question repository's question bank, and
never calls the Gemini SDK directly. Its only responsibilities are:

* Accept an already-completed ``InterviewSession`` and read every
  ``EvaluationResult`` it contains.
* Aggregate those evaluations into per-competency score summaries,
  merged strengths, merged weaknesses, and a mean confidence.
* Compute the overall interview score itself -- Gemini is never asked
  to calculate scores, only to write the report explaining them.
* Determine the hiring decision, entirely from ``thresholds.yaml``
  ``decision_thresholds`` -- never hardcoded.
* Ask ``PromptService`` to build the report prompt -- this service
  never renders prompt text itself.
* Ask ``GeminiService`` to generate the report text -- this service
  never touches the Gemini SDK, or any Gemini SDK type, directly.
* Validate the generated report's structure and content.
* Return a strongly typed, immutable ``ReportResult``.
* Track thread-safe, in-memory usage statistics.

Callers should catch ``ReportServiceError`` (or a specific subclass
from ``exceptions.report_exceptions``) -- this module never lets a
raw exception from ``PromptService`` or ``GeminiService`` escape to
its own callers.

Configuration ownership: decision thresholds, confidence levels, and
competency names/weights are never hardcoded here. ``QuestionRepository``
already loads and validates ``competencies.yaml`` and
``thresholds.yaml`` at startup; ``ReportService`` reuses that
already-loaded reference data exclusively, the same pattern
``PromptService`` and ``EvaluationService`` already use, to avoid
duplicated configuration and drift across services.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from config.settings import settings as default_settings
from exceptions.gemini_exceptions import GeminiServiceError
from exceptions.report_exceptions import (
    ReportAIError,
    ReportConfigurationError,
    ReportPromptError,
    ReportServiceError,
    ReportValidationError,
)
from models.evaluation_result import EvaluationResult
from models.interview_session import InterviewSession
from models.report_result import CompetencyScoreSummary, ReportResult
from repository.question_repository import QuestionRepository
from services.gemini_service import GeminiService
from services.prompt_service import PromptService, PromptServiceError

# ============================================================
# Module-level constants
# ============================================================

# Exact headings required in every generated report, in required
# order, per report_prompt.md's REQUIRED OUTPUT FORMAT contract.
# Never hardcode this expectation anywhere else in the codebase.
_REQUIRED_REPORT_HEADINGS: tuple[str, ...] = (
    "## Executive Summary",
    "## Overall Assessment",
    "## Competency Breakdown",
    "## Technical Strengths",
    "## Technical Weaknesses",
    "## Communication Assessment",
    "## Evidence-Based Justification",
    "## Hiring Recommendation",
    "## Development Plan",
    "## Confidence Explanation",
)

# Markers that indicate the model left placeholder text instead of
# real content, which report_prompt.md explicitly forbids.
_PLACEHOLDER_MARKERS: tuple[str, ...] = ("TBD", "TODO", "[placeholder]", "N/A - TODO")

_MIN_REPORT_LENGTH_CHARS = 200

# Fallback recommendation used only when a weakness's text is too
# short/generic to derive a specific action from -- keeps
# recommendations grounded in the weakness itself rather than
# fabricating unrelated advice.
_GENERIC_RECOMMENDATION_TEMPLATE = "Address the following gap: {weakness}"


class ReportService:
    """
    Business service responsible for aggregating interview results and
    producing the final, validated hiring report.

    Thread safety: one instance is safe to share across concurrent
    report-generation requests. In-memory statistics and the
    thresholds/competency configuration cache are each protected by
    their own ``threading.RLock``; no mutable state is read or written
    outside those locks.
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
                instance. Its already-parsed ``thresholds.yaml`` and
                ``competencies.yaml`` reference data are reused
                directly rather than re-parsed, the same pattern
                ``PromptService`` and ``EvaluationService`` use --
                avoiding duplicated configuration and drift between
                services. ``ReportService`` never queries the question
                bank itself; it only reads reference configuration.
            prompt_service: Injected ``PromptService`` used to build
                the report prompt. ``ReportService`` never renders
                prompt text itself.
            gemini_service: Injected ``GeminiService`` used to
                generate the report text. ``ReportService`` never
                touches the Gemini SDK directly.
            settings: Optional injected settings object (defaults to
                the project-wide ``config.settings.settings``
                singleton). Reserved for configuration not sourced
                from ``thresholds.yaml`` via ``question_repository``;
                not required for current functionality.
            logger: Optional injected logger.

        Raises:
            ReportConfigurationError: If the reporting reference
                configuration (as already loaded by
                ``question_repository`` from ``thresholds.yaml`` /
                ``competencies.yaml``) is missing required sections.
        """
        self.question_repository = question_repository
        self.prompt_service = prompt_service
        self.gemini_service = gemini_service
        self.settings = settings if settings is not None else default_settings
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._cache_lock = threading.RLock()
        self._configuration_cache: dict[str, Any] = {}
        self._competencies_cache: dict[str, dict[str, Any]] = {}

        self._stats_lock = threading.RLock()
        self._statistics: dict[str, Any] = {
            "reports_generated": 0,
            "successful_reports": 0,
            "failed_reports": 0,
            "average_generation_time": 0.0,
            "average_score": 0.0,
            "decision_distribution": {},
        }

        self._refresh_configuration_cache()

    # =======================================================
    # Configuration
    # =======================================================

    def _refresh_configuration_cache(self) -> None:
        """
        Populate the thresholds/competency configuration cache from
        ``question_repository.reference_data`` without re-parsing any
        YAML file.

        Raises:
            ReportConfigurationError: Required sections are missing
                from the loaded ``thresholds.yaml`` /
                ``competencies.yaml`` reference data.
        """
        thresholds = dict(
            self.question_repository.reference_data.get("thresholds", {})
        )

        decision_thresholds = thresholds.get("decision_thresholds")
        if not isinstance(decision_thresholds, Mapping) or not decision_thresholds:
            raise ReportConfigurationError(
                "thresholds.yaml is missing the 'decision_thresholds' section",
                config_key="decision_thresholds",
            )

        confidence_scoring = thresholds.get("confidence_scoring")
        if not isinstance(confidence_scoring, Mapping) or not isinstance(
            confidence_scoring.get("confidence_levels"), Mapping
        ):
            raise ReportConfigurationError(
                "thresholds.yaml is missing the "
                "'confidence_scoring.confidence_levels' section",
                config_key="confidence_scoring.confidence_levels",
            )

        competency_scoring = thresholds.get("competency_scoring")
        if not isinstance(competency_scoring, Mapping):
            raise ReportConfigurationError(
                "thresholds.yaml is missing the 'competency_scoring' section",
                config_key="competency_scoring",
            )

        answer_scoring = thresholds.get("answer_scoring")
        if not isinstance(answer_scoring, Mapping):
            raise ReportConfigurationError(
                "thresholds.yaml is missing the 'answer_scoring' section",
                config_key="answer_scoring",
            )

        competencies_data = self.question_repository.reference_data.get(
            "competencies", {}
        )
        raw_competencies = competencies_data.get("competencies")
        if not isinstance(raw_competencies, list) or not raw_competencies:
            raise ReportConfigurationError(
                "competencies.yaml has no 'competencies' list configured",
                config_key="competencies",
            )

        competencies: dict[str, dict[str, Any]] = {
            str(entry["id"]): dict(entry)
            for entry in raw_competencies
            if isinstance(entry, Mapping) and entry.get("id")
        }
        if not competencies:
            raise ReportConfigurationError(
                "competencies.yaml produced no usable competency entries",
                config_key="competencies",
            )

        with self._cache_lock:
            self._configuration_cache = {"thresholds": thresholds}
            self._competencies_cache = competencies

    def reload_configuration(self) -> None:
        """
        Reload threshold and competency configuration from the
        question bank's reference data.

        Does not reload the question bank itself -- callers that need
        that should reload ``question_repository`` directly, then call
        this method to pick up any resulting change to
        ``thresholds.yaml`` / ``competencies.yaml``.

        Raises:
            ReportConfigurationError: Required sections are missing
                from the reloaded reference configuration.
        """
        self.logger.info("Reloading ReportService configuration")
        self._refresh_configuration_cache()

    def _get_thresholds(self) -> Mapping[str, Any]:
        with self._cache_lock:
            return dict(self._configuration_cache.get("thresholds", {}))

    def _get_decision_thresholds(self) -> Mapping[str, Any]:
        return self._get_thresholds().get("decision_thresholds", {})

    def _get_confidence_levels(self) -> Mapping[str, Any]:
        return (
            self._get_thresholds()
            .get("confidence_scoring", {})
            .get("confidence_levels", {})
        )

    def _get_score_bounds(self) -> tuple[int, int]:
        answer_scoring = self._get_thresholds().get("answer_scoring", {})
        minimum = answer_scoring.get("minimum_score")
        maximum = answer_scoring.get("maximum_score")
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise ReportConfigurationError(
                "thresholds.yaml answer_scoring.minimum_score / "
                "maximum_score must both be configured integers",
                config_key="answer_scoring.minimum_score/maximum_score",
            )
        return minimum, maximum

    def _get_overall_score_bounds(self) -> tuple[float, float]:
        competency_scoring = self._get_thresholds().get("competency_scoring", {})
        minimum = competency_scoring.get("minimum_score", 0)
        maximum = competency_scoring.get("maximum_score", 100)
        return float(minimum), float(maximum)

    def _get_competency_config(self, competency_id: str) -> Mapping[str, Any]:
        with self._cache_lock:
            competency = self._competencies_cache.get(competency_id)
        if competency is None:
            raise ReportConfigurationError(
                f"Unknown competency id: {competency_id}",
                config_key=f"competencies.{competency_id}",
            )
        return competency

    # =======================================================
    # Public API -- Report Generation
    # =======================================================

    def generate_report(
        self,
        session: InterviewSession,
        **gemini_overrides: Any,
    ) -> ReportResult:
        """
        Generate the final hiring report for a completed interview.

        Orchestrates the full reporting flow: validates the session,
        aggregates every ``EvaluationResult`` into per-competency
        scores, computes the overall score and confidence, determines
        the hiring decision, builds the report prompt via
        ``PromptService``, generates the report text via
        ``GeminiService``, validates the result, and returns a
        populated ``ReportResult``.

        Args:
            session: An ``InterviewSession`` with a non-empty
                ``evaluation_history``. Its lifecycle status is not
                enforced here (that is ``InterviewService``'s
                responsibility) -- this method only requires that
                evaluations exist to aggregate.
            **gemini_overrides: Forwarded to
                ``GeminiService.generate_text()`` (e.g.
                ``temperature=0``) to override per-call generation
                defaults.

        Returns:
            A populated ``ReportResult``.

        Raises:
            ReportValidationError: ``session`` has no evaluations to
                aggregate, or the generated report fails validation.
            ReportConfigurationError: Required threshold/competency
                configuration is unusable.
            ReportPromptError: ``PromptService`` failed to build the
                report prompt.
            ReportAIError: ``GeminiService`` failed to generate the
                report text.
        """
        request_id = self._make_request_id()
        start = time.monotonic()

        try:
            self._validate_session(session)

            competency_scores = self.calculate_competency_scores(session)
            overall_score = self.calculate_overall_score(competency_scores)
            decision = self.determine_decision(overall_score)
            confidence_level, confidence_percentage = self._aggregate_confidence(
                session
            )
            strengths = self._aggregate_strengths(session)
            weaknesses = self._aggregate_weaknesses(session)
            recommendations = self._generate_recommendations(weaknesses)

            competency_summary_text = self._format_competency_summary(
                competency_scores
            )

            prompt_text = self._build_prompt(
                session=session,
                competency_scores=competency_scores,
                overall_score=overall_score,
                decision=decision,
                confidence_level=confidence_level,
                confidence_percentage=confidence_percentage,
                strengths=strengths,
                weaknesses=weaknesses,
                request_id=request_id,
            )

            report_markdown = self._call_gemini(
                prompt_text, request_id=request_id, **gemini_overrides
            )

            self.validate_report(
                overall_score=overall_score,
                decision=decision,
                competency_scores=competency_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                confidence_level=confidence_level,
                confidence_percentage=confidence_percentage,
                report_markdown=report_markdown,
            )

            minimum_overall, maximum_overall = self._get_overall_score_bounds()
            overall_percentage = (
                (overall_score / maximum_overall) * 100
                if maximum_overall
                else overall_score
            )

            result = ReportResult(
                candidate_name=str(session.candidate.get("name", "")),
                overall_score=overall_score,
                overall_percentage=round(overall_percentage, 2),
                decision=decision,
                confidence_level=confidence_level,
                confidence_percentage=confidence_percentage,
                competency_scores=tuple(competency_scores),
                competency_summary=competency_summary_text,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                report_markdown=report_markdown,
                generated_at=datetime.now(timezone.utc),
                request_id=request_id,
                metadata={
                    "session_id": session.session_id,
                    "questions_evaluated": len(session.evaluation_history),
                    "competencies_assessed": len(competency_scores),
                },
            )

        except ReportServiceError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_failure_statistics()
            self.logger.warning(
                "report.failed",
                extra={
                    "request_id": request_id,
                    "session_id": getattr(session, "session_id", None),
                    "candidate": (
                        session.candidate.get("name")
                        if isinstance(session.candidate, Mapping)
                        else None
                    ),
                    "error_type": type(exc).__name__,
                    "error": exc.message,
                    "latency_ms": latency_ms,
                },
            )
            raise

        latency_ms = (time.monotonic() - start) * 1000
        self._record_success_statistics(
            overall_score=overall_score,
            decision=decision,
            latency_ms=latency_ms,
        )
        self.logger.info(
            "report.success",
            extra={
                "request_id": request_id,
                "session_id": session.session_id,
                "candidate": result.candidate_name,
                "overall_score": result.overall_score,
                "decision": result.decision,
                "latency_ms": latency_ms,
                "report_length": len(result.report_markdown),
            },
        )
        return result

    def calculate_competency_scores(
        self, session: InterviewSession
    ) -> list[CompetencyScoreSummary]:
        """
        Aggregate every ``EvaluationResult`` in ``session`` grouped by
        competency.

        Args:
            session: The interview session whose
                ``evaluation_history`` is aggregated. Evaluations
                missing a ``metadata['competency']`` value are
                skipped, since they cannot be attributed to any
                competency.

        Returns:
            One ``CompetencyScoreSummary`` per competency present in
            ``session.evaluation_history``, sorted by competency id.

        Raises:
            ReportValidationError: ``session.evaluation_history`` is
                empty.
            ReportConfigurationError: A competency referenced by an
                evaluation is not present in the loaded
                ``competencies.yaml`` reference data.
        """
        self._validate_session(session)

        grouped: dict[str, list[EvaluationResult]] = {}
        for result in session.evaluation_history:
            competency = result.metadata.get("competency")
            if not competency:
                continue
            grouped.setdefault(str(competency), []).append(result)

        minimum_score, maximum_score = self._get_score_bounds()
        score_range = maximum_score - minimum_score or 1

        summaries: list[CompetencyScoreSummary] = []
        for competency_id in sorted(grouped):
            results = grouped[competency_id]
            competency_cfg = self._get_competency_config(competency_id)

            average_score = statistics.fmean(result.score for result in results)
            average_confidence = statistics.fmean(
                result.confidence for result in results
            )
            percentage = ((average_score - minimum_score) / score_range) * 100

            summaries.append(
                CompetencyScoreSummary(
                    competency=competency_id,
                    competency_name=str(
                        competency_cfg.get("name", competency_id)
                    ),
                    average_score=round(average_score, 2),
                    percentage=round(percentage, 2),
                    average_confidence=round(average_confidence, 2),
                    question_count=len(results),
                    strengths=self._deduplicate_preserve_order(
                        item for result in results for item in result.strengths
                    ),
                    weaknesses=self._deduplicate_preserve_order(
                        item for result in results for item in result.weaknesses
                    ),
                )
            )

        return summaries

    def calculate_overall_score(
        self, competency_scores: Sequence[CompetencyScoreSummary]
    ) -> float:
        """
        Compute the weighted overall interview score from
        per-competency aggregates.

        Weighted by each competency's configured
        ``evaluation_weight`` in ``competencies.yaml``, renormalized
        to the competencies actually present in ``competency_scores``
        so an interview that only covered a subset of competencies is
        not penalized for competencies it never assessed.

        Args:
            competency_scores: Per-competency aggregates, as returned
                by ``calculate_competency_scores``.

        Returns:
            The weighted overall score, 0-100, rounded to 2 decimal
            places. ``0.0`` if ``competency_scores`` is empty.

        Raises:
            ReportConfigurationError: A competency in
                ``competency_scores`` is not present in the loaded
                ``competencies.yaml`` reference data.
        """
        if not competency_scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for summary in competency_scores:
            competency_cfg = self._get_competency_config(summary.competency)
            weight = float(competency_cfg.get("evaluation_weight", 0))
            weighted_sum += summary.percentage * weight
            total_weight += weight

        if total_weight == 0:
            return round(
                statistics.fmean(summary.percentage for summary in competency_scores),
                2,
            )
        return round(weighted_sum / total_weight, 2)

    def determine_decision(self, overall_score: float) -> str:
        """
        Determine the hiring decision for an overall score.

        Reads ``thresholds.yaml`` ``decision_thresholds`` exclusively
        -- the set of decisions (e.g. ``SELECT`` / ``HOLD`` /
        ``REJECT``) and their score bands are entirely defined by
        configuration, never hardcoded here.

        Args:
            overall_score: The weighted overall score, 0-100, as
                returned by ``calculate_overall_score``.

        Returns:
            The decision label whose configured
            ``minimum_score``/``maximum_score`` band contains
            ``overall_score``. If no configured band contains the
            score (a malformed configuration), falls back to the
            decision with the lowest configured ``minimum_score``.

        Raises:
            ReportConfigurationError: ``decision_thresholds`` is not
                configured in ``thresholds.yaml``.
        """
        decision_thresholds = self._get_decision_thresholds()
        if not decision_thresholds:
            raise ReportConfigurationError(
                "thresholds.yaml is missing 'decision_thresholds'",
                config_key="decision_thresholds",
            )

        for decision, bounds in decision_thresholds.items():
            minimum = bounds.get("minimum_score", 0)
            maximum = bounds.get("maximum_score", 100)
            if minimum <= overall_score <= maximum:
                return str(decision)

        return str(
            min(
                decision_thresholds,
                key=lambda d: decision_thresholds[d].get("minimum_score", 0),
            )
        )

    def validate_report(
        self,
        overall_score: float,
        decision: str,
        competency_scores: Sequence[CompetencyScoreSummary],
        strengths: Sequence[str],
        weaknesses: Sequence[str],
        confidence_level: str,
        confidence_percentage: float,
        report_markdown: str,
    ) -> bool:
        """
        Validate every aggregate and the generated Markdown report
        before packaging a ``ReportResult``.

        Args:
            overall_score: Weighted overall score to validate against
                configured bounds.
            decision: Decision label to validate against configured
                ``decision_thresholds``.
            competency_scores: Per-competency aggregates; must be
                non-empty.
            strengths: Merged strengths list.
            weaknesses: Merged weaknesses list.
            confidence_level: Confidence label to validate against
                configured ``confidence_scoring.confidence_levels``.
            confidence_percentage: Confidence percentage, must be
                0-100.
            report_markdown: The generated report text to validate.

        Returns:
            ``True`` if every check passes.

        Raises:
            ReportValidationError: Any check fails -- an out-of-range
                score, an unknown decision, an unknown confidence
                level, an out-of-range confidence percentage, empty
                ``competency_scores``, a report missing required
                headings (or with headings out of order), a report
                below the minimum length, or a report containing
                placeholder text.
        """
        minimum_overall, maximum_overall = self._get_overall_score_bounds()
        if not (minimum_overall <= overall_score <= maximum_overall):
            raise ReportValidationError(
                f"overall_score {overall_score} is outside configured bounds "
                f"[{minimum_overall}, {maximum_overall}]",
                field="overall_score",
            )

        decision_thresholds = self._get_decision_thresholds()
        if decision not in decision_thresholds:
            raise ReportValidationError(
                f"Unknown decision value: {decision}", field="decision"
            )

        if not competency_scores:
            raise ReportValidationError(
                "competency_scores must be a non-empty sequence",
                field="competency_scores",
            )

        confidence_levels = self._get_confidence_levels()
        if confidence_level not in confidence_levels:
            raise ReportValidationError(
                f"Unknown confidence_level value: {confidence_level}",
                field="confidence_level",
            )
        if not (0 <= confidence_percentage <= 100):
            raise ReportValidationError(
                f"confidence_percentage {confidence_percentage} is outside "
                "bounds [0, 100]",
                field="confidence_percentage",
            )

        self._validate_report_markdown(report_markdown)

        return True

    def _validate_report_markdown(self, report_markdown: str) -> None:
        """
        Raises:
            ReportValidationError: ``report_markdown`` is empty, below
                the minimum length, missing a required heading, has
                headings out of order, or contains placeholder text.
        """
        if not isinstance(report_markdown, str) or not report_markdown.strip():
            raise ReportValidationError(
                "report_markdown must be a non-empty string",
                field="report_markdown",
            )

        if len(report_markdown) < _MIN_REPORT_LENGTH_CHARS:
            raise ReportValidationError(
                f"report_markdown is {len(report_markdown)} characters, "
                f"below the minimum of {_MIN_REPORT_LENGTH_CHARS}",
                field="report_markdown",
            )

        last_index = -1
        for heading in _REQUIRED_REPORT_HEADINGS:
            index = report_markdown.find(heading)
            if index == -1:
                raise ReportValidationError(
                    f"report_markdown is missing required heading: {heading}",
                    field="report_markdown",
                )
            if index < last_index:
                raise ReportValidationError(
                    f"report_markdown heading out of order: {heading}",
                    field="report_markdown",
                )
            last_index = index

        lowered = report_markdown.lower()
        for marker in _PLACEHOLDER_MARKERS:
            if marker.lower() in lowered:
                raise ReportValidationError(
                    f"report_markdown contains placeholder text: {marker}",
                    field="report_markdown",
                )

    # =======================================================
    # Public API -- Statistics / Health
    # =======================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Return aggregated ``ReportService`` usage statistics.

        Returns:
            A dictionary with ``reports_generated``,
            ``successful_reports``, ``failed_reports``,
            ``average_generation_time`` (seconds),
            ``average_score`` (mean overall score across successful
            reports), and ``decision_distribution`` (count per
            decision label across successful reports).
        """
        with self._stats_lock:
            statistics_copy = dict(self._statistics)
            statistics_copy["decision_distribution"] = dict(
                self._statistics["decision_distribution"]
            )
            return statistics_copy

    def health_check(self) -> bool:
        """
        Lightweight liveness check for every collaborator this service
        depends on.

        Returns:
            ``True`` if the question repository has a loaded question
            bank and ``GeminiService`` reports itself healthy.
            ``False`` if any check raises or reports unhealthy -- this
            method never lets an exception propagate to a caller
            polling it.
        """
        try:
            self.question_repository.get_total_questions()
            return bool(self.gemini_service.health_check())
        except Exception as exc:  # noqa: BLE001 - liveness probe, never raises
            self.logger.warning(
                "report.health_check.failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return False

    # =======================================================
    # Internal Helpers -- Validation
    # =======================================================

    @staticmethod
    def _validate_session(session: InterviewSession) -> None:
        """
        Raises:
            ReportValidationError: ``session`` is not an
                ``InterviewSession``, or has no evaluations to
                aggregate.
        """
        if not isinstance(session, InterviewSession):
            raise ReportValidationError(
                "session must be an InterviewSession", field="session"
            )
        if not session.evaluation_history:
            raise ReportValidationError(
                "session has no evaluation history to aggregate; "
                "a report cannot be generated for an interview with no "
                "answered questions",
                field="evaluation_history",
            )

    # =======================================================
    # Internal Helpers -- Aggregation
    # =======================================================

    @staticmethod
    def _deduplicate_preserve_order(items: Any) -> tuple[str, ...]:
        """Deduplicate a sequence of strings while preserving first-seen order."""
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return tuple(deduped)

    def _aggregate_strengths(self, session: InterviewSession) -> tuple[str, ...]:
        """Merge strengths from every evaluation in the interview, deduplicated,
        order preserved."""
        return self._deduplicate_preserve_order(
            item
            for result in session.evaluation_history
            for item in result.strengths
        )

    def _aggregate_weaknesses(self, session: InterviewSession) -> tuple[str, ...]:
        """Merge weaknesses from every evaluation in the interview, deduplicated,
        order preserved."""
        return self._deduplicate_preserve_order(
            item
            for result in session.evaluation_history
            for item in result.weaknesses
        )

    def _generate_recommendations(
        self, weaknesses: Sequence[str]
    ) -> tuple[str, ...]:
        """
        Generate actionable development recommendations, derived only
        from identified weaknesses -- never fabricated beyond what the
        evaluation data supports.

        Each weakness maps to exactly one recommendation, in the same
        order, so a hiring manager can trace every recommendation back
        to the observation that produced it.
        """
        return tuple(
            _GENERIC_RECOMMENDATION_TEMPLATE.format(weakness=weakness)
            for weakness in weaknesses
        )

    def _aggregate_confidence(
        self, session: InterviewSession
    ) -> tuple[str, float]:
        """
        Compute mean confidence across every evaluation in the
        interview and resolve it to a configured confidence label.

        Raises:
            ReportConfigurationError: No confidence level configuration
                could resolve any label (e.g. all
                ``minimum_percentage`` values malformed).
        """
        confidence_percentage = round(
            statistics.fmean(
                result.confidence for result in session.evaluation_history
            ),
            2,
        )

        levels = self._get_confidence_levels()
        confidence_level = "LOW"
        best_minimum = -1.0
        for label, bounds in levels.items():
            minimum = float(bounds.get("minimum_percentage", 0))
            if confidence_percentage >= minimum and minimum >= best_minimum:
                confidence_level = str(label)
                best_minimum = minimum

        return confidence_level, confidence_percentage

    def _format_competency_summary(
        self, competency_scores: Sequence[CompetencyScoreSummary]
    ) -> str:
        """
        Render ``competency_scores`` into the exact text block passed
        to ``PromptService.build_report_prompt`` as
        ``competency_summaries``, retained on ``ReportResult`` for
        auditability.
        """
        lines: list[str] = []
        for summary in competency_scores:
            lines.append(
                f"{summary.competency_name}: {summary.percentage}% "
                f"[{summary.question_count} questions]"
            )
        return "\n".join(lines) if lines else "None provided."

    # =======================================================
    # Internal Helpers -- Collaborators
    # =======================================================

    def _build_prompt(
        self,
        session: InterviewSession,
        competency_scores: Sequence[CompetencyScoreSummary],
        overall_score: float,
        decision: str,
        confidence_level: str,
        confidence_percentage: float,
        strengths: Sequence[str],
        weaknesses: Sequence[str],
        request_id: str,
    ) -> str:
        """
        Build the report prompt via ``PromptService``.

        Raises:
            ReportPromptError: ``PromptService`` failed to build the
                prompt.
        """
        competency_summaries_payload = [
            {
                "competency": summary.competency_name,
                "competency_percentage": summary.percentage,
                "weight_pct": self._get_competency_config(
                    summary.competency
                ).get("evaluation_weight"),
                "question_count": summary.question_count,
            }
            for summary in competency_scores
        ]

        try:
            prompt_result = self.prompt_service.build_report_prompt(
                candidate=session.candidate,
                competency_summaries=competency_summaries_payload,
                overall_score=overall_score,
                decision=decision,
                confidence_level=confidence_level,
                confidence_percentage=confidence_percentage,
                strengths=list(strengths),
                weaknesses=list(weaknesses),
            )
        except PromptServiceError as exc:
            raise ReportPromptError(
                f"Failed to build report prompt: {exc}", request_id=request_id
            ) from exc

        return prompt_result.prompt

    def _call_gemini(
        self,
        prompt: str,
        request_id: str,
        **generation_kwargs: Any,
    ) -> str:
        """
        Call ``GeminiService.generate_text()`` and return the report
        text.

        Raises:
            ReportAIError: ``GeminiService`` failed for any reason
                (timeout, rate limit, safety block, malformed
                response, or configuration problem).
        """
        try:
            return self.gemini_service.generate_text(prompt, **generation_kwargs)
        except GeminiServiceError as exc:
            raise self._translate_gemini_exception(exc, request_id) from exc

    @staticmethod
    def _translate_gemini_exception(
        exc: GeminiServiceError, request_id: str | None = None
    ) -> ReportAIError:
        """Translate any ``GeminiServiceError`` into ``ReportAIError``."""
        return ReportAIError(
            f"Gemini report generation failed: {exc}",
            request_id=request_id or getattr(exc, "request_id", None),
        )

    # =======================================================
    # Internal Helpers -- Statistics
    # =======================================================

    def _record_success_statistics(
        self, overall_score: float, decision: str, latency_ms: float
    ) -> None:
        with self._stats_lock:
            total_before = self._statistics["reports_generated"]
            self._statistics["reports_generated"] = total_before + 1
            self._statistics["successful_reports"] += 1

            successful_before = self._statistics["successful_reports"] - 1
            previous_average_time = self._statistics["average_generation_time"]
            self._statistics["average_generation_time"] = (
                (previous_average_time * successful_before) + (latency_ms / 1000.0)
            ) / self._statistics["successful_reports"]

            previous_average_score = self._statistics["average_score"]
            self._statistics["average_score"] = (
                (previous_average_score * successful_before) + overall_score
            ) / self._statistics["successful_reports"]

            distribution: dict[str, int] = self._statistics["decision_distribution"]
            distribution[decision] = distribution.get(decision, 0) + 1

    def _record_failure_statistics(self) -> None:
        with self._stats_lock:
            self._statistics["reports_generated"] += 1
            self._statistics["failed_reports"] += 1

    # =======================================================
    # Internal Helpers -- Misc
    # =======================================================

    @staticmethod
    def _make_request_id() -> str:
        """Generate a short correlation id for a single report-generation call."""
        return uuid.uuid4().hex[:12]