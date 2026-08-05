"""
tests/test_report_service.py

Comprehensive pytest suite for ReportService.

Testing strategy
-----------------
* Every collaborator (QuestionRepository, PromptService, GeminiService)
  is a MagicMock/stub built from fixtures below -- ReportService is
  tested in isolation, never against real Gemini calls or a real
  question bank.
* ``question_repository.reference_data`` is populated with minimal but
  structurally valid ``thresholds.yaml`` / ``competencies.yaml``
  fixtures mirroring the real files' shape, since ReportService reads
  that structure directly rather than re-parsing YAML.
* EvaluationResult objects are constructed directly (not through
  EvaluationService) so each test controls exactly the
  score/confidence/strengths/weaknesses it wants to drive aggregation
  logic deterministically.
* InterviewSession objects are constructed directly with a populated
  ``evaluation_history`` -- ReportService never mutates or drives
  interview flow, so no InterviewService involvement is needed.
* Tests are grouped by public method, with additional groups for
  configuration loading, exception translation, statistics, and
  thread-safety bookkeeping.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

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
from services.gemini_service import GeminiService
from services.prompt_service import PromptResult, PromptService, PromptServiceError
from services.report_service import ReportService


# ============================================================
# Fixtures -- reference configuration
# ============================================================


@pytest.fixture
def competencies_config() -> dict:
    return {
        "competencies": [
            {
                "id": "C1",
                "name": "SQL & Query Optimization",
                "evaluation_weight": 15,
                "minimum_questions": 1,
            },
            {
                "id": "C2",
                "name": "Data Modeling & Warehousing",
                "evaluation_weight": 15,
                "minimum_questions": 1,
            },
        ]
    }


@pytest.fixture
def thresholds_config() -> dict:
    return {
        "answer_scoring": {
            "minimum_score": 0,
            "maximum_score": 5,
        },
        "competency_scoring": {
            "minimum_score": 0,
            "maximum_score": 100,
        },
        "decision_thresholds": {
            "SELECT": {"minimum_score": 75, "maximum_score": 100},
            "HOLD": {"minimum_score": 60, "maximum_score": 74},
            "REJECT": {"minimum_score": 0, "maximum_score": 59},
        },
        "confidence_scoring": {
            "confidence_levels": {
                "HIGH": {"minimum_percentage": 85},
                "MEDIUM": {"minimum_percentage": 70},
                "LOW": {"minimum_percentage": 0},
            }
        },
    }


# ============================================================
# Fixtures -- collaborators
# ============================================================


@pytest.fixture
def question_repository(competencies_config, thresholds_config):
    repo = MagicMock()
    repo.reference_data = {
        "competencies": competencies_config,
        "thresholds": thresholds_config,
    }
    repo.get_total_questions.return_value = 10
    return repo


@pytest.fixture
def prompt_service():
    service = MagicMock(spec=PromptService)

    def _report_prompt(*args, **kwargs):
        return PromptResult(
            prompt="REPORT PROMPT TEXT",
            prompt_type="REPORT",
            version="1.0",
            estimated_tokens=20,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )

    service.build_report_prompt.side_effect = _report_prompt
    service.reload_configuration.return_value = None
    return service


VALID_REPORT_MARKDOWN = "\n\n".join(
    [
        "## Executive Summary\nCandidate performed well overall.",
        "## Overall Assessment\nStrong technical performance across the board.",
        "## Competency Breakdown\nSQL: 90%. Data Modeling: 85%.",
        "## Technical Strengths\nStrong grasp of joins and indexing.",
        "## Technical Weaknesses\nSome gaps in edge case handling.",
        "## Communication Assessment\nClear and structured responses.",
        "## Evidence-Based Justification\nScores support the decision reached.",
        "## Hiring Recommendation\nRecommend proceeding per the decision.",
        "## Development Plan\nFocus on edge cases and null handling.",
        "## Confidence Explanation\nConfidence is high given consistent scores.",
    ]
)


@pytest.fixture
def gemini_service():
    service = MagicMock(spec=GeminiService)
    service.generate_text.return_value = VALID_REPORT_MARKDOWN
    service.health_check.return_value = True
    return service


@pytest.fixture
def report_service(question_repository, prompt_service, gemini_service):
    return ReportService(
        question_repository=question_repository,
        prompt_service=prompt_service,
        gemini_service=gemini_service,
    )


def make_evaluation_result(
    score: int = 4,
    confidence: int = 90,
    strengths: list[str] | None = None,
    weaknesses: list[str] | None = None,
    competency: str = "C1",
    question_id: str = "Q1",
) -> EvaluationResult:
    return EvaluationResult(
        score=score,
        confidence=confidence,
        strengths=strengths if strengths is not None else ["good use of joins"],
        weaknesses=weaknesses if weaknesses is not None else ["missed edge case"],
        missing_concepts=[],
        evidence=["mentioned LEFT JOIN correctly"],
        recommendation="Solid grasp of joins.",
        needs_followup=False,
        next_action="ADVANCE",
        missing_concept=None,
        followup_reason=None,
        request_id="req123",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "question_id": question_id,
            "competency": competency,
            "stage": "S1",
            "difficulty": "Medium",
        },
    )


@pytest.fixture
def candidate() -> dict:
    return {"name": "Prathiksha", "experience_years": 2, "target_role": "Data Engineer"}


@pytest.fixture
def completed_session(candidate) -> InterviewSession:
    session = InterviewSession(
        session_id="sess-1",
        candidate=candidate,
        started_at=datetime.now(timezone.utc),
    )
    session.evaluation_history = [
        make_evaluation_result(
            score=4,
            confidence=90,
            strengths=["good use of joins"],
            weaknesses=["missed null handling"],
            competency="C1",
            question_id="Q1",
        ),
        make_evaluation_result(
            score=5,
            confidence=95,
            strengths=["excellent indexing knowledge", "good use of joins"],
            weaknesses=[],
            competency="C1",
            question_id="Q2",
        ),
        make_evaluation_result(
            score=3,
            confidence=80,
            strengths=["understands star schema"],
            weaknesses=["missed null handling", "weak on SCD types"],
            competency="C2",
            question_id="Q3",
        ),
    ]
    return session


# ============================================================
# Construction / configuration
# ============================================================


class TestConstruction:
    def test_successful_construction_loads_configuration_cache(self, report_service):
        assert "C1" in report_service._competencies_cache
        assert "decision_thresholds" in report_service._configuration_cache[
            "thresholds"
        ]

    def test_missing_decision_thresholds_raises_configuration_error(
        self, competencies_config, prompt_service, gemini_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "competencies": competencies_config,
            "thresholds": {
                "answer_scoring": {"minimum_score": 0, "maximum_score": 5},
                "competency_scoring": {},
                "confidence_scoring": {"confidence_levels": {"LOW": {}}},
            },
        }
        with pytest.raises(ReportConfigurationError):
            ReportService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
            )

    def test_missing_confidence_levels_raises_configuration_error(
        self, competencies_config, prompt_service, gemini_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "competencies": competencies_config,
            "thresholds": {
                "answer_scoring": {"minimum_score": 0, "maximum_score": 5},
                "competency_scoring": {},
                "decision_thresholds": {"SELECT": {"minimum_score": 0}},
            },
        }
        with pytest.raises(ReportConfigurationError):
            ReportService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
            )

    def test_missing_competency_scoring_raises_configuration_error(
        self, competencies_config, prompt_service, gemini_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "competencies": competencies_config,
            "thresholds": {
                "answer_scoring": {"minimum_score": 0, "maximum_score": 5},
                "decision_thresholds": {"SELECT": {"minimum_score": 0}},
                "confidence_scoring": {"confidence_levels": {"LOW": {}}},
            },
        }
        with pytest.raises(ReportConfigurationError):
            ReportService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
            )

    def test_missing_answer_scoring_raises_configuration_error(
        self, competencies_config, prompt_service, gemini_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "competencies": competencies_config,
            "thresholds": {
                "competency_scoring": {},
                "decision_thresholds": {"SELECT": {"minimum_score": 0}},
                "confidence_scoring": {"confidence_levels": {"LOW": {}}},
            },
        }
        with pytest.raises(ReportConfigurationError):
            ReportService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
            )

    def test_missing_competencies_raises_configuration_error(
        self, thresholds_config, prompt_service, gemini_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "competencies": {"competencies": []},
            "thresholds": thresholds_config,
        }
        with pytest.raises(ReportConfigurationError):
            ReportService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
            )

    def test_reload_configuration_refreshes_cache(self, report_service):
        report_service.reload_configuration()
        assert "C1" in report_service._competencies_cache


# ============================================================
# calculate_competency_scores
# ============================================================


class TestCalculateCompetencyScores:
    def test_aggregates_by_competency(self, report_service, completed_session):
        summaries = report_service.calculate_competency_scores(completed_session)
        by_id = {summary.competency: summary for summary in summaries}

        assert set(by_id) == {"C1", "C2"}
        assert by_id["C1"].question_count == 2
        assert by_id["C2"].question_count == 1

    def test_average_score_computed_correctly(self, report_service, completed_session):
        summaries = report_service.calculate_competency_scores(completed_session)
        by_id = {summary.competency: summary for summary in summaries}
        # C1 scores: 4, 5 -> average 4.5
        assert by_id["C1"].average_score == 4.5

    def test_percentage_computed_from_score_bounds(
        self, report_service, completed_session
    ):
        summaries = report_service.calculate_competency_scores(completed_session)
        by_id = {summary.competency: summary for summary in summaries}
        # average_score=4.5, bounds 0-5 -> percentage = (4.5/5)*100 = 90.0
        assert by_id["C1"].percentage == 90.0

    def test_strengths_and_weaknesses_deduplicated(
        self, report_service, completed_session
    ):
        summaries = report_service.calculate_competency_scores(completed_session)
        by_id = {summary.competency: summary for summary in summaries}
        # "good use of joins" appears in both C1 evaluations
        assert by_id["C1"].strengths.count("good use of joins") == 1

    def test_competency_name_resolved_from_config(
        self, report_service, completed_session
    ):
        summaries = report_service.calculate_competency_scores(completed_session)
        by_id = {summary.competency: summary for summary in summaries}
        assert by_id["C1"].competency_name == "SQL & Query Optimization"

    def test_evaluations_missing_competency_are_skipped(
        self, report_service, completed_session
    ):
        stray = make_evaluation_result(score=2, competency="")
        stray.metadata["competency"] = None
        completed_session.evaluation_history.append(stray)

        summaries = report_service.calculate_competency_scores(completed_session)
        assert len(summaries) == 2  # still just C1, C2

    def test_empty_evaluation_history_raises_validation_error(
        self, report_service, candidate
    ):
        empty_session = InterviewSession(
            session_id="empty",
            candidate=candidate,
            started_at=datetime.now(timezone.utc),
        )
        with pytest.raises(ReportValidationError):
            report_service.calculate_competency_scores(empty_session)

    def test_unknown_competency_raises_configuration_error(
        self, report_service, completed_session
    ):
        completed_session.evaluation_history.append(
            make_evaluation_result(score=3, competency="C99")
        )
        with pytest.raises(ReportConfigurationError):
            report_service.calculate_competency_scores(completed_session)


# ============================================================
# calculate_overall_score
# ============================================================


class TestCalculateOverallScore:
    def test_weighted_average_across_competencies(
        self, report_service, completed_session
    ):
        summaries = report_service.calculate_competency_scores(completed_session)
        overall = report_service.calculate_overall_score(summaries)
        # C1: 90% weight 15, C2: (3/5)*100=60% weight 15 -> equal weights -> mean 75.0
        assert overall == 75.0

    def test_empty_competency_scores_returns_zero(self, report_service):
        assert report_service.calculate_overall_score([]) == 0.0

    def test_zero_total_weight_falls_back_to_unweighted_mean(
        self, report_service, question_repository
    ):
        question_repository.reference_data["competencies"]["competencies"][0][
            "evaluation_weight"
        ] = 0
        question_repository.reference_data["competencies"]["competencies"][1][
            "evaluation_weight"
        ] = 0
        svc = ReportService(
            question_repository=question_repository,
            prompt_service=report_service.prompt_service,
            gemini_service=report_service.gemini_service,
        )
        summaries = [
            CompetencyScoreSummary(
                competency="C1",
                competency_name="SQL",
                average_score=4.0,
                percentage=80.0,
                average_confidence=90.0,
                question_count=1,
                strengths=(),
                weaknesses=(),
            ),
            CompetencyScoreSummary(
                competency="C2",
                competency_name="Modeling",
                average_score=3.0,
                percentage=60.0,
                average_confidence=80.0,
                question_count=1,
                strengths=(),
                weaknesses=(),
            ),
        ]
        assert svc.calculate_overall_score(summaries) == 70.0


# ============================================================
# determine_decision
# ============================================================


class TestDetermineDecision:
    def test_high_score_yields_select(self, report_service):
        assert report_service.determine_decision(90.0) == "SELECT"

    def test_mid_score_yields_hold(self, report_service):
        assert report_service.determine_decision(65.0) == "HOLD"

    def test_low_score_yields_reject(self, report_service):
        assert report_service.determine_decision(10.0) == "REJECT"

    def test_boundary_score_is_inclusive(self, report_service):
        assert report_service.determine_decision(75.0) == "SELECT"
        assert report_service.determine_decision(74.0) == "HOLD"

    def test_missing_decision_thresholds_raises_configuration_error(
        self, report_service
    ):
        report_service._configuration_cache["thresholds"]["decision_thresholds"] = {}
        with pytest.raises(ReportConfigurationError):
            report_service.determine_decision(50.0)


# ============================================================
# generate_report -- happy path
# ============================================================


class TestGenerateReportSuccess:
    def test_generate_report_returns_populated_result(
        self, report_service, completed_session
    ):
        result = report_service.generate_report(completed_session)

        assert isinstance(result, ReportResult)
        assert result.candidate_name == "Prathiksha"
        assert result.decision == "SELECT"  # overall 75.0 boundary -> SELECT actually
        assert result.report_markdown == VALID_REPORT_MARKDOWN
        assert len(result.competency_scores) == 2

    def test_generate_report_calls_prompt_service_with_aggregated_data(
        self, report_service, completed_session
    ):
        report_service.generate_report(completed_session)
        call_kwargs = report_service.prompt_service.build_report_prompt.call_args.kwargs
        assert call_kwargs["candidate"]["name"] == "Prathiksha"
        assert call_kwargs["decision"] in {"SELECT", "HOLD", "REJECT"}

    def test_generate_report_calls_gemini_generate_text(
        self, report_service, completed_session
    ):
        report_service.generate_report(completed_session)
        report_service.gemini_service.generate_text.assert_called_once()

    def test_generate_report_recommendations_derived_from_weaknesses(
        self, report_service, completed_session
    ):
        result = report_service.generate_report(completed_session)
        assert len(result.recommendations) == len(result.weaknesses)
        for weakness, recommendation in zip(result.weaknesses, result.recommendations):
            assert weakness in recommendation

    def test_generate_report_strengths_and_weaknesses_deduplicated(
        self, report_service, completed_session
    ):
        result = report_service.generate_report(completed_session)
        assert result.strengths.count("good use of joins") == 1
        assert result.weaknesses.count("missed null handling") == 1

    def test_generate_report_confidence_aggregated(
        self, report_service, completed_session
    ):
        result = report_service.generate_report(completed_session)
        # confidences: 90, 95, 80 -> mean = 88.33
        assert result.confidence_percentage == pytest.approx(88.33, abs=0.01)
        assert result.confidence_level == "HIGH"

    def test_generate_report_metadata_populated(
        self, report_service, completed_session
    ):
        result = report_service.generate_report(completed_session)
        assert result.metadata["session_id"] == "sess-1"
        assert result.metadata["questions_evaluated"] == 3
        assert result.metadata["competencies_assessed"] == 2


# ============================================================
# generate_report -- error paths
# ============================================================


class TestGenerateReportErrors:
    def test_generate_report_empty_session_raises_validation_error(
        self, report_service, candidate
    ):
        empty_session = InterviewSession(
            session_id="empty",
            candidate=candidate,
            started_at=datetime.now(timezone.utc),
        )
        with pytest.raises(ReportValidationError):
            report_service.generate_report(empty_session)

    def test_generate_report_translates_prompt_service_error(
        self, report_service, completed_session
    ):
        report_service.prompt_service.build_report_prompt.side_effect = (
            PromptServiceError("bad template")
        )
        with pytest.raises(ReportPromptError):
            report_service.generate_report(completed_session)

    def test_generate_report_translates_gemini_service_error(
        self, report_service, completed_session
    ):
        report_service.gemini_service.generate_text.side_effect = (
            GeminiServiceError("gemini down")
        )
        with pytest.raises(ReportAIError):
            report_service.generate_report(completed_session)

    def test_generate_report_invalid_markdown_raises_validation_error(
        self, report_service, completed_session
    ):
        report_service.gemini_service.generate_text.return_value = "too short"
        with pytest.raises(ReportValidationError):
            report_service.generate_report(completed_session)

    def test_generate_report_missing_heading_raises_validation_error(
        self, report_service, completed_session
    ):
        broken_markdown = VALID_REPORT_MARKDOWN.replace(
            "## Confidence Explanation", "## Something Else"
        )
        report_service.gemini_service.generate_text.return_value = broken_markdown
        with pytest.raises(ReportValidationError):
            report_service.generate_report(completed_session)

    def test_generate_report_placeholder_text_raises_validation_error(
        self, report_service, completed_session
    ):
        polluted = VALID_REPORT_MARKDOWN + "\n\nTBD"
        report_service.gemini_service.generate_text.return_value = polluted
        with pytest.raises(ReportValidationError):
            report_service.generate_report(completed_session)

    def test_generate_report_headings_out_of_order_raises_validation_error(
        self, report_service, completed_session
    ):
        # Swap two heading blocks to break required order.
        parts = VALID_REPORT_MARKDOWN.split("\n\n")
        parts[0], parts[1] = parts[1], parts[0]
        reordered = "\n\n".join(parts)
        report_service.gemini_service.generate_text.return_value = reordered
        with pytest.raises(ReportValidationError):
            report_service.generate_report(completed_session)


# ============================================================
# validate_report (direct calls)
# ============================================================


class TestValidateReport:
    def _valid_kwargs(self, report_service, completed_session):
        summaries = report_service.calculate_competency_scores(completed_session)
        overall = report_service.calculate_overall_score(summaries)
        decision = report_service.determine_decision(overall)
        return {
            "overall_score": overall,
            "decision": decision,
            "competency_scores": summaries,
            "strengths": ["x"],
            "weaknesses": ["y"],
            "confidence_level": "HIGH",
            "confidence_percentage": 90.0,
            "report_markdown": VALID_REPORT_MARKDOWN,
        }

    def test_valid_report_returns_true(self, report_service, completed_session):
        kwargs = self._valid_kwargs(report_service, completed_session)
        assert report_service.validate_report(**kwargs) is True

    def test_out_of_range_overall_score_raises(self, report_service, completed_session):
        kwargs = self._valid_kwargs(report_service, completed_session)
        kwargs["overall_score"] = 150.0
        with pytest.raises(ReportValidationError):
            report_service.validate_report(**kwargs)

    def test_unknown_decision_raises(self, report_service, completed_session):
        kwargs = self._valid_kwargs(report_service, completed_session)
        kwargs["decision"] = "MAYBE"
        with pytest.raises(ReportValidationError):
            report_service.validate_report(**kwargs)

    def test_empty_competency_scores_raises(self, report_service, completed_session):
        kwargs = self._valid_kwargs(report_service, completed_session)
        kwargs["competency_scores"] = []
        with pytest.raises(ReportValidationError):
            report_service.validate_report(**kwargs)

    def test_unknown_confidence_level_raises(self, report_service, completed_session):
        kwargs = self._valid_kwargs(report_service, completed_session)
        kwargs["confidence_level"] = "EXTREME"
        with pytest.raises(ReportValidationError):
            report_service.validate_report(**kwargs)

    def test_out_of_range_confidence_percentage_raises(
        self, report_service, completed_session
    ):
        kwargs = self._valid_kwargs(report_service, completed_session)
        kwargs["confidence_percentage"] = 150.0
        with pytest.raises(ReportValidationError):
            report_service.validate_report(**kwargs)

    def test_empty_report_markdown_raises(self, report_service, completed_session):
        kwargs = self._valid_kwargs(report_service, completed_session)
        kwargs["report_markdown"] = ""
        with pytest.raises(ReportValidationError):
            report_service.validate_report(**kwargs)


# ============================================================
# Statistics
# ============================================================


class TestStatistics:
    def test_statistics_initial_state(self, report_service):
        stats = report_service.get_statistics()
        assert stats["reports_generated"] == 0
        assert stats["successful_reports"] == 0
        assert stats["failed_reports"] == 0
        assert stats["decision_distribution"] == {}

    def test_statistics_after_successful_report(
        self, report_service, completed_session
    ):
        result = report_service.generate_report(completed_session)
        stats = report_service.get_statistics()

        assert stats["reports_generated"] == 1
        assert stats["successful_reports"] == 1
        assert stats["failed_reports"] == 0
        assert stats["average_score"] == result.overall_score
        assert stats["decision_distribution"][result.decision] == 1
        assert stats["average_generation_time"] >= 0.0

    def test_statistics_after_failed_report(self, report_service, completed_session):
        report_service.gemini_service.generate_text.side_effect = (
            GeminiServiceError("boom")
        )
        with pytest.raises(ReportAIError):
            report_service.generate_report(completed_session)

        stats = report_service.get_statistics()
        assert stats["reports_generated"] == 1
        assert stats["failed_reports"] == 1
        assert stats["successful_reports"] == 0

    def test_statistics_return_a_copy_not_live_reference(self, report_service):
        stats = report_service.get_statistics()
        stats["reports_generated"] = 999
        stats["decision_distribution"]["SELECT"] = 999
        fresh = report_service.get_statistics()
        assert fresh["reports_generated"] == 0
        assert fresh["decision_distribution"] == {}

    def test_decision_distribution_accumulates_across_reports(
        self, report_service, completed_session
    ):
        report_service.generate_report(completed_session)
        report_service.generate_report(completed_session)
        stats = report_service.get_statistics()
        total_decisions = sum(stats["decision_distribution"].values())
        assert total_decisions == 2


# ============================================================
# Health check
# ============================================================


class TestHealthCheck:
    def test_health_check_true_when_all_collaborators_healthy(self, report_service):
        assert report_service.health_check() is True

    def test_health_check_false_when_gemini_unhealthy(self, report_service):
        report_service.gemini_service.health_check.return_value = False
        assert report_service.health_check() is False

    def test_health_check_false_on_repository_exception(self, report_service):
        report_service.question_repository.get_total_questions.side_effect = (
            RuntimeError("db down")
        )
        assert report_service.health_check() is False

    def test_health_check_never_raises(self, report_service):
        report_service.gemini_service.health_check.side_effect = RuntimeError(
            "unexpected"
        )
        try:
            result = report_service.health_check()
        except Exception as exc:  # pragma: no cover - defensive
            pytest.fail(f"health_check raised unexpectedly: {exc}")
        assert result is False


# ============================================================
# Exception hierarchy sanity checks
# ============================================================


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            ReportValidationError,
            ReportPromptError,
            ReportAIError,
            ReportConfigurationError,
        ],
    )
    def test_all_exceptions_are_report_service_errors(self, exc_cls):
        assert issubclass(exc_cls, ReportServiceError)

    def test_base_exception_carries_request_id(self):
        exc = ReportServiceError("boom", request_id="r1")
        assert exc.request_id == "r1"
        assert str(exc) == "boom"


# ============================================================
# Logging
# ============================================================


class TestLogging:
    def test_success_logs_include_expected_fields(
        self, report_service, completed_session, caplog
    ):
        import logging

        report_service.logger.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO):
            report_service.generate_report(completed_session)

        matching = [
            record for record in caplog.records if record.msg == "report.success"
        ]
        assert matching
        record = matching[0]
        assert hasattr(record, "request_id")
        assert hasattr(record, "candidate")
        assert hasattr(record, "overall_score")
        assert hasattr(record, "decision")
        assert hasattr(record, "latency_ms")
        assert hasattr(record, "report_length")

    def test_failure_logs_include_error_type(
        self, report_service, completed_session, caplog
    ):
        import logging

        report_service.gemini_service.generate_text.side_effect = (
            GeminiServiceError("boom")
        )
        report_service.logger.setLevel(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ReportAIError):
                report_service.generate_report(completed_session)

        matching = [
            record for record in caplog.records if record.msg == "report.failed"
        ]
        assert matching
        assert matching[0].error_type == "ReportAIError"


# ============================================================
# Thread safety (smoke test)
# ============================================================


class TestThreadSafety:
    def test_concurrent_report_generation_does_not_corrupt_statistics(
        self, report_service, candidate
    ):
        errors: list[Exception] = []

        def _generate(idx: int) -> None:
            try:
                session = InterviewSession(
                    session_id=f"sess-{idx}",
                    candidate=dict(candidate),
                    started_at=datetime.now(timezone.utc),
                )
                session.evaluation_history = [
                    make_evaluation_result(score=4, competency="C1"),
                    make_evaluation_result(score=3, competency="C2"),
                ]
                report_service.generate_report(session)
            except Exception as exc:  # pragma: no cover - fail via errors list
                errors.append(exc)

        threads = [threading.Thread(target=_generate, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        stats = report_service.get_statistics()
        assert stats["reports_generated"] == 10
        assert stats["successful_reports"] == 10