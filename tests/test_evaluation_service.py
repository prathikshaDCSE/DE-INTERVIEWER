"""
tests/test_evaluation_service.py

Unit tests for ``services.evaluation_service.EvaluationService``.

``QuestionRepository``, ``PromptService``, and ``GeminiService`` are
all mocked -- these tests exercise only ``EvaluationService``'s own
orchestration, validation, follow-up, and normalization logic.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from exceptions.evaluation_exceptions import (
    EvaluationAIError,
    EvaluationConfigurationError,
    EvaluationPromptError,
    EvaluationRepositoryError,
    EvaluationValidationError,
)
from exceptions.gemini_exceptions import GeminiTimeoutError, GeminiSafetyError
from models.evaluation_result import EvaluationResult
from repository.question_repository import QuestionBankError, QuestionNotFoundError
from services.evaluation_service import EvaluationService
from services.prompt_service import PromptRenderError

# ============================================================
# Fixtures
# ============================================================


THRESHOLDS = {
    "answer_scoring": {
        "minimum_score": 0,
        "maximum_score": 5,
        "score_scale": {
            "0": {"label": "No Answer", "description": "Blank."},
            "1": {"label": "Very Weak", "description": "Mostly wrong."},
            "2": {"label": "Partial", "description": "Gaps remain."},
            "3": {"label": "Meets Expectations", "description": "Acceptable."},
            "4": {"label": "Strong", "description": "Thorough."},
            "5": {"label": "Exceptional", "description": "Expert."},
        },
    },
    "adaptive_questioning": {
        "followup_threshold": {"score_less_than_or_equal": 2},
        "advance_threshold": {"score_greater_than_or_equal": 4},
        "maximum_followups": 2,
    },
}


def _question_record(**overrides) -> dict:
    record = {
        "question_id": "Q1",
        "competency": "C1",
        "stage": "S1",
        "difficulty": "Easy",
        "question": "What is a JOIN?",
        "expected_concepts": "INNER JOIN, OUTER JOIN",
    }
    record.update(overrides)
    return record


def _valid_evaluation(**overrides) -> dict:
    payload = {
        "score": 3,
        "confidence": 80,
        "strengths": ["Explained INNER JOIN correctly"],
        "weaknesses": ["Did not mention OUTER JOIN"],
        "missing_concepts": ["OUTER JOIN"],
        "evidence": ["Candidate described equality matching"],
        "recommendation": "Solid foundational understanding.",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def question_repository():
    repo = MagicMock()
    repo.reference_data = {"thresholds": THRESHOLDS}
    repo.valid_competencies = {"C1", "C2"}
    repo.valid_stages = {"S1", "S2"}
    repo.valid_difficulties = {"Easy", "Medium", "Hard"}
    repo.get_questions_by_stage.return_value = [_question_record()]
    return repo


@pytest.fixture
def prompt_service():
    service = MagicMock()
    prompt_result = MagicMock()
    prompt_result.prompt = "rendered evaluation prompt"
    service.build_evaluation_prompt.return_value = prompt_result
    return service


@pytest.fixture
def gemini_service():
    service = MagicMock()
    service.generate_json.return_value = _valid_evaluation()
    return service


@pytest.fixture
def evaluation_service(question_repository, prompt_service, gemini_service):
    return EvaluationService(
        question_repository=question_repository,
        prompt_service=prompt_service,
        gemini_service=gemini_service,
        logger=logging.getLogger("test.evaluation_service"),
    )


# ============================================================
# Successful evaluation
# ============================================================


class TestEvaluateAnswerSuccess:
    def test_successful_evaluation_returns_evaluation_result(self, evaluation_service):
        result = evaluation_service.evaluate_answer(
            question_id="Q1",
            candidate_answer="A JOIN combines rows from two tables.",
            competency="C1",
            stage="S1",
            difficulty="Easy",
        )
        assert isinstance(result, EvaluationResult)
        assert result.score == 3
        assert result.confidence == 80
        assert result.recommendation == "Solid foundational understanding."

    def test_successful_evaluation_calls_collaborators_in_order(
        self, evaluation_service, question_repository, prompt_service, gemini_service
    ):
        evaluation_service.evaluate_answer(
            question_id="Q1",
            candidate_answer="A JOIN combines rows from two tables.",
            competency="C1",
            stage="S1",
            difficulty="Easy",
        )
        question_repository.get_questions_by_stage.assert_called_once_with("S1")
        prompt_service.build_evaluation_prompt.assert_called_once()
        gemini_service.generate_json.assert_called_once()

    def test_timestamp_is_a_datetime(self, evaluation_service):
        result = evaluation_service.evaluate_answer(
            question_id="Q1",
            candidate_answer="Answer text",
            competency="C1",
            stage="S1",
            difficulty="Easy",
        )
        assert isinstance(result.timestamp, datetime)

    def test_confidence_is_coerced_to_int(self, evaluation_service, gemini_service):
        gemini_service.generate_json.return_value = _valid_evaluation(confidence=87.6)
        result = evaluation_service.evaluate_answer(
            question_id="Q1",
            candidate_answer="Answer text",
            competency="C1",
            stage="S1",
            difficulty="Easy",
        )
        assert result.confidence == 88
        assert isinstance(result.confidence, int)

    def test_metadata_carries_question_context(self, evaluation_service):
        result = evaluation_service.evaluate_answer(
            question_id="Q1",
            candidate_answer="Answer text",
            competency="C1",
            stage="S1",
            difficulty="Easy",
        )
        assert result.metadata["question_id"] == "Q1"
        assert result.metadata["competency"] == "C1"

    def test_statistics_updated_after_success(self, evaluation_service):
        evaluation_service.evaluate_answer(
            question_id="Q1",
            candidate_answer="Answer text",
            competency="C1",
            stage="S1",
            difficulty="Easy",
        )
        stats = evaluation_service.get_statistics()
        assert stats["total_evaluations"] == 1
        assert stats["successful"] == 1
        assert stats["failed"] == 0
        assert stats["average_score"] == 3


# ============================================================
# Invalid input
# ============================================================


class TestInputValidation:
    def test_none_answer_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer=None,
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )

    def test_empty_answer_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )

    def test_blank_answer_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="   \n\t  ",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )

    def test_non_string_answer_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer=12345,
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )

    def test_unknown_competency_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="UNKNOWN",
                stage="S1",
                difficulty="Easy",
            )

    def test_unknown_stage_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="UNKNOWN",
                difficulty="Easy",
            )

    def test_unknown_difficulty_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="UNKNOWN",
            )


# ============================================================
# Question not found / repository failure
# ============================================================


class TestQuestionLookup:
    def test_question_not_found_raises_repository_error(self, evaluation_service, question_repository):
        question_repository.get_questions_by_stage.return_value = []
        with pytest.raises(EvaluationRepositoryError):
            evaluation_service.evaluate_answer(
                question_id="MISSING",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )

    def test_competency_mismatch_raises_validation_error(self, evaluation_service, question_repository):
        question_repository.get_questions_by_stage.return_value = [
            _question_record(competency="C2")
        ]
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )

    def test_repository_failure_translated(self, evaluation_service, question_repository):
        question_repository.get_questions_by_stage.side_effect = QuestionBankError("db down")
        with pytest.raises(EvaluationRepositoryError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )


# ============================================================
# Prompt failure
# ============================================================


class TestPromptFailure:
    def test_prompt_generation_failure_translated(self, evaluation_service, prompt_service):
        prompt_service.build_evaluation_prompt.side_effect = PromptRenderError(
            "missing placeholder"
        )
        with pytest.raises(EvaluationPromptError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )


# ============================================================
# Gemini failures
# ============================================================


class TestGeminiFailure:
    def test_gemini_timeout_translated(self, evaluation_service, gemini_service):
        gemini_service.generate_json.side_effect = GeminiTimeoutError("timed out")
        with pytest.raises(EvaluationAIError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )

    def test_gemini_safety_error_translated(self, evaluation_service, gemini_service):
        gemini_service.generate_json.side_effect = GeminiSafetyError("blocked")
        with pytest.raises(EvaluationAIError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )

    def test_invalid_gemini_json_shape_raises_validation_error(
        self, evaluation_service, gemini_service
    ):
        gemini_service.generate_json.return_value = {"not": "a valid evaluation"}
        with pytest.raises(EvaluationValidationError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )


# ============================================================
# validate_evaluation()
# ============================================================


class TestValidateEvaluation:
    def test_valid_payload_passes(self, evaluation_service):
        assert evaluation_service.validate_evaluation(_valid_evaluation()) is True

    def test_invalid_score_out_of_range(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.validate_evaluation(_valid_evaluation(score=99))

    def test_invalid_score_wrong_type(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.validate_evaluation(_valid_evaluation(score="3"))

    def test_invalid_confidence_out_of_range(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.validate_evaluation(_valid_evaluation(confidence=150))

    def test_missing_recommendation(self, evaluation_service):
        payload = _valid_evaluation()
        del payload["recommendation"]
        with pytest.raises(EvaluationValidationError):
            evaluation_service.validate_evaluation(payload)

    def test_blank_recommendation(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.validate_evaluation(_valid_evaluation(recommendation="   "))

    def test_strengths_not_a_list(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.validate_evaluation(_valid_evaluation(strengths="not a list"))


# ============================================================
# determine_followup()
# ============================================================


class TestDetermineFollowup:
    def test_low_score_triggers_followup(self, evaluation_service):
        decision = evaluation_service.determine_followup(
            _valid_evaluation(score=1, missing_concepts=["OUTER JOIN"]),
            current_followup_count=0,
        )
        assert decision["needs_followup"] is True
        assert decision["next_action"] == "FOLLOWUP"
        assert decision["missing_concept"] == "OUTER JOIN"

    def test_high_score_advances_without_followup(self, evaluation_service):
        decision = evaluation_service.determine_followup(
            _valid_evaluation(score=5), current_followup_count=0
        )
        assert decision["needs_followup"] is False
        assert decision["next_action"] == "ADVANCE"

    def test_mid_score_continues_without_followup(self, evaluation_service):
        decision = evaluation_service.determine_followup(
            _valid_evaluation(score=3), current_followup_count=0
        )
        assert decision["needs_followup"] is False
        assert decision["next_action"] == "CONTINUE"

    def test_maximum_followups_forces_advance(self, evaluation_service):
        decision = evaluation_service.determine_followup(
            _valid_evaluation(score=0), current_followup_count=2
        )
        assert decision["needs_followup"] is False
        assert decision["next_action"] == "ADVANCE"

    def test_negative_followup_count_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.determine_followup(
                _valid_evaluation(score=1), current_followup_count=-1
            )


# ============================================================
# get_score_level()
# ============================================================


class TestGetScoreLevel:
    def test_known_score_returns_configured_label(self, evaluation_service):
        assert evaluation_service.get_score_level(3) == "Meets Expectations"
        assert evaluation_service.get_score_level(5) == "Exceptional"

    def test_out_of_range_score_rejected(self, evaluation_service):
        with pytest.raises(EvaluationValidationError):
            evaluation_service.get_score_level(99)


# ============================================================
# Configuration errors
# ============================================================


class TestConfigurationErrors:
    def test_missing_answer_scoring_section_raises_configuration_error(
        self, prompt_service, gemini_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "thresholds": {"adaptive_questioning": THRESHOLDS["adaptive_questioning"]}
        }
        with pytest.raises(EvaluationConfigurationError):
            EvaluationService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
            )

    def test_missing_adaptive_questioning_section_raises_configuration_error(
        self, prompt_service, gemini_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "thresholds": {"answer_scoring": THRESHOLDS["answer_scoring"]}
        }
        with pytest.raises(EvaluationConfigurationError):
            EvaluationService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
            )


# ============================================================
# Statistics
# ============================================================


class TestStatistics:
    def test_failed_evaluation_increments_failed_count(self, evaluation_service, gemini_service):
        gemini_service.generate_json.side_effect = GeminiTimeoutError("timed out")
        with pytest.raises(EvaluationAIError):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )
        stats = evaluation_service.get_statistics()
        assert stats["total_evaluations"] == 1
        assert stats["failed"] == 1
        assert stats["successful"] == 0

    def test_followups_triggered_counter(self, evaluation_service, gemini_service):
        gemini_service.generate_json.return_value = _valid_evaluation(score=1)
        evaluation_service.evaluate_answer(
            question_id="Q1",
            candidate_answer="valid answer",
            competency="C1",
            stage="S1",
            difficulty="Easy",
        )
        stats = evaluation_service.get_statistics()
        assert stats["followups_triggered"] == 1

    def test_average_score_across_multiple_evaluations(self, evaluation_service, gemini_service):
        gemini_service.generate_json.side_effect = [
            _valid_evaluation(score=2),
            _valid_evaluation(score=4),
        ]
        for _ in range(2):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )
        stats = evaluation_service.get_statistics()
        assert stats["average_score"] == 3


# ============================================================
# Logging
# ============================================================


class TestLogging:
    def test_success_logs_info(self, evaluation_service, caplog):
        with caplog.at_level(logging.INFO, logger="test.evaluation_service"):
            evaluation_service.evaluate_answer(
                question_id="Q1",
                candidate_answer="valid answer",
                competency="C1",
                stage="S1",
                difficulty="Easy",
            )
        assert any("evaluation.success" in record.message for record in caplog.records)

    def test_failure_logs_warning(self, evaluation_service, gemini_service, caplog):
        gemini_service.generate_json.side_effect = GeminiTimeoutError("timed out")
        with caplog.at_level(logging.WARNING, logger="test.evaluation_service"):
            with pytest.raises(EvaluationAIError):
                evaluation_service.evaluate_answer(
                    question_id="Q1",
                    candidate_answer="valid answer",
                    competency="C1",
                    stage="S1",
                    difficulty="Easy",
                )
        assert any("evaluation.failed" in record.message for record in caplog.records)


# ============================================================
# Dependency injection
# ============================================================


class TestDependencyInjection:
    def test_constructor_accepts_injected_collaborators_only(
        self, question_repository, prompt_service, gemini_service
    ):
        service = EvaluationService(
            question_repository=question_repository,
            prompt_service=prompt_service,
            gemini_service=gemini_service,
        )
        assert service.question_repository is question_repository
        assert service.prompt_service is prompt_service
        assert service.gemini_service is gemini_service

    def test_no_module_level_globals_required(self, evaluation_service):
        # A fresh service instance carries its own statistics rather
        # than sharing module-level mutable state.
        stats = evaluation_service.get_statistics()
        assert stats["total_evaluations"] == 0


# ============================================================
# Exception translation
# ============================================================


class TestExceptionTranslation:
    def test_translate_gemini_exception_returns_evaluation_ai_error(self):
        original = GeminiTimeoutError("timed out", request_id="abc123")
        translated = EvaluationService._translate_gemini_exception(original)
        assert isinstance(translated, EvaluationAIError)
        assert translated.request_id == "abc123"

    def test_translate_repository_exception_returns_evaluation_repository_error(self):
        original = QuestionNotFoundError("Q404 not found")
        translated = EvaluationService._translate_repository_exception(original)
        assert isinstance(translated, EvaluationRepositoryError)


# ============================================================
# Health check
# ============================================================


class TestHealthCheck:
    def test_healthy_when_repository_and_gemini_ok(
        self, evaluation_service, question_repository, gemini_service
    ):
        question_repository.get_total_questions.return_value = 42
        gemini_service.health_check.return_value = True
        assert evaluation_service.health_check() is True

    def test_unhealthy_when_gemini_reports_unhealthy(
        self, evaluation_service, gemini_service
    ):
        gemini_service.health_check.return_value = False
        assert evaluation_service.health_check() is False

    def test_unhealthy_when_repository_raises(
        self, evaluation_service, question_repository
    ):
        question_repository.get_total_questions.side_effect = QuestionBankError(
            "bank not loaded"
        )
        assert evaluation_service.health_check() is False

    def test_health_check_never_raises(self, evaluation_service, gemini_service):
        gemini_service.health_check.side_effect = RuntimeError("unexpected")
        assert evaluation_service.health_check() is False


# ============================================================
# Thread safety
# ============================================================


class TestThreadSafety:
    def test_concurrent_evaluations_produce_consistent_statistics(
        self, question_repository, prompt_service, gemini_service
    ):
        gemini_service.generate_json.return_value = _valid_evaluation(score=3)
        service = EvaluationService(
            question_repository=question_repository,
            prompt_service=prompt_service,
            gemini_service=gemini_service,
        )

        errors: list[Exception] = []

        def run_evaluation() -> None:
            try:
                service.evaluate_answer(
                    question_id="Q1",
                    candidate_answer="valid answer",
                    competency="C1",
                    stage="S1",
                    difficulty="Easy",
                )
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(exc)

        threads = [threading.Thread(target=run_evaluation) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        stats = service.get_statistics()
        assert stats["total_evaluations"] == 20
        assert stats["successful"] == 20
        assert stats["average_score"] == 3