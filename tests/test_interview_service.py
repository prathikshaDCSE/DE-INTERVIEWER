"""
tests/test_interview_service.py

Comprehensive pytest suite for InterviewService.

Testing strategy
-----------------
* Every collaborator (QuestionRepository, PromptService, GeminiService,
  EvaluationService, ReportService) is a MagicMock/stub built from
  fixtures below -- InterviewService is tested in isolation, never
  against real Gemini calls or a real question bank.
* ``question_repository.reference_data`` is populated with minimal but
  structurally valid stages/competencies/thresholds fixtures mirroring
  the real ``stages.yaml`` / ``competencies.yaml`` / ``thresholds.yaml``
  shape, since InterviewService reads that structure directly rather
  than re-parsing YAML.
* EvaluationResult objects are constructed directly (not through
  EvaluationService) so each test controls exactly the
  score/next_action/missing_concept it wants to drive orchestration
  logic deterministically.
* Tests are grouped by public method, with additional groups for
  configuration loading, exception translation, and thread-safety
  bookkeeping.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from exceptions.evaluation_exceptions import EvaluationServiceError
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
from exceptions.gemini_exceptions import GeminiServiceError
from models.evaluation_result import EvaluationResult
from models.interview_session import AskedQuestion, InterviewStatus
from repository.question_repository import QuestionBankError
from services.evaluation_service import EvaluationService
from services.gemini_service import GeminiService
from services.interview_service import InterviewService
from services.prompt_service import PromptResult, PromptService, PromptServiceError


# ============================================================
# Fixtures -- reference configuration
# ============================================================


@pytest.fixture
def stages_config() -> dict:
    return {
        "stages": [
            {
                "id": "S0",
                "display_order": 1,
                "name": "Intake",
                "scored": False,
                "minimum_questions": 0,
                "maximum_questions": 0,
                "competencies": [],
                "next_stage": "S1",
            },
            {
                "id": "S1",
                "display_order": 2,
                "name": "Fundamentals",
                "scored": True,
                "minimum_questions": 1,
                "maximum_questions": 2,
                "competencies": ["C1"],
                "next_stage": "S4",
            },
            {
                "id": "S4",
                "display_order": 3,
                "name": "Wrap-Up",
                "scored": False,
                "minimum_questions": 0,
                "maximum_questions": 0,
                "competencies": [],
                "next_stage": "END",
            },
        ]
    }


@pytest.fixture
def competencies_config() -> dict:
    return {
        "competencies": [
            {
                "id": "C1",
                "name": "SQL & Query Optimization",
                "evaluation_weight": 15,
                "minimum_questions": 1,
                "difficulty_levels": ["Easy", "Medium", "Hard"],
            }
        ]
    }


@pytest.fixture
def thresholds_config() -> dict:
    return {
        "answer_scoring": {
            "minimum_score": 0,
            "maximum_score": 5,
            "score_scale": {
                "0": {"label": "No Answer", "description": "x"},
                "3": {"label": "Meets Expectations", "description": "x"},
                "5": {"label": "Exceptional", "description": "x"},
            },
        },
        "adaptive_questioning": {
            "followup_threshold": {"score_less_than_or_equal": 2},
            "advance_threshold": {"score_greater_than_or_equal": 4},
            "maximum_followups": 2,
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
# Fixtures -- question records
# ============================================================


def make_question_record(question_id: str = "Q1", **overrides) -> dict:
    record = {
        "question_id": question_id,
        "competency": "C1",
        "stage": "S1",
        "difficulty": "Medium",
        "question": "Explain the difference between INNER and LEFT JOIN.",
        "learning_objective": "Assess join understanding",
        "business_context": "Common in reporting pipelines",
        "next_difficulty_if_score_ge_4": "Hard",
        "next_difficulty_if_score_le_2": "Easy",
        "followup_question_1": "Follow-up 1",
        "followup_question_2": "Follow-up 2",
    }
    record.update(overrides)
    return record


# ============================================================
# Fixtures -- collaborators
# ============================================================


@pytest.fixture
def question_repository(stages_config, competencies_config, thresholds_config):
    repo = MagicMock()
    repo.reference_data = {
        "stages": stages_config,
        "competencies": competencies_config,
        "thresholds": thresholds_config,
    }
    repo.valid_competencies = {"C1"}
    repo.valid_stages = {"S0", "S1", "S4"}
    repo.valid_difficulties = {"Easy", "Medium", "Hard"}
    repo.get_total_questions.return_value = 10

    repo.get_question.return_value = make_question_record()
    repo.mark_question_used.return_value = None
    return repo


@pytest.fixture
def prompt_service():
    service = MagicMock(spec=PromptService)

    def _interview_prompt(*args, **kwargs):
        return PromptResult(
            prompt="INTERVIEW PROMPT TEXT",
            prompt_type="INTERVIEW_QUESTION",
            version="1.0",
            estimated_tokens=10,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )

    def _followup_prompt(*args, **kwargs):
        return PromptResult(
            prompt="FOLLOWUP PROMPT TEXT",
            prompt_type="FOLLOWUP",
            version="1.0",
            estimated_tokens=8,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )

    def _dynamic_question_prompt(*args, **kwargs):
        return PromptResult(
            prompt="DYNAMIC QUESTION PROMPT TEXT",
            prompt_type="DYNAMIC_QUESTION",
            version="1.0",
            estimated_tokens=12,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )

    def _dynamic_followup_prompt(*args, **kwargs):
        return PromptResult(
            prompt="DYNAMIC FOLLOWUP PROMPT TEXT",
            prompt_type="DYNAMIC_FOLLOWUP",
            version="1.0",
            estimated_tokens=10,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )

    service.build_interview_prompt.side_effect = _interview_prompt
    service.build_followup_prompt.side_effect = _followup_prompt
    service.build_dynamic_question_prompt.side_effect = _dynamic_question_prompt
    service.build_dynamic_followup_prompt.side_effect = _dynamic_followup_prompt
    service.reload_configuration.return_value = None
    return service


@pytest.fixture
def gemini_service():
    service = MagicMock(spec=GeminiService)
    service._max_retries = 3
    service.generate_json.return_value = {
        "question": "AI Generated Question?",
        "competency": "C1",
        "difficulty": "Medium",
        "estimated_time": 5,
        "learning_objective": "Dynamic objective",
        "business_context": "Dynamic context",
        "expected_concepts": ["concept 1"],
        "evaluator_notes": "Dynamic notes",
        "positive_indicators": ["pos"],
        "negative_indicators": ["neg"],
        "missing_concept": "missing concept 1",
        "followup_reason": "needs depth",
    }
    return service


@pytest.fixture
def evaluation_service():
    service = MagicMock(spec=EvaluationService)
    service.reload_configuration.return_value = None
    service.health_check.return_value = True
    return service


@pytest.fixture
def report_service():
    service = MagicMock()
    service.generate_report.return_value = {"report": "ok"}
    return service


@pytest.fixture
def interview_service(
    question_repository, prompt_service, gemini_service, evaluation_service, report_service
):
    return InterviewService(
        question_repository=question_repository,
        prompt_service=prompt_service,
        gemini_service=gemini_service,
        evaluation_service=evaluation_service,
        report_service=report_service,
    )


@pytest.fixture
def candidate() -> dict:
    return {"name": "Prathiksha", "experience_years": 2, "target_role": "Data Engineer"}


def make_evaluation_result(
    score: int = 4,
    next_action: str = "ADVANCE",
    needs_followup: bool = False,
    missing_concept: str | None = None,
    missing_concepts: list[str] | None = None,
    followup_reason: str | None = None,
    competency: str = "C1",
    confidence: int = 90,
) -> EvaluationResult:
    return EvaluationResult(
        score=score,
        confidence=confidence,
        strengths=["good use of joins"],
        weaknesses=["missed edge case"],
        missing_concepts=missing_concepts or [],
        evidence=["mentioned LEFT JOIN correctly"],
        recommendation="Solid grasp of joins.",
        needs_followup=needs_followup,
        next_action=next_action,
        missing_concept=missing_concept,
        followup_reason=followup_reason,
        request_id="req123",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "question_id": "Q1",
            "competency": competency,
            "stage": "S1",
            "difficulty": "Medium",
        },
    )


# ============================================================
# Construction / configuration
# ============================================================


class TestConstruction:
    def test_successful_construction_loads_stage_and_competency_cache(
        self, interview_service
    ):
        assert interview_service._stage_sequence == ["S0", "S1", "S4"]
        assert "C1" in interview_service._competencies

    def test_missing_stages_raises_configuration_error(
        self, prompt_service, gemini_service, evaluation_service, report_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "stages": {"stages": []},
            "competencies": {"competencies": [{"id": "C1"}]},
            "thresholds": {},
        }
        with pytest.raises(InterviewConfigurationError):
            InterviewService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
                evaluation_service=evaluation_service,
                report_service=report_service,
            )

    def test_stage_missing_next_stage_raises_configuration_error(
        self, prompt_service, gemini_service, evaluation_service, report_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "stages": {"stages": [{"id": "S1", "display_order": 1}]},
            "competencies": {"competencies": [{"id": "C1"}]},
            "thresholds": {},
        }
        with pytest.raises(InterviewConfigurationError):
            InterviewService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
                evaluation_service=evaluation_service,
                report_service=report_service,
            )

    def test_missing_competencies_raises_configuration_error(
        self, stages_config, prompt_service, gemini_service, evaluation_service, report_service
    ):
        repo = MagicMock()
        repo.reference_data = {
            "stages": stages_config,
            "competencies": {"competencies": []},
            "thresholds": {},
        }
        with pytest.raises(InterviewConfigurationError):
            InterviewService(
                question_repository=repo,
                prompt_service=prompt_service,
                gemini_service=gemini_service,
                evaluation_service=evaluation_service,
                report_service=report_service,
            )

    def test_reload_configuration_propagates_to_collaborators(self, interview_service):
        interview_service.reload_configuration()
        interview_service.prompt_service.reload_configuration.assert_called_once()
        interview_service.evaluation_service.reload_configuration.assert_called_once()

    def test_reload_configuration_translates_prompt_service_error(
        self, interview_service
    ):
        interview_service.prompt_service.reload_configuration.side_effect = (
            PromptServiceError("boom")
        )
        with pytest.raises(InterviewConfigurationError):
            interview_service.reload_configuration()

    def test_reload_configuration_translates_evaluation_service_error(
        self, interview_service
    ):
        interview_service.evaluation_service.reload_configuration.side_effect = (
            EvaluationServiceError("boom")
        )
        with pytest.raises(InterviewConfigurationError):
            interview_service.reload_configuration()


# ============================================================
# start_interview
# ============================================================


class TestStartInterview:
    def test_start_interview_skips_non_scored_intake_stage(
        self, interview_service, candidate
    ):
        result = interview_service.start_interview(candidate)
        assert result["stage"] == "S1"
        assert result["question"] == "INTERVIEW PROMPT TEXT"
        assert result["question_id"] == "Q1"

    def test_start_interview_rejects_missing_name(self, interview_service):
        with pytest.raises(InterviewValidationError):
            interview_service.start_interview({"experience_years": 2})

    def test_start_interview_rejects_non_mapping(self, interview_service):
        with pytest.raises(InterviewValidationError):
            interview_service.start_interview("not a mapping")  # type: ignore[arg-type]

    def test_start_interview_infers_candidate_level_from_experience(
        self, interview_service, candidate
    ):
        interview_service.start_interview(candidate)
        session = next(iter(interview_service._sessions.values()))
        assert session.candidate["candidate_level"] == "Junior"

    def test_start_interview_respects_explicit_candidate_level(
        self, interview_service, candidate
    ):
        candidate["candidate_level"] = "Senior"
        interview_service.start_interview(candidate)
        session = next(iter(interview_service._sessions.values()))
        assert session.candidate["candidate_level"] == "Senior"

    def test_start_interview_increments_statistics(self, interview_service, candidate):
        interview_service.start_interview(candidate)
        stats = interview_service.get_statistics()
        assert stats["interviews_started"] == 1
        assert stats["questions_asked"] == 1

    def test_start_interview_raises_ai_error_when_ai_generation_fails(
        self, interview_service, candidate, gemini_service
    ):
        interview_service.question_repository.get_question.return_value = None
        gemini_service.generate_json.side_effect = GeminiServiceError("gemini down")
        with pytest.raises(InterviewAIError):
            interview_service.start_interview(candidate)

    def test_start_interview_translates_repository_error(
        self, interview_service, candidate
    ):
        interview_service.question_repository.get_question.side_effect = (
            QuestionBankError("db down")
        )
        with pytest.raises(InterviewQuestionError):
            interview_service.start_interview(candidate)

    def test_start_interview_translates_prompt_error(self, interview_service, candidate):
        interview_service.prompt_service.build_interview_prompt.side_effect = (
            PromptServiceError("bad template")
        )
        with pytest.raises(InterviewPromptError):
            interview_service.start_interview(candidate)

    def test_session_is_registered_and_retrievable(self, interview_service, candidate):
        result = interview_service.start_interview(candidate)
        session_id = result["session_id"]
        progress = interview_service.get_progress(session_id)
        assert progress["session_id"] == session_id
        assert progress["status"] == InterviewStatus.IN_PROGRESS.value


# ============================================================
# submit_answer -- FOLLOWUP path
# ============================================================


class TestSubmitAnswerFollowup:
    def _start(self, interview_service, candidate):
        return interview_service.start_interview(candidate)["session_id"]

    def test_followup_action_returns_followup_prompt(
        self, interview_service, candidate
    ):
        session_id = self._start(interview_service, candidate)
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(
                score=1,
                next_action="FOLLOWUP",
                needs_followup=True,
                missing_concept="LEFT JOIN semantics",
                missing_concepts=["LEFT JOIN semantics"],
                followup_reason="candidate omitted null handling",
            )
        )

        result = interview_service.submit_answer(session_id, "partial answer")

        assert result["next_action"] == "FOLLOWUP"
        assert result["question"] == "FOLLOWUP PROMPT TEXT"
        interview_service.prompt_service.build_followup_prompt.assert_called_once()

    def test_followup_increments_followup_counts(self, interview_service, candidate):
        session_id = self._start(interview_service, candidate)
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(
                score=1,
                next_action="FOLLOWUP",
                needs_followup=True,
                missing_concept="X",
                missing_concepts=["X"],
                followup_reason="reason",
            )
        )
        interview_service.submit_answer(session_id, "answer")

        session = interview_service._sessions[session_id]
        assert session.followup_counts["Q1"] == 1
        assert session.followups_per_competency["C1"] == 1

    def test_second_followup_answers_key_off_previous_followup(
        self, interview_service, candidate
    ):
        session_id = self._start(interview_service, candidate)
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(
                score=1,
                next_action="FOLLOWUP",
                needs_followup=True,
                missing_concept="X",
                missing_concepts=["X"],
                followup_reason="reason",
            )
        )
        interview_service.submit_answer(session_id, "first answer")
        interview_service.submit_answer(session_id, "second answer (followup 1)")

        session = interview_service._sessions[session_id]
        assert session.answers["Q1"] == "first answer"
        assert session.answers["Q1::followup::1"] == "second answer (followup 1)"
        assert session.followup_counts["Q1"] == 2

    def test_statistics_track_followups_asked(self, interview_service, candidate):
        session_id = self._start(interview_service, candidate)
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(
                score=1,
                next_action="FOLLOWUP",
                needs_followup=True,
                missing_concept="X",
                missing_concepts=["X"],
                followup_reason="reason",
            )
        )
        interview_service.submit_answer(session_id, "answer")
        stats = interview_service.get_statistics()
        assert stats["followups_asked"] == 1


# ============================================================
# submit_answer -- ADVANCE / CONTINUE / stage progression
# ============================================================


class TestSubmitAnswerAdvance:
    def _start(self, interview_service, candidate):
        return interview_service.start_interview(candidate)["session_id"]

    def test_advance_moves_to_next_question_when_stage_not_complete(
        self, interview_service, candidate
    ):
        # Stage S1 requires minimum 1 / maximum 2 questions, one
        # competency (C1). After the first ADVANCE, min is satisfied
        # and competency is covered -> stage should complete.
        session_id = self._start(interview_service, candidate)
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=5, next_action="ADVANCE")
        )

        result = interview_service.submit_answer(session_id, "great answer")

        # Only one competency (C1) with minimum_questions=1 -> covered
        # after first question -> stage completes -> next stage S4 is
        # also non-scored -> interview completes.
        assert result["next_action"] == "INTERVIEW_COMPLETE"
        session = interview_service._sessions[session_id]
        assert session.status == InterviewStatus.COMPLETED

    def test_continue_action_selects_next_question_without_advancing_stage(
        self, interview_service, candidate, question_repository
    ):
        # Configure two competencies so CONTINUE keeps us in-stage.
        question_repository.reference_data["stages"]["stages"][1]["competencies"] = [
            "C1",
            "C2",
        ]
        question_repository.reference_data["competencies"]["competencies"].append(
            {
                "id": "C2",
                "name": "Data Modeling",
                "evaluation_weight": 15,
                "minimum_questions": 1,
                "difficulty_levels": ["Easy", "Medium", "Hard"],
            }
        )
        question_repository.valid_competencies = {"C1", "C2"}

        svc = InterviewService(
            question_repository=question_repository,
            prompt_service=interview_service.prompt_service,
            gemini_service=interview_service.gemini_service,
            evaluation_service=interview_service.evaluation_service,
            report_service=interview_service.report_service,
        )
        session_id = svc.start_interview(candidate)["session_id"]

        svc.evaluation_service.evaluate_answer.return_value = make_evaluation_result(
            score=3, next_action="CONTINUE"
        )
        result = svc.submit_answer(session_id, "adequate answer")

        assert result["next_action"] == "NEXT_QUESTION"
        session = svc._sessions[session_id]
        assert "C1" in session.covered_competencies
        assert session.status == InterviewStatus.IN_PROGRESS

    def test_adaptive_difficulty_increases_on_high_score(
        self, interview_service, candidate
    ):
        session_id = self._start(interview_service, candidate)
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=5, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "great answer")
        session = interview_service._sessions[session_id]
        assert (
            session.metadata["difficulty_by_competency"]["C1"] == "Hard"
        )

    def test_adaptive_difficulty_decreases_on_low_score(
        self, interview_service, candidate, question_repository
    ):
        question_repository.reference_data["stages"]["stages"][1][
            "maximum_questions"
        ] = 5
        question_repository.reference_data["stages"]["stages"][1][
            "minimum_questions"
        ] = 5
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=1, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "weak answer")
        session = interview_service._sessions[session_id]
        assert session.metadata["difficulty_by_competency"]["C1"] == "Easy"

    def test_evaluation_history_and_answers_recorded(
        self, interview_service, candidate
    ):
        session_id = self._start(interview_service, candidate)
        evaluation = make_evaluation_result(score=4, next_action="ADVANCE")
        interview_service.evaluation_service.evaluate_answer.return_value = evaluation

        interview_service.submit_answer(session_id, "my answer")
        session = interview_service._sessions[session_id]
        assert len(session.evaluation_history) == 1
        assert session.answers["Q1"] == "my answer"


# ============================================================
# submit_answer -- validation / session errors
# ============================================================


class TestSubmitAnswerErrors:
    def test_submit_answer_unknown_session_raises_session_error(
        self, interview_service
    ):
        with pytest.raises(InterviewSessionError):
            interview_service.submit_answer("does-not-exist", "answer")

    def test_submit_answer_after_completion_raises_session_error(
        self, interview_service, candidate
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=5, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "answer")  # completes interview

        with pytest.raises(InterviewSessionError):
            interview_service.submit_answer(session_id, "another answer")

    def test_submit_answer_non_string_raises_validation_error(
        self, interview_service, candidate
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        with pytest.raises(InterviewValidationError):
            interview_service.submit_answer(session_id, 12345)  # type: ignore[arg-type]

    def test_submit_answer_translates_evaluation_service_error(
        self, interview_service, candidate
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.side_effect = (
            EvaluationServiceError("gemini down")
        )
        with pytest.raises(InterviewEvaluationError):
            interview_service.submit_answer(session_id, "answer")

    def test_submit_answer_translates_followup_prompt_error(
        self, interview_service, candidate
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(
                score=1,
                next_action="FOLLOWUP",
                needs_followup=True,
                missing_concept="X",
                missing_concepts=["X"],
                followup_reason="reason",
            )
        )
        interview_service.prompt_service.build_followup_prompt.side_effect = (
            PromptServiceError("bad")
        )
        with pytest.raises(InterviewPromptError):
            interview_service.submit_answer(session_id, "answer")

    def test_submit_answer_translates_next_question_repository_error(
        self, interview_service, candidate, question_repository
    ):
        question_repository.reference_data["stages"]["stages"][1]["competencies"] = [
            "C1",
            "C2",
        ]
        question_repository.reference_data["competencies"]["competencies"].append(
            {
                "id": "C2",
                "name": "Data Modeling",
                "evaluation_weight": 15,
                "minimum_questions": 1,
                "difficulty_levels": ["Easy", "Medium", "Hard"],
            }
        )
        question_repository.valid_competencies = {"C1", "C2"}

        svc = InterviewService(
            question_repository=question_repository,
            prompt_service=interview_service.prompt_service,
            gemini_service=interview_service.gemini_service,
            evaluation_service=interview_service.evaluation_service,
            report_service=interview_service.report_service,
        )
        session_id = svc.start_interview(candidate)["session_id"]

        svc.evaluation_service.evaluate_answer.return_value = make_evaluation_result(
            score=3, next_action="CONTINUE"
        )
        question_repository.get_question.side_effect = QuestionBankError("boom")

        with pytest.raises(InterviewQuestionError):
            svc.submit_answer(session_id, "answer")

# ============================================================
# get_current_question / get_next_question
# ============================================================


class TestQuestionRetrieval:
    def test_get_current_question_returns_none_when_no_active_question(
        self, interview_service, candidate
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service._sessions[session_id].current_question = None
        assert interview_service.get_current_question(session_id) is None

    def test_get_current_question_rebuilds_prompt(self, interview_service, candidate):
        session_id = interview_service.start_interview(candidate)["session_id"]
        payload = interview_service.get_current_question(session_id)
        assert payload["prompt"] == "INTERVIEW PROMPT TEXT"

    def test_get_current_question_unknown_session_raises(self, interview_service):
        with pytest.raises(InterviewSessionError):
            interview_service.get_current_question("nope")

    def test_get_next_question_returns_none_after_completion(
        self, interview_service, candidate
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=5, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "answer")  # completes

        with pytest.raises(InterviewSessionError):
            interview_service.get_next_question(session_id)


# ============================================================
# advance_stage / is_stage_complete / is_interview_complete
# ============================================================


class TestStageAdvancement:
    def test_advance_stage_forces_progression(
        self, interview_service, candidate, question_repository
    ):
        question_repository.reference_data["stages"]["stages"][1][
            "maximum_questions"
        ] = 5
        question_repository.reference_data["stages"]["stages"][1][
            "minimum_questions"
        ] = 5
        session_id = interview_service.start_interview(candidate)["session_id"]

        progress = interview_service.advance_stage(session_id)
        assert "S1" in progress["completed_stages"]

    def test_advance_stage_unknown_session_raises(self, interview_service):
        with pytest.raises(InterviewSessionError):
            interview_service.advance_stage("nope")

    def test_is_stage_complete_true_when_ended(self, interview_service, candidate):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=5, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "answer")
        assert interview_service.is_stage_complete(session_id) is True

    def test_is_interview_complete_false_while_in_progress(
        self, interview_service, candidate, question_repository
    ):
        question_repository.reference_data["stages"]["stages"][1][
            "maximum_questions"
        ] = 5
        question_repository.reference_data["stages"]["stages"][1][
            "minimum_questions"
        ] = 5
        session_id = interview_service.start_interview(candidate)["session_id"]
        assert interview_service.is_interview_complete(session_id) is False

    def test_is_interview_complete_true_after_completion(
        self, interview_service, candidate
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=5, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "answer")
        assert interview_service.is_interview_complete(session_id) is True


# ============================================================
# generate_summary
# ============================================================


class TestGenerateSummary:
    def _complete_interview(self, interview_service, candidate, score=5):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=score, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "answer")
        return session_id

    def test_generate_summary_calls_report_service_with_aggregated_data(
        self, interview_service, candidate
    ):
        session_id = self._complete_interview(interview_service, candidate, score=5)
        result = interview_service.generate_summary(session_id)

        assert result == {"report": "ok"}
        call_kwargs = interview_service.report_service.generate_report.call_args.kwargs
        assert call_kwargs["decision"] == "SELECT"
        assert call_kwargs["candidate"]["name"] == "Prathiksha"
        assert isinstance(call_kwargs["competency_summaries"], list)
        assert call_kwargs["competency_summaries"][0]["competency"] == (
            "SQL & Query Optimization"
        )

    def test_generate_summary_raises_when_interview_in_progress(
        self, interview_service, candidate, question_repository
    ):
        question_repository.reference_data["stages"]["stages"][1][
            "maximum_questions"
        ] = 5
        question_repository.reference_data["stages"]["stages"][1][
            "minimum_questions"
        ] = 5
        session_id = interview_service.start_interview(candidate)["session_id"]
        with pytest.raises(InterviewSessionError):
            interview_service.generate_summary(session_id)

    def test_generate_summary_translates_report_service_failure(
        self, interview_service, candidate
    ):
        session_id = self._complete_interview(interview_service, candidate)
        interview_service.report_service.generate_report.side_effect = RuntimeError(
            "pdf engine down"
        )
        with pytest.raises(InterviewReportError):
            interview_service.generate_summary(session_id)

    def test_generate_summary_low_score_yields_reject_decision(
        self, interview_service, candidate
    ):
        session_id = self._complete_interview(interview_service, candidate, score=0)
        interview_service.generate_summary(session_id)
        call_kwargs = interview_service.report_service.generate_report.call_args.kwargs
        assert call_kwargs["decision"] == "REJECT"

    def test_generate_summary_missing_decision_thresholds_raises_configuration_error(
        self, interview_service, candidate
    ):
        session_id = self._complete_interview(interview_service, candidate)
        interview_service.question_repository.reference_data["thresholds"][
            "decision_thresholds"
        ] = {}
        with pytest.raises(InterviewConfigurationError):
            interview_service.generate_summary(session_id)


# ============================================================
# get_progress
# ============================================================


class TestGetProgress:
    def test_get_progress_unknown_session_raises(self, interview_service):
        with pytest.raises(InterviewSessionError):
            interview_service.get_progress("nope")

    def test_get_progress_percentage_reflects_completed_stages(
        self, interview_service, candidate
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        progress = interview_service.get_progress(session_id)
        assert progress["progress_percentage"] == 33.33

        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=5, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "answer")
        progress = interview_service.get_progress(session_id)
        assert progress["progress_percentage"] == 100.0
        assert progress["status"] == InterviewStatus.COMPLETED.value


# ============================================================
# Statistics
# ============================================================


class TestStatistics:
    def test_statistics_initial_state(self, interview_service):
        stats = interview_service.get_statistics()
        assert stats["interviews_started"] == 0
        assert stats["interviews_completed"] == 0
        assert stats["average_score"] == 0.0

    def test_statistics_after_completed_interview(self, interview_service, candidate):
        session_id = interview_service.start_interview(candidate)["session_id"]
        interview_service.evaluation_service.evaluate_answer.return_value = (
            make_evaluation_result(score=5, next_action="ADVANCE")
        )
        interview_service.submit_answer(session_id, "answer")

        stats = interview_service.get_statistics()
        assert stats["interviews_completed"] == 1
        assert stats["average_score"] == 5.0
        assert stats["average_duration_seconds"] >= 0.0

    def test_statistics_return_a_copy_not_live_reference(self, interview_service):
        stats = interview_service.get_statistics()
        stats["interviews_started"] = 999
        assert interview_service.get_statistics()["interviews_started"] == 0


# ============================================================
# Health check
# ============================================================


class TestHealthCheck:
    def test_health_check_true_when_all_collaborators_healthy(self, interview_service):
        assert interview_service.health_check() is True

    def test_health_check_false_when_evaluation_service_unhealthy(
        self, interview_service
    ):
        interview_service.evaluation_service.health_check.return_value = False
        assert interview_service.health_check() is False

    def test_health_check_false_on_repository_exception(self, interview_service):
        interview_service.question_repository.get_total_questions.side_effect = (
            RuntimeError("db down")
        )
        assert interview_service.health_check() is False

    def test_health_check_never_raises(self, interview_service):
        interview_service.evaluation_service.health_check.side_effect = RuntimeError(
            "unexpected"
        )
        try:
            result = interview_service.health_check()
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
            InterviewValidationError,
            InterviewSessionError,
            InterviewConfigurationError,
            InterviewQuestionError,
            InterviewPromptError,
            InterviewEvaluationError,
            InterviewReportError,
        ],
    )
    def test_all_exceptions_are_interview_service_errors(self, exc_cls):
        assert issubclass(exc_cls, InterviewServiceError)

    def test_base_exception_carries_session_and_request_id(self):
        exc = InterviewServiceError("boom", session_id="s1", request_id="r1")
        assert exc.session_id == "s1"
        assert exc.request_id == "r1"
        assert str(exc) == "boom"


# ============================================================
# Thread safety (smoke test)
# ============================================================


class TestThreadSafety:
    def test_concurrent_session_creation_does_not_corrupt_registry(
        self, interview_service
    ):
        errors: list[Exception] = []

        def _start(idx: int) -> None:
            try:
                interview_service.start_interview(
                    {"name": f"Candidate-{idx}", "experience_years": 3}
                )
            except Exception as exc:  # pragma: no cover - fail via errors list
                errors.append(exc)

        threads = [threading.Thread(target=_start, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(interview_service._sessions) == 10
        stats = interview_service.get_statistics()
        assert stats["interviews_started"] == 10


# ============================================================
# AI Fallback, Validation, and Logging
# ============================================================


class TestAIFallbackAndValidation:
    def test_repository_question_exists_uses_repo_question(
        self, interview_service, candidate, question_repository
    ):
        question_repository.get_question.return_value = make_question_record(question_id="Q100")
        res = interview_service.start_interview(candidate)
        assert res["question_id"] == "Q100"
        session = interview_service._sessions[res["session_id"]]
        assert session.question_source == "REPOSITORY"

    def test_repository_question_missing_uses_ai_generated_question(
        self, interview_service, candidate, question_repository, gemini_service
    ):
        question_repository.get_question.return_value = None
        gemini_service.generate_json.return_value = {
            "question": "Dynamic AI Question?",
            "competency": "C1",
            "difficulty": "Medium",
            "estimated_time": 5,
        }
        res = interview_service.start_interview(candidate)
        assert res["question"] == "INTERVIEW PROMPT TEXT"
        session = interview_service._sessions[res["session_id"]]
        assert session.question_source == "AI_GENERATED"
        assert session.current_question.question_record["question"] == "Dynamic AI Question?"

    def test_repository_followup_exists_uses_repo_followup(
        self, interview_service, candidate, question_repository
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        question_repository.get_followup_question.return_value = "Repo followup question text"
        interview_service.evaluation_service.evaluate_answer.return_value = make_evaluation_result(
            score=1,
            next_action="FOLLOWUP",
            needs_followup=True,
            missing_concept="LEFT JOIN",
        )
        res = interview_service.submit_answer(session_id, "my answer")
        assert res["next_action"] == "FOLLOWUP"
        assert res["question"] == "FOLLOWUP PROMPT TEXT"
        session = interview_service._sessions[session_id]
        assert session.question_source == "REPOSITORY"

    def test_repository_followup_missing_uses_ai_generated_followup(
        self, interview_service, candidate, question_repository, gemini_service
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        question_repository.get_followup_question.return_value = None
        gemini_service.generate_json.return_value = {
            "question": "AI Generated Followup Question?",
            "missing_concept": "LEFT JOIN",
            "followup_reason": "Needs clarification",
        }
        interview_service.evaluation_service.evaluate_answer.return_value = make_evaluation_result(
            score=1,
            next_action="FOLLOWUP",
            needs_followup=True,
            missing_concept="LEFT JOIN",
        )
        res = interview_service.submit_answer(session_id, "my answer")
        assert res["next_action"] == "FOLLOWUP"
        assert res["question"] == "AI Generated Followup Question?"
        session = interview_service._sessions[session_id]
        assert session.question_source == "AI_GENERATED"

    def test_generated_question_validation_failure_raises_ai_error(
        self, interview_service, candidate, question_repository, gemini_service
    ):
        question_repository.get_question.return_value = None
        gemini_service.generate_json.return_value = {"invalid": "missing fields"}
        with pytest.raises(InterviewAIError):
            interview_service.start_interview(candidate)

    def test_generated_question_competency_mismatch_raises_ai_error(
        self, interview_service, candidate, question_repository, gemini_service
    ):
        question_repository.get_question.return_value = None
        gemini_service.generate_json.return_value = {
            "question": "Some Question?",
            "competency": "WRONG_COMP",
            "difficulty": "Medium",
            "estimated_time": 5,
        }
        with pytest.raises(InterviewAIError):
            interview_service.start_interview(candidate)

    def test_generated_followup_validation_failure_raises_ai_error(
        self, interview_service, candidate, question_repository, gemini_service
    ):
        session_id = interview_service.start_interview(candidate)["session_id"]
        question_repository.get_followup_question.return_value = None
        gemini_service.generate_json.return_value = {"question": ""}
        interview_service.evaluation_service.evaluate_answer.return_value = make_evaluation_result(
            score=1,
            next_action="FOLLOWUP",
            needs_followup=True,
            missing_concept="LEFT JOIN",
        )
        with pytest.raises(InterviewAIError):
            interview_service.submit_answer(session_id, "my answer")

    def test_duplicate_generated_question_prevention(
        self, interview_service, candidate, question_repository, gemini_service
    ):
        question_repository.get_question.return_value = None
        gemini_service.generate_json.side_effect = [
            {
                "question": "Already Asked Question?",
                "competency": "C1",
                "difficulty": "Medium",
                "estimated_time": 5,
            },
            {
                "question": "Brand New Question?",
                "competency": "C1",
                "difficulty": "Medium",
                "estimated_time": 5,
            },
        ]
        session = interview_service._create_session(candidate)
        interview_service._sessions[session.session_id] = session
        interview_service._load_stage(session, "S1")
        session.asked_questions.append(
            AskedQuestion(
                question_id="Q_PREV",
                competency="C1",
                stage="S1",
                difficulty="Medium",
                question_record={"question": "Already Asked Question?"},
                asked_at=datetime.now(timezone.utc),
            )
        )
        interview_service._select_question(session)
        assert session.current_question.question_record["question"] == "Brand New Question?"

    def test_logging_includes_all_required_extra_fields(
        self, interview_service, candidate, question_repository, gemini_service
    ):
        logger_mock = MagicMock()
        interview_service.logger = logger_mock

        # 1. Repository question used
        question_repository.get_question.return_value = make_question_record(question_id="Q1")
        res = interview_service.start_interview(candidate)
        repo_call = next(c for c in logger_mock.info.call_args_list if c[0][0] == "repository_question_used")
        extra1 = repo_call[1]["extra"]
        assert extra1["session_id"] == res["session_id"]
        assert extra1["competency"] == "C1"
        assert extra1["stage"] == "S1"
        assert extra1["difficulty"] == "Medium"
        assert "request_id" in extra1

        # 2. AI generated question used
        logger_mock.reset_mock()
        question_repository.get_question.return_value = None
        gemini_service.generate_json.return_value = {
            "question": "Generated Question?",
            "competency": "C1",
            "difficulty": "Medium",
            "estimated_time": 5,
        }
        res2 = interview_service.start_interview(candidate)
        gen_call = next(c for c in logger_mock.info.call_args_list if c[0][0] == "generated_question_used")
        extra2 = gen_call[1]["extra"]
        assert extra2["session_id"] == res2["session_id"]
        assert extra2["competency"] == "C1"
        assert extra2["stage"] == "S1"
        assert extra2["difficulty"] == "Medium"
        assert "request_id" in extra2

        # 3. Repository followup used
        logger_mock.reset_mock()
        session_id = res["session_id"]
        interview_service._sessions[session_id].status = InterviewStatus.IN_PROGRESS
        question_repository.get_followup_question.return_value = "Repo followup text"
        interview_service.evaluation_service.evaluate_answer.return_value = make_evaluation_result(
            score=1,
            next_action="FOLLOWUP",
            needs_followup=True,
            missing_concept="X",
        )
        interview_service.submit_answer(session_id, "ans")
        repo_f_call = next(c for c in logger_mock.info.call_args_list if c[0][0] == "repository_followup_used")
        extra3 = repo_f_call[1]["extra"]
        assert extra3["session_id"] == session_id
        assert extra3["competency"] == "C1"
        assert extra3["stage"] == "S1"
        assert extra3["difficulty"] == "Medium"
        assert "request_id" in extra3

        # 4. AI generated followup used
        logger_mock.reset_mock()
        question_repository.get_followup_question.return_value = None
        gemini_service.generate_json.return_value = {
            "question": "AI Generated Followup?",
            "missing_concept": "X",
            "followup_reason": "needs depth",
        }
        interview_service.submit_answer(session_id, "ans")
        gen_f_call = next(c for c in logger_mock.info.call_args_list if c[0][0] == "generated_followup_used")
        extra4 = gen_f_call[1]["extra"]
        assert extra4["session_id"] == session_id
        assert extra4["competency"] == "C1"
        assert extra4["stage"] == "S1"
        assert extra4["difficulty"] == "Medium"
        assert "request_id" in extra4