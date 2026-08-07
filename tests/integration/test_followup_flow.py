"""
tests/integration/test_followup_flow.py

Production-quality integration tests verifying the adaptive follow-up workflow
in DE-INTERVIEWER.

Collaborators under test:
- InterviewService (orchestrates question progression and follow-up limits)
- EvaluationService (detects weak/partial answers requiring follow-up)
- PromptService (renders adaptive follow-up prompts)
- GeminiService (mocked API client)
- QuestionRepository (real configuration & question bank)
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from models.interview_session import InterviewStatus
from services.interview_service import InterviewService


class TestFollowupFlow:
    """
    Integration suite testing adaptive follow-up questioning, candidate clarification,
    max follow-up cap enforcement, and progression back to normal flow.
    """

    def test_single_followup_trigger_and_resolution(
        self,
        interview_service: InterviewService,
        candidate_info: Dict[str, Any],
        mock_gemini_client: MagicMock,
        make_eval_response: Any,
    ) -> None:
        """
        Verify the standard follow-up lifecycle:
        1. Candidate gives a weak/partial answer (score=2, needs_followup=True).
        2. InterviewService returns next_action="FOLLOWUP" with rendered followup prompt.
        3. Candidate submits clarification answer.
        4. Gemini evaluates clarification with high score (score=5).
        5. InterviewService resolves follow-up and advances to next main question.
        """
        # Step 1: Start interview
        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        initial_q_id = start_result["question_id"]

        # Step 2: Configure Gemini to trigger a follow-up on 1st answer, then pass on 2nd
        weak_eval = make_eval_response(
            score=2,
            confidence=75.0,
            missing_concepts=["Partition Key Selection"],
            weaknesses=["Vague response on partitioning strategy"],
            recommendation="NEEDS_CLARIFICATION",
        )
        strong_eval = make_eval_response(
            score=3,
            confidence=95.0,
            missing_concepts=[],
            weaknesses=[],
            recommendation="PASS",
        )

        responses = [weak_eval, strong_eval]

        def _side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            if responses:
                return responses.pop(0)
            return strong_eval

        mock_gemini_client.models.generate_content.side_effect = _side_effect

        # Step 3: Submit initial weak answer
        weak_answer = "I would just use partition keys to split data."
        submit_1 = interview_service.submit_answer(session_id, weak_answer)

        assert submit_1["next_action"] == "FOLLOWUP"
        assert submit_1["question"] is not None
        assert len(submit_1["question"]) > 0

        session = interview_service._get_session(session_id)
        assert session.pending_followup_prompt_metadata is not None
        assert session.followup_counts.get(initial_q_id, 0) == 1

        # Step 4: Submit clarification answer for follow-up
        clarification = "To be specific, I would use hash partitioning on user_id to prevent data skew across nodes."
        submit_2 = interview_service.submit_answer(session_id, clarification)

        assert submit_2["next_action"] in ("ADVANCE", "NEXT_QUESTION", "STAGE_ADVANCED", "COMPLETE", "INTERVIEW_COMPLETE")

        session_after = interview_service._get_session(session_id)
        assert session_after.pending_followup_prompt_metadata is None

    def test_maximum_followups_enforcement_per_question(
        self,
        interview_service: InterviewService,
        candidate_info: Dict[str, Any],
        mock_gemini_client: MagicMock,
        make_eval_response: Any,
    ) -> None:
        """
        Verify cap on follow-up questions:
        Even if Gemini repeatedly returns low scores (score=1), InterviewService MUST
        NOT exceed the configured maximum_followups (2 per question/topic).
        After hitting 2 follow-ups, it must force advancement to the next question.
        """
        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        initial_q_id = start_result["question_id"]

        # Gemini persistently returns low score (score=1) requiring clarification
        persistent_weak_eval = make_eval_response(
            score=1,
            confidence=80.0,
            missing_concepts=["Indexing", "Query Execution Plan"],
            weaknesses=["Inability to elaborate"],
            recommendation="FAIL",
        )
        from tests.integration.conftest import _handle_context_aware_mock

        def _persistent_weak_side_effect(model: str, contents: str, config: Any = None) -> MagicMock:
            return _handle_context_aware_mock(contents, default_eval_resp=persistent_weak_eval)

        mock_gemini_client.models.generate_content.side_effect = _persistent_weak_side_effect

        # Turn 1: Main question -> Answer poorly -> triggers Follow-up 1
        sub1 = interview_service.submit_answer(session_id, "Bad answer 1")
        assert sub1["next_action"] == "FOLLOWUP"

        session = interview_service._get_session(session_id)
        assert session.pending_followup_prompt_metadata is not None
        assert session.followup_counts.get(initial_q_id, 0) == 1

        # Turn 2: Follow-up 1 -> Answer poorly again -> triggers Follow-up 2
        sub2 = interview_service.submit_answer(session_id, "Bad answer 2")
        assert sub2["next_action"] == "FOLLOWUP"

        session = interview_service._get_session(session_id)
        assert session.pending_followup_prompt_metadata is not None
        assert session.followup_counts.get(initial_q_id, 0) == 2

        # Turn 3: Follow-up 2 -> Answer poorly again -> MAX REACHED (2), MUST ADVANCE!
        sub3 = interview_service.submit_answer(session_id, "Bad answer 3")
        assert sub3["next_action"] in ("ADVANCE", "NEXT_QUESTION", "STAGE_ADVANCED", "COMPLETE", "INTERVIEW_COMPLETE")

        session_after = interview_service._get_session(session_id)
        assert session_after.pending_followup_prompt_metadata is None

    def test_followup_evaluation_history_accumulation(
        self,
        interview_service: InterviewService,
        candidate_info: Dict[str, Any],
        mock_gemini_client: MagicMock,
        make_eval_response: Any,
    ) -> None:
        """
        Verify that evaluations from follow-up turns are properly recorded
        in session.evaluation_history and factored into competency metrics.
        """
        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]

        eval1 = make_eval_response(score=2, weaknesses=["Initial weakness"])
        eval2 = make_eval_response(score=3, strengths=["Corrected in followup"])

        responses = [eval1, eval2]

        def _side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            if responses:
                return responses.pop(0)
            return eval2

        mock_gemini_client.models.generate_content.side_effect = _side_effect

        interview_service.submit_answer(session_id, "Initial attempt")
        interview_service.submit_answer(session_id, "Followup attempt")

        session = interview_service._get_session(session_id)
        assert len(session.evaluation_history) == 2
        assert session.evaluation_history[0].score == 2
        assert session.evaluation_history[1].score == 3

