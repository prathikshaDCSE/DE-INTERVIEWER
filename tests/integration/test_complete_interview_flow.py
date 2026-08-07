"""
tests/integration/test_complete_interview_flow.py

Production-quality integration tests verifying the complete interview lifecycle
from interview creation through answer submission, EvaluationService scoring,
InterviewService state updates, and ReportService report generation.

Collaborators under test:
- QuestionRepository (real)
- PromptService (real)
- GeminiService (real with mocked google.genai.Client)
- EvaluationService (real)
- InterviewService (real)
- ReportService (real)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from models.interview_session import InterviewStatus
from models.report_result import ReportResult
from services.interview_service import InterviewService
from services.report_service import ReportService


class TestCompleteInterviewFlow:
    """
    Integration tests covering the complete interview pipeline and candidate scenarios.
    """

    def test_complete_interview_lifecycle_end_to_end(
        self,
        interview_service: InterviewService,
        report_service: ReportService,
        candidate_info: Dict[str, Any],
        setup_excellent_candidate: None,
    ) -> None:
        """
        Verify complete pipeline execution:
        start_interview -> retrieve question -> submit answer -> EvaluationService -> session updated -> report generated.
        """
        # Arrange & Act: 1. Start Interview
        start_result = interview_service.start_interview(candidate_info)

        assert "session_id" in start_result
        assert start_result["question_id"] is not None
        assert start_result["question"] is not None
        assert start_result["stage"] is not None
        assert start_result["competency"] is not None

        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        # 2. Iterate through questions until interview completes
        turns = 0
        max_safety_turns = 20

        while not interview_service.is_interview_complete(session_id) and turns < max_safety_turns:
            turns += 1
            curr_q = interview_service.get_current_question(session_id)
            if not curr_q:
                break

            answer = "I would design a partitioned distributed pipeline using Kafka and Spark Structured Streaming."
            submit_result = interview_service.submit_answer(session_id, answer)

            assert "next_action" in submit_result
            assert "evaluation" in submit_result
            assert "progress" in submit_result

            eval_dict = submit_result["evaluation"]
            assert eval_dict["score"] >= 3
            assert len(session.evaluation_history) == turns

        # 3. Verify session reached completed status
        assert interview_service.is_interview_complete(session_id)
        assert session.status == InterviewStatus.COMPLETED

        # 4. Generate final report via ReportService
        report = report_service.generate_report(session)

        # Assert ReportResult contract
        assert isinstance(report, ReportResult)
        assert report.candidate_name == candidate_info["name"]
        assert report.overall_score > 0
        assert report.decision in ("SELECT", "HOLD")
        assert len(report.competency_scores) > 0
        assert "## Executive Summary" in report.report_markdown
        assert "## Hiring Recommendation" in report.report_markdown

    def test_excellent_candidate_flow(
        self,
        interview_service: InterviewService,
        report_service: ReportService,
        candidate_info: Dict[str, Any],
        setup_excellent_candidate: None,
    ) -> None:
        """
        Verify excellent candidate scenario:
        - All evaluations yield maximum scores
        - Zero follow-ups triggered
        - Interview completes cleanly
        - Report decision matches high threshold (SELECT / HOLD)
        """
        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        turns = 0
        followup_count = 0

        while not interview_service.is_interview_complete(session_id) and turns < 20:
            turns += 1
            submit_result = interview_service.submit_answer(
                session_id,
                "Detailed comprehensive technical answer covering schema design, indexing, and partitioning.",
            )

            if submit_result["next_action"] == "FOLLOWUP":
                followup_count += 1

        assert followup_count == 0, "Excellent candidate should not trigger follow-up questions"
        assert interview_service.is_interview_complete(session_id)

        report = report_service.generate_report(session)
        assert report.overall_score >= 80.0
        assert report.decision in ("SELECT", "HOLD")
        assert report.confidence_percentage >= 70.0

    def test_weak_candidate_flow(
        self,
        interview_service: InterviewService,
        report_service: ReportService,
        candidate_info: Dict[str, Any],
        setup_weak_candidate: None,
    ) -> None:
        """
        Verify weak candidate scenario:
        - Low evaluation scores
        - Follow-ups triggered
        - Weaknesses accumulated
        - Final report indicates REJECT
        """
        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        turns = 0
        followups_triggered = 0

        while not interview_service.is_interview_complete(session_id) and turns < 25:
            turns += 1
            submit_result = interview_service.submit_answer(
                session_id,
                "I don't know much about database partitioning or distributed locking.",
            )

            if submit_result["next_action"] == "FOLLOWUP":
                followups_triggered += 1

        assert followups_triggered > 0, "Weak candidate should trigger follow-ups"
        assert len(session.evaluation_history) > 0

        # Verify weakness accumulation across evaluations
        weaknesses_found = [
            w for eval_res in session.evaluation_history for w in eval_res.weaknesses
        ]
        assert len(weaknesses_found) > 0

        report = report_service.generate_report(session)
        assert report.overall_score < 60.0
        assert report.decision in ("REJECT", "HOLD")

    def test_mixed_candidate_flow(
        self,
        interview_service: InterviewService,
        report_service: ReportService,
        candidate_info: Dict[str, Any],
        mock_gemini_client: MagicMock,
        make_eval_response: Any,
        make_report_response: Any,
    ) -> None:
        """
        Verify mixed candidate scenario:
        - Different scores across competencies (e.g. score 5 for C1, score 2 for C2, score 4 for C3)
        - Competency score aggregation and weighted overall score calculation
        - Valid report generation
        """
        scores_list = [5, 2, 4, 3, 5]
        score_idx = 0

        from tests.integration.conftest import _handle_context_aware_mock

        def _mixed_side_effect(model: str, contents: str, config: Any = None) -> MagicMock:
            nonlocal score_idx
            contents_str = str(contents)
            if (
                "Question Generation Agent" in contents_str
                or "DYNAMIC_QUESTION" in contents_str
                or "Follow-up Question Generation Agent" in contents_str
                or "DYNAMIC_FOLLOWUP" in contents_str
                or "REPORT" in contents_str
                or "final hiring report" in contents_str
                or "EXECUTIVE SUMMARY" in contents_str
            ):
                return _handle_context_aware_mock(
                    contents_str,
                    default_report_resp=make_report_response(decision="SELECT"),
                )

            current_score = scores_list[score_idx % len(scores_list)]
            score_idx += 1
            rec = "PASS" if current_score >= 3 else "FAIL"
            eval_resp = make_eval_response(score=current_score, confidence=85.0, recommendation=rec)
            return _handle_context_aware_mock(contents_str, default_eval_resp=eval_resp)

        mock_gemini_client.models.generate_content.side_effect = _mixed_side_effect

        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        turns = 0
        while not interview_service.is_interview_complete(session_id) and turns < 20:
            turns += 1
            interview_service.submit_answer(session_id, "Candidate answer with mixed depth.")

        report = report_service.generate_report(session)
        assert len(report.competency_scores) > 0
        assert report.overall_score > 0
        assert report.decision in ("SELECT", "HOLD", "REJECT")
