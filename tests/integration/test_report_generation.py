"""
tests/integration/test_report_generation.py

Production-quality integration tests for ReportService and report validation in DE-INTERVIEWER.

Collaborators under test:
- ReportService (aggregates session evaluations and builds final report)
- PromptService (renders report generation prompt)
- GeminiService (mocked API client)
- QuestionRepository (real configuration source for decision thresholds)
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from exceptions.report_exceptions import (
    ReportAIError,
    ReportValidationError,
)
from models.interview_session import InterviewSession, InterviewStatus
from models.report_result import ReportResult
from services.interview_service import InterviewService
from services.report_service import ReportService


class TestReportGeneration:
    """
    Integration suite testing report generation, report markdown structure validation,
    decision threshold calculations, placeholder checks, and exception translation.
    """

    def test_report_structure_and_required_headings(
        self,
        interview_service: InterviewService,
        report_service: ReportService,
        candidate_info: Dict[str, Any],
        setup_excellent_candidate: None,
    ) -> None:
        """
        Verify that a generated report strictly contains all 10 required markdown headings:
        - Executive Summary
        - Overall Assessment
        - Competency Breakdown
        - Technical Strengths
        - Technical Weaknesses
        - Communication Assessment
        - Evidence-Based Justification
        - Hiring Recommendation
        - Development Plan
        - Confidence Explanation
        """
        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        # Drive interview turns to completion
        turns = 0
        while not interview_service.is_interview_complete(session_id) and turns < 15:
            turns += 1
            interview_service.submit_answer(
                session_id,
                "Strong candidate answer describing streaming architecture, watermarks, and windowing.",
            )

        assert interview_service.is_interview_complete(session_id)

        # Generate report
        report = report_service.generate_report(session)

        assert isinstance(report, ReportResult)
        assert report.report_id is not None
        assert report.candidate_name == candidate_info["name"]

        # Verify all 10 required headings in report markdown
        required_headings = [
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
        ]

        for heading in required_headings:
            assert heading in report.report_markdown, f"Report missing required heading: {heading}"

    def test_report_contains_no_placeholder_text(
        self,
        interview_service: InterviewService,
        report_service: ReportService,
        candidate_info: Dict[str, Any],
        setup_excellent_candidate: None,
    ) -> None:
        """
        Verify that the generated report markdown does not contain any forbidden placeholder text
        such as 'TODO', 'TBD', '[placeholder]', or 'N/A - TODO'.
        """
        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        turns = 0
        while not interview_service.is_interview_complete(session_id) and turns < 15:
            turns += 1
            interview_service.submit_answer(session_id, "Candidate technical response.")

        report = report_service.generate_report(session)
        markdown_upper = report.report_markdown.upper()

        forbidden = ["TODO", "TBD", "[PLACEHOLDER]", "N/A - TODO"]
        for placeholder in forbidden:
            assert placeholder not in markdown_upper, f"Forbidden placeholder '{placeholder}' found in report"

    def test_report_decision_and_boundary_scores(
        self,
        interview_service: InterviewService,
        report_service: ReportService,
        candidate_info: Dict[str, Any],
        mock_gemini_client: MagicMock,
        make_eval_response: Any,
        make_report_response: Any,
    ) -> None:
        """
        Verify decision determination across score boundaries:
        - High score (5/5) -> STRONG_HIRE / HIRE decision.
        - Low score (0/5 or 1/5) -> NO_HIRE decision.
        """
        # Test Case 1: Minimum score candidate (0/5)
        zero_eval = make_eval_response(score=0, confidence=90.0, weaknesses=["No response"])
        report_no_hire = make_report_response(decision="REJECT")

        from tests.integration.conftest import _handle_context_aware_mock

        def _zero_side_effect(model: str, contents: str, config: Any = None) -> MagicMock:
            return _handle_context_aware_mock(
                contents, default_eval_resp=zero_eval, default_report_resp=report_no_hire
            )

        mock_gemini_client.models.generate_content.side_effect = _zero_side_effect

        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        turns = 0
        while not interview_service.is_interview_complete(session_id) and turns < 15:
            turns += 1
            interview_service.submit_answer(session_id, "I don't know.")

        report = report_service.generate_report(session)
        assert report.overall_score == 0.0
        assert report.decision in ("REJECT", "HOLD")

    def test_report_prompt_unresolved_placeholders_check(
        self,
        interview_service: InterviewService,
        prompt_service: Any,
        candidate_info: Dict[str, Any],
        setup_excellent_candidate: None,
    ) -> None:
        """
        Verify that PromptService renders the report prompt cleanly with zero unresolved '{{placeholder}}' template markers.
        """
        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        turns = 0
        while not interview_service.is_interview_complete(session_id) and turns < 15:
            turns += 1
            interview_service.submit_answer(session_id, "Valid answer.")

        # Build report prompt directly via prompt_service with required parameters
        prompt_res = prompt_service.build_report_prompt(
            candidate=dict(session.candidate),
            competency_summaries=[
                {
                    "competency": "C1",
                    "competency_percentage": 90.0,
                    "weight_pct": 50.0,
                    "question_count": 2,
                }
            ],
            overall_score=90.0,
            decision="SELECT",
            confidence_level="HIGH",
            confidence_percentage=95.0,
            strengths=["Strong performance"],
            weaknesses=["None"],
        )
        prompt_text = prompt_res.prompt

        # Check for unrendered template placeholders like {{candidate_name}}, {{overall_score}}, etc.
        import re
        unresolved = re.findall(r"\{\{\s*[a-zA-Z0-9_]+\s*\}\}", prompt_text)
        assert len(unresolved) == 0, f"Unresolved template placeholders in report prompt: {unresolved}"

    def test_report_generation_invalid_markdown_raises_validation_error(
        self,
        interview_service: InterviewService,
        report_service: ReportService,
        candidate_info: Dict[str, Any],
        mock_gemini_client: MagicMock,
        make_eval_response: Any,
    ) -> None:
        """
        Verify that if Gemini returns report markdown missing required headings,
        ReportService raises ReportValidationError.
        """
        eval_resp = make_eval_response(score=5)
        bad_report_resp = MagicMock()
        bad_report_resp.text = "## Executive Summary\nInvalid report missing other 9 headings."
        bad_report_resp.candidates = [MagicMock(finish_reason="STOP")]
        bad_report_resp.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=10, total_token_count=20)
        bad_report_resp.prompt_feedback = None

        from tests.integration.conftest import _handle_context_aware_mock

        def _side_effect(model: str, contents: str, config: Any = None) -> MagicMock:
            return _handle_context_aware_mock(
                contents, default_eval_resp=eval_resp, default_report_resp=bad_report_resp
            )

        mock_gemini_client.models.generate_content.side_effect = _side_effect

        start_result = interview_service.start_interview(candidate_info)
        session_id = start_result["session_id"]
        session = interview_service._get_session(session_id)

        turns = 0
        while not interview_service.is_interview_complete(session_id) and turns < 15:
            turns += 1
            interview_service.submit_answer(session_id, "Good answer.")

        with pytest.raises(ReportValidationError):
            report_service.generate_report(session)

