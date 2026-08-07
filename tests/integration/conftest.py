"""
tests/integration/conftest.py

Shared pytest fixtures and test helpers for the DE-INTERVIEWER integration test suite.

This file provides real service instances (QuestionRepository, PromptService,
EvaluationService, InterviewService, ReportService) wired together, with only
the external Gemini API client mocked via google.genai.Client.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

# Ensure project root directory is in sys.path for package imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from config.settings import settings
from exceptions.gemini_exceptions import (
    GeminiRateLimitError,
    GeminiSafetyError,
    GeminiTimeoutError,
)
from google.genai import types as genai_types
from repository.question_repository import QuestionRepository
from services.evaluation_service import EvaluationService
from services.gemini_service import GeminiService
from services.interview_service import InterviewService
from services.prompt_service import PromptService
from services.report_service import ReportService


# ============================================================
# Helper Functions & Factories
# ============================================================

def create_mock_sdk_response(
    text: str,
    finish_reason: str = "STOP",
    prompt_tokens: int = 100,
    response_tokens: int = 100,
) -> MagicMock:
    """
    Construct a MagicMock mimicking google.genai.types.GenerateContentResponse.
    Matches the structure expected by GeminiService._parse_response.
    """
    mock_response = MagicMock()
    mock_response.text = text

    mock_candidate = MagicMock()
    mock_candidate.finish_reason = finish_reason
    mock_candidate.safety_ratings = []
    mock_response.candidates = [mock_candidate]

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = response_tokens
    mock_usage.total_token_count = prompt_tokens + response_tokens
    mock_response.usage_metadata = mock_usage

    mock_response.prompt_feedback = None
    mock_response.model_version = settings.GEMINI_MODEL
    return mock_response


def default_evaluation_payload(
    score: int = 5,
    confidence: float = 90.0,
    strengths: Optional[List[str]] = None,
    weaknesses: Optional[List[str]] = None,
    missing_concepts: Optional[List[str]] = None,
    evidence: Optional[List[str]] = None,
    recommendation: str = "STRONG_PASS",
) -> Dict[str, Any]:
    """
    Generate a valid evaluation JSON dict required by EvaluationService schema.
    Note: score must be an int (0-5), confidence numeric (0-100).
    """
    return {
        "score": score,
        "confidence": confidence,
        "strengths": strengths if strengths is not None else ["Strong domain knowledge", "Clear explanation"],
        "weaknesses": weaknesses if weaknesses is not None else [],
        "missing_concepts": missing_concepts if missing_concepts is not None else [],
        "evidence": evidence if evidence is not None else ["Provided accurate technical details."],
        "recommendation": recommendation,
    }


_q_counter = 0


def default_question_payload(
    competency: str = "C1", difficulty: str = "Medium"
) -> Dict[str, Any]:
    """
    Generate a valid question JSON dict required by InterviewService AI question generation.
    """
    global _q_counter
    _q_counter += 1
    return {
        "question": f"Explain data engineering pipeline architecture for {competency} ({difficulty}) - topic {_q_counter}.",
        "competency": competency,
        "difficulty": difficulty,
        "estimated_time": 5,
        "learning_objective": f"Dynamic question objective for {competency}",
        "business_context": "Dynamic question context for pipeline design",
        "expected_concepts": ["partitioning", "streaming", "data_warehousing"],
        "evaluator_notes": "Scoring guidance for evaluators",
        "positive_indicators": ["mentions key distributed patterns"],
        "negative_indicators": ["confuses indexing with partitioning"],
    }


def default_followup_payload(
    missing_concept: str = "partitioning",
    followup_reason: str = "needs additional technical depth",
) -> Dict[str, Any]:
    """
    Generate a valid followup JSON dict required by InterviewService AI followup generation.
    """
    concept = (
        missing_concept.strip()
        if isinstance(missing_concept, str) and missing_concept.strip()
        else "partitioning"
    )
    if concept.startswith("[") or concept.startswith("None") or not concept:
        concept = "partitioning"
    return {
        "question": f"Could you elaborate on {concept} and how it applies here?",
        "missing_concept": concept,
        "followup_reason": followup_reason or "needs additional technical depth",
    }


def default_report_markdown(
    overall_assessment: str = "Overall score is high.",
    decision: str = "SELECT",
) -> str:
    """
    Generate a valid report markdown string containing all 10 required headings
    in exact required order per ReportService contract.
    """
    return f"""## Executive Summary
The candidate performed exceptionally well across all Data Engineering competencies.

## Overall Assessment
{overall_assessment}

## Competency Breakdown
- Data Architecture: 5.0 / 5.0
- Data Warehousing: 5.0 / 5.0

## Technical Strengths
- Deep understanding of distributed data processing.
- Strong SQL and database optimization skills.

## Technical Weaknesses
- None observed during this evaluation session.

## Communication Assessment
Clear, structured, and articulate technical communication throughout.

## Evidence-Based Justification
Candidate provided comprehensive answers with concrete examples.

## Hiring Recommendation
{decision}

## Development Plan
Continue deepening expertise in real-time streaming architectures.

## Confidence Explanation
High confidence based on detailed evidence gathered across multiple competencies.
"""


# ============================================================
# Core Service Fixtures
# ============================================================

@pytest.fixture
def candidate_info() -> Dict[str, Any]:
    """
    Standard candidate profile matching InterviewService expectations.
    """
    return {
        "name": "Alex Taylor",
        "experience_years": 1.0,
        "candidate_level": "Junior",
        "target_role": "Data Engineer",
        "email": "alex.taylor@example.com",
    }


import re

def _handle_context_aware_mock(
    contents: str,
    default_eval_resp: Optional[MagicMock] = None,
    default_report_resp: Optional[MagicMock] = None,
) -> MagicMock:
    """
    Inspect prompt text and return appropriate response payload schema based on request type.
    """
    contents_str = str(contents)

    # 1. Dynamic Question Generation
    if "Question Generation Agent" in contents_str or "DYNAMIC_QUESTION" in contents_str:
        comp_match = re.search(r"Competency:\s*([A-Za-z0-9_]+)", contents_str)
        diff_match = re.search(r"Difficulty:\s*([A-Za-z0-9_]+)", contents_str)
        comp = comp_match.group(1) if comp_match else "C1"
        diff = diff_match.group(1) if diff_match else "Medium"
        q_payload = default_question_payload(competency=comp, difficulty=diff)
        return create_mock_sdk_response(json.dumps(q_payload))

    # 2. Dynamic Follow-up Generation
    if "Follow-up Question Generation Agent" in contents_str or "DYNAMIC_FOLLOWUP" in contents_str:
        missing_match = re.search(r"Missing Concepts:\s*(.+)", contents_str)
        missing_raw = missing_match.group(1).strip() if missing_match else "partitioning"
        # Clean list formatting if present (e.g. ['Partitioning', 'Data Skew'])
        missing_clean = re.sub(r"[\[\]'\"`]", "", missing_raw).split(",")[0].strip()
        f_payload = default_followup_payload(missing_concept=missing_clean)
        return create_mock_sdk_response(json.dumps(f_payload))

    # 3. Report Generation
    if "REPORT" in contents_str or "final hiring report" in contents_str or "EXECUTIVE SUMMARY" in contents_str:
        if default_report_resp is not None:
            return default_report_resp
        return create_mock_sdk_response(default_report_markdown())

    # 4. Evaluation Request
    if default_eval_resp is not None:
        return default_eval_resp
    return create_mock_sdk_response(json.dumps(default_evaluation_payload()))


@pytest.fixture
def mock_gemini_client() -> MagicMock:
    """
    Mocked google.genai.Client instance.
    Context-aware response generator matching requested payload schema per prompt type.
    """
    mock_client = MagicMock()

    def _default_side_effect(model: str, contents: str, config: Any = None) -> MagicMock:
        return _handle_context_aware_mock(contents)

    mock_client.models.generate_content.side_effect = _default_side_effect
    return mock_client


@pytest.fixture
def question_repository() -> QuestionRepository:
    """
    Real QuestionRepository instance loading actual question bank & YAML configs.
    """
    return QuestionRepository()


@pytest.fixture
def prompt_service(question_repository: QuestionRepository) -> PromptService:
    """
    Real PromptService instance initialized with real QuestionRepository.
    """
    return PromptService(question_repository=question_repository)


@pytest.fixture
def gemini_service(mock_gemini_client: MagicMock) -> GeminiService:
    """
    Real GeminiService instance using injected mock_gemini_client and project settings.
    """
    return GeminiService(
        client=mock_gemini_client,
    )


@pytest.fixture
def evaluation_service(
    question_repository: QuestionRepository,
    prompt_service: PromptService,
    gemini_service: GeminiService,
) -> EvaluationService:
    """
    Real EvaluationService instance collaborating with real services and mock Gemini client.
    """
    return EvaluationService(
        question_repository=question_repository,
        prompt_service=prompt_service,
        gemini_service=gemini_service,
    )


@pytest.fixture
def report_service(
    question_repository: QuestionRepository,
    prompt_service: PromptService,
    gemini_service: GeminiService,
) -> ReportService:
    """
    Real ReportService instance collaborating with real services and mock Gemini client.
    """
    return ReportService(
        question_repository=question_repository,
        prompt_service=prompt_service,
        gemini_service=gemini_service,
    )


@pytest.fixture
def interview_service(
    question_repository: QuestionRepository,
    prompt_service: PromptService,
    gemini_service: GeminiService,
    evaluation_service: EvaluationService,
    report_service: ReportService,
) -> InterviewService:
    """
    Real InterviewService instance orchestrating all system services.
    """
    return InterviewService(
        question_repository=question_repository,
        prompt_service=prompt_service,
        gemini_service=gemini_service,
        evaluation_service=evaluation_service,
        report_service=report_service,
    )


# ============================================================
# Response Generator Fixtures & Helpers
# ============================================================

@pytest.fixture
def make_eval_response() -> Callable[..., MagicMock]:
    """
    Factory fixture to build custom evaluation SDK responses.
    """
    def _make(**kwargs: Any) -> MagicMock:
        payload = default_evaluation_payload(**kwargs)
        return create_mock_sdk_response(json.dumps(payload))
    return _make


@pytest.fixture
def make_report_response() -> Callable[..., MagicMock]:
    """
    Factory fixture to build custom report SDK responses.
    """
    def _make(markdown_text: Optional[str] = None, **kwargs: Any) -> MagicMock:
        text = markdown_text if markdown_text is not None else default_report_markdown(**kwargs)
        return create_mock_sdk_response(text)
    return _make


# ============================================================
# Reusable Session & Candidate Scenario Helpers
# ============================================================

@pytest.fixture
def setup_excellent_candidate(mock_gemini_client: MagicMock, make_eval_response: Callable[..., MagicMock], make_report_response: Callable[..., MagicMock]) -> None:
    """
    Configure mock Gemini to simulate an excellent candidate (score 5, no followups).
    When report generation occurs, returns valid report markdown.
    """
    eval_resp = make_eval_response(score=5, confidence=95.0, missing_concepts=[], weaknesses=[])
    report_resp = make_report_response(decision="SELECT")

    def _side_effect(model: str, contents: str, config: Any = None) -> MagicMock:
        return _handle_context_aware_mock(contents, default_eval_resp=eval_resp, default_report_resp=report_resp)

    mock_gemini_client.models.generate_content.side_effect = _side_effect


@pytest.fixture
def setup_weak_candidate(mock_gemini_client: MagicMock, make_eval_response: Callable[..., MagicMock], make_report_response: Callable[..., MagicMock]) -> None:
    """
    Configure mock Gemini to simulate a weak candidate (score 1, missing concepts, weaknesses).
    """
    eval_resp = make_eval_response(
        score=1,
        confidence=80.0,
        missing_concepts=["Partitioning", "Data Skew"],
        weaknesses=["Lacks fundamental understanding of distributed state"],
        recommendation="FAIL",
    )
    report_resp = make_report_response(decision="REJECT")

    def _side_effect(model: str, contents: str, config: Any = None) -> MagicMock:
        return _handle_context_aware_mock(contents, default_eval_resp=eval_resp, default_report_resp=report_resp)

    mock_gemini_client.models.generate_content.side_effect = _side_effect


# ============================================================
# Failure Simulation Helpers for Error Handling Tests
# ============================================================

@pytest.fixture
def simulate_gemini_invalid_json(mock_gemini_client: MagicMock) -> None:
    """
    Configure Gemini to return malformed JSON text.
    """
    bad_response = create_mock_sdk_response("THIS IS NOT VALID JSON {[[")
    mock_gemini_client.models.generate_content.side_effect = None
    mock_gemini_client.models.generate_content.return_value = bad_response


@pytest.fixture
def simulate_gemini_safety_block(mock_gemini_client: MagicMock) -> None:
    """
    Configure Gemini to return a response blocked by safety filters.
    """
    blocked_response = create_mock_sdk_response("Blocked content", finish_reason="SAFETY")
    mock_gemini_client.models.generate_content.side_effect = None
    mock_gemini_client.models.generate_content.return_value = blocked_response

