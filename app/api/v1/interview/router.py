"""
app/api/v1/interview/router.py

FastAPI router for interview session management endpoints under /api/v1/interviews.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, Request, status

from app.api.v1.schemas_common import APIResponse, wrap_response
from app.api.v1.interview.schemas import (
    InterviewStartRequest,
    InterviewStartData,
    AnswerSubmitRequest,
    AnswerSubmitData,
    InterviewQuestionData,
    InterviewProgressData,
)
from app.api.v1.report.schemas import ReportData
from app.dependencies import (
    get_interview_service,
    get_report_service,
    get_container,
    InterviewService,
    ReportService,
)

router = APIRouter(prefix="/interviews", tags=["Interview Lifecycle"])


@router.post(
    "/start",
    response_model=APIResponse[InterviewStartData],
    status_code=status.HTTP_200_OK,
    summary="Start Interview Session",
    description="Initialize a new candidate interview session and return the first assigned question.",
)
def start_interview(
    request_data: InterviewStartRequest,
    raw_request: Request,
    interview_service: InterviewService = Depends(get_interview_service),
) -> Dict[str, Any]:
    candidate_info = {
        "name": request_data.candidate_name,
        "experience_years": request_data.experience,
        "candidate_level": request_data.candidate_level or "Mid",
        "target_role": request_data.target_role or "Data Engineer",
    }
    result = interview_service.start_interview(candidate_info)
    if "question" not in result and "prompt" in result:
        result["question"] = result["prompt"]
    request_id = getattr(raw_request.state, "request_id", None)
    return wrap_response(result, request_id=request_id)


@router.post(
    "/{session_id}/answer",
    response_model=APIResponse[AnswerSubmitData],
    status_code=status.HTTP_200_OK,
    summary="Submit Candidate Answer",
    description="Submit an answer for evaluation and receive evaluation results plus the next question.",
)
def submit_answer(
    session_id: str,
    request_data: AnswerSubmitRequest,
    raw_request: Request,
    interview_service: InterviewService = Depends(get_interview_service),
) -> Dict[str, Any]:
    result = interview_service.submit_answer(
        session_id=session_id, answer_text=request_data.answer
    )
    result["session_id"] = session_id
    if "question" not in result and "prompt" in result:
        result["question"] = result["prompt"]
    request_id = getattr(raw_request.state, "request_id", None)
    return wrap_response(result, request_id=request_id)


@router.get(
    "/{session_id}/question",
    response_model=APIResponse[InterviewQuestionData],
    status_code=status.HTTP_200_OK,
    summary="Get Current Question",
    description="Fetch active question details for an in-progress interview session.",
)
def get_current_question(
    session_id: str,
    raw_request: Request,
    interview_service: InterviewService = Depends(get_interview_service),
) -> Dict[str, Any]:
    result = interview_service.get_current_question(session_id) or {}
    result["session_id"] = session_id
    if "question" not in result and "prompt" in result:
        result["question"] = result["prompt"]
    request_id = getattr(raw_request.state, "request_id", None)
    return wrap_response(result, request_id=request_id)


@router.get(
    "/{session_id}/progress",
    response_model=APIResponse[InterviewProgressData],
    status_code=status.HTTP_200_OK,
    summary="Get Interview Progress",
    description="Retrieve progress percentage, completed questions, and competency coverage for a session.",
)
def get_interview_progress(
    session_id: str,
    raw_request: Request,
    interview_service: InterviewService = Depends(get_interview_service),
) -> Dict[str, Any]:
    raw_progress = interview_service.get_progress(session_id)
    mapped_progress = {
        "session_id": session_id,
        "current_stage": raw_progress.get("stage", "S1"),
        "completed_questions": raw_progress.get("questions_asked", 0),
        "remaining_questions": len(raw_progress.get("remaining_competencies", [])),
        "total_questions": len(raw_progress.get("covered_competencies", [])) + len(raw_progress.get("remaining_competencies", [])),
        "competencies_covered": raw_progress.get("covered_competencies", []),
        "progress_percentage": raw_progress.get("progress_percentage", 0.0),
        "elapsed_time_seconds": 0.0,
        "followups_asked": raw_progress.get("followups_asked", 0),
        "is_completed": raw_progress.get("status") == "COMPLETED",
        "current_question": interview_service.get_current_question(session_id) or {},
    }
    request_id = getattr(raw_request.state, "request_id", None)
    return wrap_response(mapped_progress, request_id=request_id)


@router.post(
    "/{session_id}/complete",
    response_model=APIResponse[ReportData],
    status_code=status.HTTP_200_OK,
    summary="Complete Interview & Generate Report",
    description="Finish the interview session and trigger final hiring report generation.",
)
def complete_interview(
    session_id: str,
    raw_request: Request,
    interview_service: InterviewService = Depends(get_interview_service),
    report_service: ReportService = Depends(get_report_service),
) -> Dict[str, Any]:
    session = interview_service._get_session(session_id)
    report_result = report_service.generate_report(session)

    container = get_container()
    container.save_report(report_result)

    report_dict = report_result.to_dict()
    report_dict["report_id"] = report_result.report_id

    mapped_competencies = []
    for comp in report_dict.get("competency_scores", []):
        mapped_competencies.append({
            "competency_id": comp.get("competency", ""),
            "competency_name": comp.get("competency_name", ""),
            "score": comp.get("average_score", 0.0),
            "percentage": comp.get("percentage", 0.0),
            "level": "PROFICIENT" if comp.get("percentage", 0.0) >= 70 else "DEVELOPING",
            "weight": 1.0,
            "questions_count": comp.get("question_count", 0),
        })
    report_dict["competency_scores"] = mapped_competencies

    request_id = getattr(raw_request.state, "request_id", None)
    return wrap_response(report_dict, request_id=request_id)
