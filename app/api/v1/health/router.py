"""
app/api/v1/health/router.py

FastAPI router exposing health monitoring and configuration endpoints.
"""

import time
from typing import Any, Dict
from fastapi import APIRouter, Depends, Request, status

from app.api.v1.schemas_common import APIResponse, wrap_response
from app.api.v1.health.schemas import HealthData, ConfigData
from app.dependencies import (
    get_container,
    get_question_repository,
    get_gemini_service,
    get_evaluation_service,
    get_interview_service,
    get_report_service,
    ServiceContainer,
    QuestionRepository,
    GeminiService,
    EvaluationService,
    InterviewService,
    ReportService,
)
from config.settings import settings

router = APIRouter(tags=["Health & Diagnostics"])


@router.get(
    "/health",
    response_model=APIResponse[HealthData],
    status_code=status.HTTP_200_OK,
    summary="Subsystem Health Check",
    description="Check uptime and sub-service health statuses.",
)
def get_health(
    raw_request: Request,
    container: ServiceContainer = Depends(get_container),
    question_repo: QuestionRepository = Depends(get_question_repository),
    gemini_svc: GeminiService = Depends(get_gemini_service),
    eval_svc: EvaluationService = Depends(get_evaluation_service),
    report_svc: ReportService = Depends(get_report_service),
) -> Dict[str, Any]:
    uptime = time.time() - container.start_time
    services_status = {
        "repository": "UP" if question_repo.get_total_questions() > 0 else "DEGRADED",
        "gemini": "UP" if gemini_svc.health_check() else "DOWN",
        "prompt": "UP",
        "evaluation": "UP" if eval_svc.health_check() else "DEGRADED",
        "interview": "UP",
        "report": "UP" if report_svc.health_check() else "DEGRADED",
    }

    is_healthy = all(s == "UP" for s in services_status.values())
    overall = "healthy" if is_healthy else "degraded"

    data = {
        "status": overall,
        "version": "1.0.0",
        "uptime_seconds": round(uptime, 2),
        "services": services_status,
    }

    request_id = getattr(raw_request.state, "request_id", None)
    return wrap_response(data, request_id=request_id)


@router.get(
    "/config",
    response_model=APIResponse[ConfigData],
    status_code=status.HTTP_200_OK,
    summary="Runtime Configuration",
    description="Retrieve loaded stages, competencies, questions, and decision parameters.",
)
def get_config(
    raw_request: Request,
    question_repo: QuestionRepository = Depends(get_question_repository),
) -> Dict[str, Any]:
    ref_data = getattr(question_repo, "reference_data", {})
    stages = ref_data.get("stages", {}).get("stages", []) if isinstance(ref_data.get("stages"), dict) else []
    competencies = ref_data.get("competencies", {}).get("competencies", []) if isinstance(ref_data.get("competencies"), dict) else []
    thresholds = ref_data.get("thresholds", {}).get("decision_thresholds", {}) if isinstance(ref_data.get("thresholds"), dict) else {}

    data = {
        "stages_count": len(stages),
        "competencies_count": len(competencies),
        "question_count": question_repo.get_total_questions(),
        "gemini_model": settings.GEMINI_MODEL,
        "decision_thresholds": thresholds,
    }

    request_id = getattr(raw_request.state, "request_id", None)
    return wrap_response(data, request_id=request_id)
