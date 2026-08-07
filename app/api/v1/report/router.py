"""
app/api/v1/report/router.py

FastAPI router for report retrieval endpoints under /api/v1/reports.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.v1.schemas_common import APIResponse, wrap_response
from app.api.v1.report.schemas import ReportData
from app.dependencies import get_container, ServiceContainer

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get(
    "/{report_id}",
    response_model=APIResponse[ReportData],
    status_code=status.HTTP_200_OK,
    summary="Get Hiring Report",
    description="Retrieve a previously generated candidate hiring evaluation report by ID.",
)
def get_report(
    report_id: str,
    raw_request: Request,
    container: ServiceContainer = Depends(get_container),
) -> Dict[str, Any]:
    report_result = container.get_report_by_id(report_id)
    if not report_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report with ID '{report_id}' was not found.",
        )

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
