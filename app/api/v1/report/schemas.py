"""
app/api/v1/report/schemas.py

Pydantic data payload models for report endpoints.
"""

from typing import List
from pydantic import BaseModel, Field


class CompetencyScoreItem(BaseModel):
    competency_id: str = Field(..., description="Competency ID.")
    competency_name: str = Field(..., description="Competency name.")
    score: float = Field(..., description="Average score.")
    percentage: float = Field(..., description="Percentage score.")
    level: str = Field(..., description="Proficiency level.")
    weight: float = Field(..., description="Weight multiplier.")
    questions_count: int = Field(..., description="Questions evaluated.")


class ReportData(BaseModel):
    report_id: str = Field(..., description="Unique report ID.")
    request_id: str = Field(..., description="Request correlation ID.")
    candidate_name: str = Field(..., description="Candidate name.")
    overall_score: float = Field(..., description="Overall weighted score.")
    overall_percentage: float = Field(..., description="Overall score percentage.")
    decision: str = Field(..., description="Hiring decision (SELECT, HOLD, REJECT).")
    confidence_level: str = Field(..., description="Aggregated confidence level.")
    confidence_percentage: float = Field(..., description="Aggregated confidence percentage.")
    competency_scores: List[CompetencyScoreItem] = Field(..., description="Per-competency breakdowns.")
    competency_summary: str = Field(..., description="Summary string.")
    strengths: List[str] = Field(..., description="Key strengths.")
    weaknesses: List[str] = Field(..., description="Identified weaknesses.")
    recommendations: List[str] = Field(..., description="Actionable recommendations.")
    report_markdown: str = Field(..., description="Full executive report in Markdown.")
    generated_at: str = Field(..., description="ISO 8601 UTC generation timestamp.")
