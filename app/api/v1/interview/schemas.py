"""
app/api/v1/interview/schemas.py

Pydantic request & data payload models for interview flow operations.
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class InterviewStartRequest(BaseModel):
    candidate_name: str = Field(..., min_length=1, description="Candidate's full name.", example="John Doe")
    experience: float = Field(..., ge=0.0, le=50.0, description="Years of professional experience.", example=3.5)
    candidate_level: Optional[str] = Field(default="Mid", description="Seniority level.", example="Mid")
    target_role: Optional[str] = Field(default="Data Engineer", description="Target role title.", example="Data Engineer")


class InterviewStartData(BaseModel):
    session_id: str = Field(..., description="Unique interview session ID.")
    question_id: str = Field(..., description="First question ID.")
    question: Optional[str] = Field(default="", description="Initial question text.")
    stage: str = Field(..., description="Current interview stage.")
    competency: str = Field(..., description="Target competency ID.")
    difficulty: Optional[str] = Field(default="Medium", description="Question difficulty.")
    progress: Dict[str, Any] = Field(default_factory=dict, description="Session progress overview.")


class AnswerSubmitRequest(BaseModel):
    answer: str = Field(..., min_length=1, description="Candidate's answer text.", example="Partitioning reduces data scan sizes by filtering folders.")


class AnswerSubmitData(BaseModel):
    session_id: str = Field(..., description="Interview session ID.")
    evaluation: Dict[str, Any] = Field(..., description="Answer evaluation output.")
    next_action: str = Field(..., description="Action recommendation (FOLLOWUP, ADVANCE, COMPLETE).")
    question: Optional[str] = Field(default=None, description="Next question text if continuing.")
    question_id: Optional[str] = Field(default=None, description="Next question ID if continuing.")
    is_completed: bool = Field(default=False, description="Whether the interview is complete.")
    progress: Dict[str, Any] = Field(default_factory=dict, description="Session progress overview.")


class InterviewQuestionData(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Interview session ID.")
    question_id: str = Field(..., description="Active question ID.")
    question: Optional[str] = Field(default="", description="Question text.")
    stage: str = Field(..., description="Active stage ID.")
    competency: str = Field(..., description="Target competency ID.")
    difficulty: Optional[str] = Field(default="Medium", description="Difficulty level.")
    is_followup: Optional[bool] = Field(default=False, description="Whether this is a dynamic follow-up.")


class InterviewProgressData(BaseModel):
    session_id: str = Field(..., description="Interview session ID.")
    current_stage: Optional[str] = Field(default="S1", description="Active stage ID.")
    completed_questions: Optional[int] = Field(default=0, description="Questions answered.")
    remaining_questions: Optional[int] = Field(default=0, description="Questions remaining.")
    total_questions: Optional[int] = Field(default=0, description="Total required questions.")
    competencies_covered: List[str] = Field(default_factory=list, description="List of competency IDs evaluated.")
    progress_percentage: float = Field(..., description="Session completion percentage.")
    elapsed_time_seconds: Optional[float] = Field(default=0.0, description="Session duration in seconds.")
    followups_asked: int = Field(..., description="Follow-ups triggered.")
    is_completed: Optional[bool] = Field(default=False, description="Completion state.")
    current_question: Dict[str, Any] = Field(default_factory=dict, description="Active question info.")
