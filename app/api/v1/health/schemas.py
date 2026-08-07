"""
app/api/v1/health/schemas.py

Pydantic data payload models for health check & diagnostic endpoints.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field


class HealthData(BaseModel):
    status: str = Field(..., description="Overall platform status (healthy, degraded, unhealthy).")
    version: str = Field(default="1.0.0", description="Application version.")
    uptime_seconds: float = Field(..., description="Application process uptime in seconds.")
    services: Dict[str, str] = Field(..., description="Subsystem operational status (UP, DOWN, DEGRADED).")


class ConfigData(BaseModel):
    stages_count: int = Field(..., description="Number of active interview stages.")
    competencies_count: int = Field(..., description="Number of evaluated competencies.")
    question_count: int = Field(..., description="Total loaded questions.")
    gemini_model: str = Field(..., description="Active Gemini model.")
    decision_thresholds: Dict[str, Any] = Field(..., description="Configured decision score thresholds.")
