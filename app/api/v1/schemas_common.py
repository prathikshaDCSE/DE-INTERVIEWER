"""
app/api/v1/schemas_common.py

Common envelope wrappers ensuring consistent JSON response schemas across all endpoints.
"""

from typing import Generic, TypeVar, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """
    Standardized success response envelope structure.
    """

    success: bool = Field(default=True, description="Success flag.")
    data: T = Field(..., description="Response payload data.")
    request_id: Optional[str] = Field(default=None, description="Correlation request ID.")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 UTC timestamp.",
    )


class ErrorPayload(BaseModel):
    code: str = Field(..., description="Error classification code.")
    message: str = Field(..., description="Human-readable message.")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Optional diagnostic details.")


class ErrorEnvelope(BaseModel):
    """
    Standardized error response envelope structure.
    """

    success: bool = Field(default=False, description="Success flag.")
    error: ErrorPayload = Field(..., description="Error payload details.")
    request_id: Optional[str] = Field(default=None, description="Correlation request ID.")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 UTC timestamp.",
    )


def wrap_response(data: Any, request_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to wrap dictionary or model payloads into the standard APIResponse envelope.
    """
    return {
        "success": True,
        "data": data,
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
