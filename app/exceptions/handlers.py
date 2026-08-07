"""
app/exceptions/handlers.py

Centralized global exception handlers converting domain exceptions into consistent ErrorEnvelope HTTP responses.
"""

import logging
from datetime import datetime
from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.api.v1.schemas_common import ErrorEnvelope, ErrorPayload
from exceptions.interview_exceptions import (
    InterviewSessionError,
    InterviewValidationError,
    InterviewConfigurationError,
    InterviewReportError,
)
from exceptions.evaluation_exceptions import (
    EvaluationValidationError,
    EvaluationAIError,
    EvaluationRepositoryError,
    EvaluationConfigurationError,
)
from exceptions.gemini_exceptions import (
    GeminiTimeoutError,
    GeminiSafetyError,
    GeminiRateLimitError,
    GeminiServiceError,
)
from exceptions.report_exceptions import ReportValidationError
from repository.question_repository import QuestionNotFoundError

logger = logging.getLogger("de_interviewer.api.exceptions")


def _create_error_json_response(
    request: Request,
    code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    envelope = ErrorEnvelope(
        success=False,
        error=ErrorPayload(code=code, message=message, details=details),
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat(),
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(),
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("Validation failure on %s: %s", request.url.path, str(exc))
    return _create_error_json_response(
        request,
        code="VALIDATION_ERROR",
        message=str(exc),
        status_code=status.HTTP_400_BAD_REQUEST,
    )


async def session_exception_handler(request: Request, exc: InterviewSessionError) -> JSONResponse:
    logger.warning("Interview session error on %s: %s", request.url.path, str(exc))
    is_not_found = "not found" in str(exc).lower()
    status_code = status.HTTP_404_NOT_FOUND if is_not_found else status.HTTP_400_BAD_REQUEST
    error_code = "SESSION_NOT_FOUND" if is_not_found else "SESSION_ERROR"
    return _create_error_json_response(
        request,
        code=error_code,
        message=str(exc),
        status_code=status_code,
        details={"session_id": getattr(exc, "session_id", None)},
    )


async def question_not_found_handler(request: Request, exc: QuestionNotFoundError) -> JSONResponse:
    logger.warning("Question not found on %s: %s", request.url.path, str(exc))
    return _create_error_json_response(
        request,
        code="QUESTION_NOT_FOUND",
        message=str(exc),
        status_code=status.HTTP_404_NOT_FOUND,
    )


async def timeout_exception_handler(request: Request, exc: GeminiTimeoutError) -> JSONResponse:
    logger.error("Gemini timeout on %s: %s", request.url.path, str(exc))
    return _create_error_json_response(
        request,
        code="SERVICE_TIMEOUT",
        message="AI generation timed out. Please retry.",
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
    )


async def rate_limit_exception_handler(request: Request, exc: GeminiRateLimitError) -> JSONResponse:
    logger.warning("Gemini rate limit hit on %s: %s", request.url.path, str(exc))
    return _create_error_json_response(
        request,
        code="RATE_LIMIT_EXCEEDED",
        message="Service rate limit exceeded. Please wait before retrying.",
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    )


async def report_validation_exception_handler(request: Request, exc: ReportValidationError) -> JSONResponse:
    logger.error("Report validation error on %s: %s", request.url.path, str(exc))
    return _create_error_json_response(
        request,
        code="REPORT_VALIDATION_ERROR",
        message=str(exc),
        status_code=status.HTTP_400_BAD_REQUEST,
    )


async def configuration_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Configuration error on %s: %s", request.url.path, str(exc))
    return _create_error_json_response(
        request,
        code="CONFIGURATION_ERROR",
        message="Platform configuration error.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


async def ai_service_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("AI Service Error on %s: %s", request.url.path, str(exc), exc_info=True)
    return _create_error_json_response(
        request,
        code="AI_SERVICE_ERROR",
        message=str(exc),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.critical("Unhandled exception on %s: %s", request.url.path, str(exc), exc_info=True)
    return _create_error_json_response(
        request,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


from fastapi import HTTPException


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
    return _create_error_json_response(
        request,
        code=code,
        message=str(exc.detail),
        status_code=exc.status_code,
    )


def register_exception_handlers(app) -> None:
    """Register all exception handlers on the FastAPI application."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(InterviewValidationError, validation_exception_handler)
    app.add_exception_handler(EvaluationValidationError, validation_exception_handler)
    app.add_exception_handler(ValueError, validation_exception_handler)
    app.add_exception_handler(InterviewSessionError, session_exception_handler)
    app.add_exception_handler(QuestionNotFoundError, question_not_found_handler)
    app.add_exception_handler(GeminiTimeoutError, timeout_exception_handler)
    app.add_exception_handler(GeminiRateLimitError, rate_limit_exception_handler)
    app.add_exception_handler(ReportValidationError, report_validation_exception_handler)
    app.add_exception_handler(InterviewConfigurationError, configuration_exception_handler)
    app.add_exception_handler(EvaluationConfigurationError, configuration_exception_handler)
    app.add_exception_handler(GeminiSafetyError, ai_service_exception_handler)
    app.add_exception_handler(EvaluationAIError, ai_service_exception_handler)
    app.add_exception_handler(InterviewReportError, ai_service_exception_handler)
    app.add_exception_handler(GeminiServiceError, ai_service_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
