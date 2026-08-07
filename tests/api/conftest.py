"""
tests/api/conftest.py

Pytest fixtures for API testing using FastAPI TestClient and mock services.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_container, ServiceContainer
from tests.integration.conftest import (
    mock_gemini_client,
    question_repository,
    prompt_service,
    make_eval_response,
    make_report_response,
)


@pytest.fixture
def api_client(mock_gemini_client: MagicMock) -> TestClient:
    """
    FastAPI TestClient with mock Gemini client backing all dependency-injected services.
    """
    get_container.cache_clear()
    container = get_container()
    container.gemini_service._client = mock_gemini_client
    container.evaluation_service.gemini_service._client = mock_gemini_client
    container.report_service.gemini_service._client = mock_gemini_client
    container.interview_service.gemini_service._client = mock_gemini_client

    client = TestClient(app)
    yield client

    get_container.cache_clear()
