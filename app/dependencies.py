"""
app/dependencies.py

Dependency Injection container for the DE-INTERVIEWER FastAPI application.
Provides singleton service instances and supports dependency overrides for testing.
"""

import time
from functools import lru_cache
from typing import Dict, Any

from repository.question_repository import QuestionRepository
from repository.report_repository import ReportRepository
from services.prompt_service import PromptService
from services.gemini_service import GeminiService
from services.evaluation_service import EvaluationService
from services.report_service import ReportService
from services.interview_service import InterviewService

APP_START_TIME = time.time()


class ServiceContainer:
    """
    Holds lazily initialized singleton service instances.
    """

    def __init__(self) -> None:
        self._question_repository: QuestionRepository | None = None
        self._report_repository: ReportRepository | None = None
        self._prompt_service: PromptService | None = None
        self._gemini_service: GeminiService | None = None
        self._evaluation_service: EvaluationService | None = None
        self._report_service: ReportService | None = None
        self._interview_service: InterviewService | None = None
        self._reports_cache: Dict[str, Any] = {}

    @property
    def start_time(self) -> float:
        return APP_START_TIME

    @property
    def question_repository(self) -> QuestionRepository:
        if self._question_repository is None:
            self._question_repository = QuestionRepository()
        return self._question_repository

    @property
    def report_repository(self) -> ReportRepository:
        if self._report_repository is None:
            self._report_repository = ReportRepository()
        return self._report_repository

    @property
    def prompt_service(self) -> PromptService:
        if self._prompt_service is None:
            self._prompt_service = PromptService(
                question_repository=self.question_repository
            )
        return self._prompt_service

    @property
    def gemini_service(self) -> GeminiService:
        if self._gemini_service is None:
            self._gemini_service = GeminiService()
        return self._gemini_service

    @property
    def evaluation_service(self) -> EvaluationService:
        if self._evaluation_service is None:
            self._evaluation_service = EvaluationService(
                question_repository=self.question_repository,
                prompt_service=self.prompt_service,
                gemini_service=self.gemini_service,
            )
        return self._evaluation_service

    @property
    def report_service(self) -> ReportService:
        if self._report_service is None:
            self._report_service = ReportService(
                question_repository=self.question_repository,
                prompt_service=self.prompt_service,
                gemini_service=self.gemini_service,
            )
        return self._report_service

    @property
    def interview_service(self) -> InterviewService:
        if self._interview_service is None:
            self._interview_service = InterviewService(
                question_repository=self.question_repository,
                prompt_service=self.prompt_service,
                evaluation_service=self.evaluation_service,
                report_service=self.report_service,
                gemini_service=self.gemini_service,
            )
        return self._interview_service

    def save_report(self, report_result: Any) -> None:
        """Cache report result in-memory by report_id for GET /api/v1/reports/{report_id}."""
        report_id = getattr(report_result, "report_id", None) or getattr(report_result, "request_id", None)
        if report_id:
            self._reports_cache[report_id] = report_result

    def get_report_by_id(self, report_id: str) -> Any | None:
        return self._reports_cache.get(report_id)


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    """
    Get or create the global singleton ServiceContainer instance.
    """
    return ServiceContainer()


def get_question_repository() -> QuestionRepository:
    return get_container().question_repository


def get_report_repository() -> ReportRepository:
    return get_container().report_repository


def get_prompt_service() -> PromptService:
    return get_container().prompt_service


def get_gemini_service() -> GeminiService:
    return get_container().gemini_service


def get_evaluation_service() -> EvaluationService:
    return get_container().evaluation_service


def get_report_service() -> ReportService:
    return get_container().report_service


def get_interview_service() -> InterviewService:
    return get_container().interview_service
