"""
tests/test_gemini_service.py

End-to-end unit tests for ``GeminiService``.

Nothing here ever calls the real Gemini API. Every test either
injects a ``MagicMock``/fake object as the SDK client (via
``GeminiService(..., client=...)``) or, for the handful of cases that
need to exercise the "no client was injected" construction path,
monkeypatches ``services.gemini_service.genai.Client`` itself.

Organisation
------------
* TestConfiguration        - __init__ / GeminiConfigurationError paths
* TestPromptValidation     - _validate_prompt / GeminiPromptError
* TestGenerationConfig     - _build_generation_config
* TestGenerateSuccess      - happy path for generate()
* TestRetryBehavior        - rate-limit retry loop
* TestTimeout              - timeout enforcement
* TestSafety               - safety-block detection
* TestResponseErrors       - malformed-response detection
* TestJsonParsing          - generate_json()
* TestGenerateText         - generate_text()
* TestHealthCheck          - health_check()
* TestBackoff              - _backoff_seconds
* TestDependencyInjection  - injected client / logger
* TestStripCodeFence       - _strip_code_fence, unit-level
* TestPromptResponseLogging- log_prompts opt-in debug logging
* TestZeroRetryConfig      - max_retries=0 behaves as "no retries"
* TestGeminiResponseModel  - GeminiResponse.__post_init__ / repr
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from exceptions.gemini_exceptions import (
    GeminiConfigurationError,
    GeminiJsonParseError,
    GeminiPromptError,
    GeminiRateLimitError,
    GeminiResponseError,
    GeminiSafetyError,
    GeminiServiceError,
    GeminiTimeoutError,
)
from models.gemini_response import GeminiResponse
from services import gemini_service as gemini_service_module
from services.gemini_service import GeminiService


# ======================================================================
# Fake SDK response builders
# ======================================================================


def make_sdk_response(
    text: str | None = "OK",
    finish_reason: str | None = "STOP",
    prompt_tokens: int | None = 10,
    response_tokens: int | None = 5,
    safety_ratings: list | None = None,
) -> SimpleNamespace:
    """A minimal stand-in for the SDK's GenerateContentResponse."""
    parts = [SimpleNamespace(text=text)] if text is not None else []
    candidate = SimpleNamespace(
        finish_reason=finish_reason,
        content=SimpleNamespace(parts=parts),
        safety_ratings=safety_ratings or [],
    )
    return SimpleNamespace(
        candidates=[candidate],
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=response_tokens,
        ),
        prompt_feedback=SimpleNamespace(block_reason=None),
    )


def make_no_candidates_response() -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[],
        text=None,
        usage_metadata=None,
        prompt_feedback=SimpleNamespace(block_reason=None),
    )


def make_blocked_prompt_response(block_reason: str = "SAFETY") -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[],
        text=None,
        usage_metadata=None,
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
    )


def make_safety_blocked_candidate_response() -> SimpleNamespace:
    candidate = SimpleNamespace(
        finish_reason="SAFETY",
        content=SimpleNamespace(parts=[]),
        safety_ratings=[
            SimpleNamespace(category="HARM_CATEGORY_HATE_SPEECH", blocked=True),
            SimpleNamespace(category="HARM_CATEGORY_HARASSMENT", blocked=False),
        ],
    )
    return SimpleNamespace(
        candidates=[candidate],
        text=None,
        usage_metadata=None,
        prompt_feedback=SimpleNamespace(block_reason=None),
    )


def make_no_text_response() -> SimpleNamespace:
    candidate = SimpleNamespace(
        finish_reason="STOP",
        content=SimpleNamespace(parts=[]),
        safety_ratings=[],
    )
    return SimpleNamespace(
        candidates=[candidate],
        text=None,
        usage_metadata=None,
        prompt_feedback=SimpleNamespace(block_reason=None),
    )


def make_client_error(code: int = 429, message: str = "quota exceeded") -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code, {"error": {"code": code, "message": message, "status": "RESOURCE_EXHAUSTED"}}
    )


def make_server_error(code: int = 500, message: str = "internal error") -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"error": {"code": code, "message": message}})


# ======================================================================
# Shared fixtures
# ======================================================================


@pytest.fixture
def fake_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def fake_logger() -> MagicMock:
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def service(fake_client: MagicMock) -> GeminiService:
    return GeminiService(
        model="gemini-test-model",
        client=fake_client,
        timeout_seconds=5,
        max_retries=2,
        max_backoff_seconds=1.0,
    )


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets sleep-free retries by default."""
    monkeypatch.setattr(gemini_service_module.time, "sleep", lambda _seconds: None)


# ======================================================================
# TestConfiguration
# ======================================================================


class TestConfiguration:
    def test_missing_model_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock
    ) -> None:
        monkeypatch.setattr(gemini_service_module.settings, "GEMINI_MODEL", "")
        with pytest.raises(GeminiConfigurationError) as exc_info:
            GeminiService(model=None, client=fake_client)
        assert exc_info.value.config_key == "GEMINI_MODEL"

    def test_missing_api_key_without_injected_client_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gemini_service_module.settings, "GOOGLE_API_KEY", "")
        with pytest.raises(GeminiConfigurationError) as exc_info:
            GeminiService(model="gemini-test-model")
        assert exc_info.value.config_key == "GOOGLE_API_KEY"

    def test_injected_client_bypasses_api_key_requirement(
        self, monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock
    ) -> None:
        monkeypatch.setattr(gemini_service_module.settings, "GOOGLE_API_KEY", "")
        # Should not raise, even though no API key is configured anywhere.
        svc = GeminiService(model="gemini-test-model", client=fake_client)
        assert svc._client is fake_client

    def test_sdk_client_construction_failure_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("network unreachable")

        monkeypatch.setattr(gemini_service_module.genai, "Client", boom)
        with pytest.raises(GeminiConfigurationError) as exc_info:
            GeminiService(api_key="fake-key", model="gemini-test-model")
        assert exc_info.value.config_key == "GOOGLE_API_KEY"

    def test_default_client_construction_passes_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel_client = MagicMock()
        captured_kwargs: dict = {}

        def fake_client_ctor(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return sentinel_client

        monkeypatch.setattr(gemini_service_module.genai, "Client", fake_client_ctor)
        svc = GeminiService(
            api_key="fake-key", model="gemini-test-model", timeout_seconds=15
        )
        assert svc._client is sentinel_client
        assert captured_kwargs["api_key"] == "fake-key"
        assert isinstance(captured_kwargs["http_options"], genai_types.HttpOptions)
        assert captured_kwargs["http_options"].timeout == 15000


# ======================================================================
# TestPromptValidation
# ======================================================================


class TestPromptValidation:
    def test_non_string_prompt_rejected(self, service: GeminiService, fake_client: MagicMock) -> None:
        with pytest.raises(GeminiPromptError):
            service.generate(12345)  # type: ignore[arg-type]
        fake_client.models.generate_content.assert_not_called()

    def test_empty_prompt_rejected(self, service: GeminiService, fake_client: MagicMock) -> None:
        with pytest.raises(GeminiPromptError):
            service.generate("")
        fake_client.models.generate_content.assert_not_called()

    def test_whitespace_only_prompt_rejected(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        with pytest.raises(GeminiPromptError):
            service.generate("   \n\t  ")
        fake_client.models.generate_content.assert_not_called()

    def test_prompt_over_max_chars_rejected(self, fake_client: MagicMock) -> None:
        svc = GeminiService(
            model="gemini-test-model", client=fake_client, max_prompt_chars=10
        )
        with pytest.raises(GeminiPromptError):
            svc.generate("this prompt is definitely too long")
        fake_client.models.generate_content.assert_not_called()

    def test_valid_prompt_reaches_client(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(text="hi")
        service.generate("a short valid prompt")
        fake_client.models.generate_content.assert_called_once()


# ======================================================================
# TestGenerationConfig
# ======================================================================


class TestGenerationConfig:
    def test_defaults_applied(self, fake_client: MagicMock) -> None:
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            default_temperature=0.3,
            default_max_output_tokens=256,
        )
        config = svc._build_generation_config()
        assert config.temperature == 0.3
        assert config.max_output_tokens == 256

    def test_overrides_take_precedence(self, fake_client: MagicMock) -> None:
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            default_temperature=0.3,
            default_max_output_tokens=256,
        )
        config = svc._build_generation_config(temperature=0.9)
        assert config.temperature == 0.9
        assert config.max_output_tokens == 256  # untouched default persists

    def test_no_defaults_means_sdk_defaults(self, fake_client: MagicMock) -> None:
        svc = GeminiService(model="gemini-test-model", client=fake_client)
        config = svc._build_generation_config()
        assert config.temperature is None
        assert config.max_output_tokens is None

    def test_config_flows_into_generate_content_call(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response()
        service.generate("hello", temperature=0.5)
        _, kwargs = fake_client.models.generate_content.call_args
        assert isinstance(kwargs["config"], genai_types.GenerateContentConfig)
        assert kwargs["config"].temperature == 0.5
        assert kwargs["model"] == "gemini-test-model"
        assert kwargs["contents"] == "hello"


# ======================================================================
# TestGenerateSuccess
# ======================================================================


class TestGenerateSuccess:
    def test_returns_populated_gemini_response(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(
            text="the answer", finish_reason="STOP", prompt_tokens=7, response_tokens=3
        )
        result = service.generate("question?")

        assert isinstance(result, GeminiResponse)
        assert result.text == "the answer"
        assert result.model == "gemini-test-model"
        assert result.finish_reason == "STOP"
        assert result.retry_count == 0
        assert result.prompt_tokens == 7
        assert result.response_tokens == 3
        assert result.total_tokens == 10
        assert result.request_id is not None
        assert result.latency_ms is not None and result.latency_ms >= 0
        assert result.timestamp is not None
        assert result.raw_response is not None

    def test_single_network_call_on_success(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response()
        service.generate("hello")
        assert fake_client.models.generate_content.call_count == 1

    def test_falls_back_to_content_parts_when_no_top_level_text(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        response = make_sdk_response(text="from parts")
        response.text = None  # force the fallback path in _parse_response
        fake_client.models.generate_content.return_value = response
        result = service.generate("hello")
        assert result.text == "from parts"


# ======================================================================
# TestRetryBehavior
# ======================================================================


class TestRetryBehavior:
    def test_retries_on_rate_limit_then_succeeds(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.side_effect = [
            make_client_error(429),
            make_client_error(429),
            make_sdk_response(text="finally"),
        ]
        result = service.generate("hello")
        assert result.text == "finally"
        assert result.retry_count == 2
        assert fake_client.models.generate_content.call_count == 3

    def test_exhausts_retry_budget_and_raises_rate_limit_error(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        # service fixture has max_retries=2 -> 3 total attempts, all fail.
        fake_client.models.generate_content.side_effect = [
            make_client_error(429),
            make_client_error(429),
            make_client_error(429),
        ]
        with pytest.raises(GeminiRateLimitError) as exc_info:
            service.generate("hello")
        assert exc_info.value.retry_count == 2
        assert fake_client.models.generate_content.call_count == 3

    def test_non_rate_limit_client_error_is_not_retried(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.side_effect = make_client_error(
            400, "bad request"
        )
        with pytest.raises(GeminiResponseError):
            service.generate("hello")
        assert fake_client.models.generate_content.call_count == 1

    def test_server_error_is_not_retried(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.side_effect = make_server_error()
        with pytest.raises(GeminiResponseError):
            service.generate("hello")
        assert fake_client.models.generate_content.call_count == 1

    def test_safety_error_is_not_retried(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = (
            make_safety_blocked_candidate_response()
        )
        with pytest.raises(GeminiSafetyError):
            service.generate("hello")
        assert fake_client.models.generate_content.call_count == 1

    def test_sleep_called_between_retries(
        self, monkeypatch: pytest.MonkeyPatch, service: GeminiService, fake_client: MagicMock
    ) -> None:
        sleep_calls: list[float] = []
        monkeypatch.setattr(
            gemini_service_module.time, "sleep", lambda s: sleep_calls.append(s)
        )
        fake_client.models.generate_content.side_effect = [
            make_client_error(429),
            make_sdk_response(),
        ]
        service.generate("hello")
        assert len(sleep_calls) == 1
        assert sleep_calls[0] >= 0


# ======================================================================
# TestTimeout
# ======================================================================


class TestTimeout:
    def test_slow_call_raises_timeout_error(self, fake_client: MagicMock) -> None:
        import time as real_time

        def slow_generate(*args, **kwargs):
            real_time.sleep(0.3)
            return make_sdk_response()

        fake_client.models.generate_content.side_effect = slow_generate
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            timeout_seconds=0.05,
            max_retries=0,
        )
        with pytest.raises(GeminiTimeoutError) as exc_info:
            svc.generate("hello")
        assert exc_info.value.timeout_seconds == 0.05

    def test_fast_call_within_timeout_succeeds(self, fake_client: MagicMock) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response()
        svc = GeminiService(
            model="gemini-test-model", client=fake_client, timeout_seconds=5
        )
        result = svc.generate("hello")
        assert result.text == "OK"


# ======================================================================
# TestSafety
# ======================================================================


class TestSafety:
    def test_prompt_blocked_before_generation(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_blocked_prompt_response(
            "SAFETY"
        )
        with pytest.raises(GeminiSafetyError) as exc_info:
            service.generate("hello")
        assert "SAFETY" in exc_info.value.blocked_categories[0]

    def test_candidate_blocked_by_safety_filters(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = (
            make_safety_blocked_candidate_response()
        )
        with pytest.raises(GeminiSafetyError) as exc_info:
            service.generate("hello")
        assert exc_info.value.blocked_categories == ["HARM_CATEGORY_HATE_SPEECH"]

    def test_safety_error_carries_request_id(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = (
            make_safety_blocked_candidate_response()
        )
        with pytest.raises(GeminiSafetyError) as exc_info:
            service.generate("hello")
        assert exc_info.value.request_id is not None


# ======================================================================
# TestResponseErrors
# ======================================================================


class TestResponseErrors:
    def test_no_candidates_raises_response_error(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_no_candidates_response()
        with pytest.raises(GeminiResponseError):
            service.generate("hello")

    def test_no_extractable_text_raises_response_error(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_no_text_response()
        with pytest.raises(GeminiResponseError):
            service.generate("hello")

    def test_client_error_wrapped_as_response_error(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.side_effect = make_client_error(
            400, "invalid argument"
        )
        with pytest.raises(GeminiResponseError) as exc_info:
            service.generate("hello")
        assert "invalid argument" in str(exc_info.value)

    def test_server_error_wrapped_as_response_error(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.side_effect = make_server_error(
            500, "oops"
        )
        with pytest.raises(GeminiResponseError) as exc_info:
            service.generate("hello")
        assert "oops" in str(exc_info.value)


# ======================================================================
# TestJsonParsing
# ======================================================================


class TestJsonParsing:
    def test_parses_plain_json_object(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(
            text='{"answer": 42}'
        )
        result = service.generate_json("give me json")
        assert result == {"answer": 42}

    def test_parses_plain_json_array(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(
            text="[1, 2, 3]"
        )
        result = service.generate_json("give me json")
        assert result == [1, 2, 3]

    def test_strips_markdown_code_fence_with_json_tag(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fenced = '```json\n{"answer": 42}\n```'
        fake_client.models.generate_content.return_value = make_sdk_response(text=fenced)
        result = service.generate_json("give me json")
        assert result == {"answer": 42}

    def test_strips_bare_code_fence(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fenced = '```\n{"answer": 42}\n```'
        fake_client.models.generate_content.return_value = make_sdk_response(text=fenced)
        result = service.generate_json("give me json")
        assert result == {"answer": 42}

    def test_invalid_json_raises_parse_error(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(
            text="this is not json"
        )
        with pytest.raises(GeminiJsonParseError) as exc_info:
            service.generate_json("give me json")
        assert exc_info.value.raw_text == "this is not json"
        assert isinstance(exc_info.value.parse_error, json.JSONDecodeError)

    def test_valid_json_scalar_top_level_raises_parse_error(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(text="42")
        with pytest.raises(GeminiJsonParseError) as exc_info:
            service.generate_json("give me json")
        assert exc_info.value.parse_error is None

    def test_json_errors_propagate_request_id(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(
            text="not json"
        )
        with pytest.raises(GeminiJsonParseError) as exc_info:
            service.generate_json("give me json")
        assert exc_info.value.request_id is not None

    def test_upstream_errors_still_propagate_from_generate_json(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = (
            make_safety_blocked_candidate_response()
        )
        with pytest.raises(GeminiSafetyError):
            service.generate_json("give me json")


# ======================================================================
# TestGenerateText
# ======================================================================


class TestGenerateText:
    def test_returns_text_only(self, service: GeminiService, fake_client: MagicMock) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(
            text="just the text"
        )
        result = service.generate_text("hello")
        assert result == "just the text"
        assert isinstance(result, str)

    def test_propagates_errors_like_generate(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_no_candidates_response()
        with pytest.raises(GeminiResponseError):
            service.generate_text("hello")


# ======================================================================
# TestHealthCheck
# ======================================================================


class TestHealthCheck:
    def test_returns_true_on_success(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.return_value = make_sdk_response(text="OK")
        assert service.health_check() is True

    def test_returns_false_on_service_error(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.side_effect = make_server_error()
        assert service.health_check() is False

    def test_returns_false_and_logs_on_safety_block(
        self, fake_client: MagicMock, fake_logger: MagicMock
    ) -> None:
        svc = GeminiService(
            model="gemini-test-model", client=fake_client, logger=fake_logger
        )
        fake_client.models.generate_content.return_value = (
            make_safety_blocked_candidate_response()
        )
        assert svc.health_check() is False
        fake_logger.warning.assert_called()

    def test_does_not_raise_even_on_rate_limit_exhaustion(
        self, service: GeminiService, fake_client: MagicMock
    ) -> None:
        fake_client.models.generate_content.side_effect = make_client_error(429)
        assert service.health_check() is False


# ======================================================================
# TestBackoff
# ======================================================================


class TestBackoff:
    def test_backoff_is_nonnegative_and_within_cap(self, fake_client: MagicMock) -> None:
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            max_backoff_seconds=5.0,
        )
        for retry_count in range(6):
            delay = svc._backoff_seconds(retry_count)
            assert 0 <= delay <= 5.0

    def test_backoff_grows_before_hitting_cap(
        self, monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock
    ) -> None:
        # Pin jitter to its upper bound so the ceiling itself is checked.
        monkeypatch.setattr(gemini_service_module.random, "uniform", lambda lo, hi: hi)
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            max_backoff_seconds=100.0,
        )
        assert svc._backoff_seconds(0) == 1.0
        assert svc._backoff_seconds(1) == 2.0
        assert svc._backoff_seconds(2) == 4.0

    def test_backoff_respects_cap_at_high_retry_counts(
        self, monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock
    ) -> None:
        monkeypatch.setattr(gemini_service_module.random, "uniform", lambda lo, hi: hi)
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            max_backoff_seconds=3.0,
        )
        assert svc._backoff_seconds(10) == 3.0


# ======================================================================
# TestDependencyInjection
# ======================================================================


class TestDependencyInjection:
    def test_injected_client_is_used_instead_of_building_one(
        self, monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock
    ) -> None:
        def fail_if_called(*args, **kwargs):
            raise AssertionError("genai.Client() should not be called when injected")

        monkeypatch.setattr(gemini_service_module.genai, "Client", fail_if_called)
        svc = GeminiService(model="gemini-test-model", client=fake_client)
        assert svc._client is fake_client

    def test_injected_logger_receives_success_logs(
        self, fake_client: MagicMock, fake_logger: MagicMock
    ) -> None:
        svc = GeminiService(
            model="gemini-test-model", client=fake_client, logger=fake_logger
        )
        fake_client.models.generate_content.return_value = make_sdk_response()
        svc.generate("hello")
        fake_logger.info.assert_called()

    def test_injected_logger_receives_retry_warnings(
        self, fake_client: MagicMock, fake_logger: MagicMock
    ) -> None:
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            logger=fake_logger,
            max_retries=1,
        )
        fake_client.models.generate_content.side_effect = [
            make_client_error(429),
            make_sdk_response(),
        ]
        svc.generate("hello")
        fake_logger.warning.assert_called()

    def test_default_logger_used_when_none_injected(self, fake_client: MagicMock) -> None:
        svc = GeminiService(model="gemini-test-model", client=fake_client)
        assert svc._logger is gemini_service_module._module_logger


# ======================================================================
# TestStripCodeFence
# ======================================================================


class TestStripCodeFence:
    """Direct, instance-independent tests of the static fence-stripping helper."""

    def test_strips_json_tagged_fence(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert GeminiService._strip_code_fence(text) == '{"a": 1}'

    def test_strips_bare_fence(self) -> None:
        text = '```\n{"a": 1}\n```'
        assert GeminiService._strip_code_fence(text) == '{"a": 1}'

    def test_strips_uppercase_json_tag(self) -> None:
        text = '```JSON\n{"a": 1}\n```'
        assert GeminiService._strip_code_fence(text) == '{"a": 1}'

    def test_leaves_unfenced_text_unchanged(self) -> None:
        text = '{"a": 1}'
        assert GeminiService._strip_code_fence(text) == '{"a": 1}'

    def test_leaves_unterminated_fence_unchanged(self) -> None:
        # Opening fence with no closing fence doesn't match the
        # "wraps the whole response" pattern, so it's left alone
        # (and will predictably fail JSON parsing downstream).
        text = '```json\n{"a": 1}'
        assert GeminiService._strip_code_fence(text) == text

    def test_strips_fence_with_surrounding_whitespace(self) -> None:
        text = '  \n```json\n{"a": 1}\n```  \n'
        assert GeminiService._strip_code_fence(text) == '{"a": 1}'

    def test_leaves_fence_embedded_mid_text_unchanged(self) -> None:
        # The pattern anchors on the whole string, not a substring —
        # a fence that doesn't wrap the *entire* response is left as-is.
        text = 'Here is the answer:\n```json\n{"a": 1}\n```\nHope that helps!'
        assert GeminiService._strip_code_fence(text) == text


# ======================================================================
# TestPromptResponseLogging
# ======================================================================


class TestPromptResponseLogging:
    def test_log_prompts_false_never_logs_prompt_or_response_text(
        self, fake_client: MagicMock, fake_logger: MagicMock
    ) -> None:
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            logger=fake_logger,
            log_prompts=False,
        )
        fake_client.models.generate_content.return_value = make_sdk_response(
            text="sensitive candidate answer"
        )
        svc.generate("sensitive prompt content")

        debug_calls = [c for c in fake_logger.debug.call_args_list]
        assert debug_calls == []

    def test_log_prompts_true_logs_prompt_before_the_call(
        self, fake_client: MagicMock, fake_logger: MagicMock
    ) -> None:
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            logger=fake_logger,
            log_prompts=True,
        )
        fake_client.models.generate_content.return_value = make_sdk_response(text="ok")
        svc.generate("the exact prompt text")

        prompt_log_calls = [
            c
            for c in fake_logger.debug.call_args_list
            if c.args and c.args[0] == "gemini.generate.prompt"
        ]
        assert len(prompt_log_calls) == 1
        assert prompt_log_calls[0].kwargs["extra"]["prompt"] == "the exact prompt text"

    def test_log_prompts_true_logs_response_text(
        self, fake_client: MagicMock, fake_logger: MagicMock
    ) -> None:
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            logger=fake_logger,
            log_prompts=True,
        )
        fake_client.models.generate_content.return_value = make_sdk_response(
            text="the response text"
        )
        svc.generate("hello")

        response_log_calls = [
            c
            for c in fake_logger.debug.call_args_list
            if c.args and c.args[0] == "gemini.generate.response_text"
        ]
        assert len(response_log_calls) == 1
        assert response_log_calls[0].kwargs["extra"]["text"] == "the response text"

    def test_log_prompts_true_still_only_logs_on_success(
        self, fake_client: MagicMock, fake_logger: MagicMock
    ) -> None:
        # A response-parse failure happens before GeminiResponse is
        # built, so the response_text debug log should not fire.
        svc = GeminiService(
            model="gemini-test-model",
            client=fake_client,
            logger=fake_logger,
            log_prompts=True,
        )
        fake_client.models.generate_content.return_value = make_no_text_response()
        with pytest.raises(GeminiResponseError):
            svc.generate("hello")

        response_log_calls = [
            c
            for c in fake_logger.debug.call_args_list
            if c.args and c.args[0] == "gemini.generate.response_text"
        ]
        assert response_log_calls == []


# ======================================================================
# TestZeroRetryConfig
# ======================================================================


class TestZeroRetryConfig:
    def test_max_retries_zero_fails_immediately_on_rate_limit(
        self, fake_client: MagicMock
    ) -> None:
        svc = GeminiService(
            model="gemini-test-model", client=fake_client, max_retries=0
        )
        fake_client.models.generate_content.side_effect = make_client_error(429)
        with pytest.raises(GeminiRateLimitError) as exc_info:
            svc.generate("hello")
        assert exc_info.value.retry_count == 0
        assert fake_client.models.generate_content.call_count == 1

    def test_max_retries_zero_never_sleeps(
        self, monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock
    ) -> None:
        sleep_calls: list[float] = []
        monkeypatch.setattr(
            gemini_service_module.time, "sleep", lambda s: sleep_calls.append(s)
        )
        svc = GeminiService(
            model="gemini-test-model", client=fake_client, max_retries=0
        )
        fake_client.models.generate_content.side_effect = make_client_error(429)
        with pytest.raises(GeminiRateLimitError):
            svc.generate("hello")
        assert sleep_calls == []

    def test_max_retries_zero_still_succeeds_on_first_try(
        self, fake_client: MagicMock
    ) -> None:
        svc = GeminiService(
            model="gemini-test-model", client=fake_client, max_retries=0
        )
        fake_client.models.generate_content.return_value = make_sdk_response(text="ok")
        result = svc.generate("hello")
        assert result.text == "ok"
        assert result.retry_count == 0


# ======================================================================
# TestGeminiResponseModel
# ======================================================================


class TestGeminiResponseModel:
    def test_total_tokens_computed_when_both_present(self) -> None:
        resp = GeminiResponse(
            text="hi",
            model="gemini-test-model",
            prompt_tokens=10,
            response_tokens=5,
        )
        assert resp.total_tokens == 15

    def test_total_tokens_left_none_when_prompt_tokens_missing(self) -> None:
        resp = GeminiResponse(
            text="hi",
            model="gemini-test-model",
            prompt_tokens=None,
            response_tokens=5,
        )
        assert resp.total_tokens is None

    def test_total_tokens_left_none_when_response_tokens_missing(self) -> None:
        resp = GeminiResponse(
            text="hi",
            model="gemini-test-model",
            prompt_tokens=10,
            response_tokens=None,
        )
        assert resp.total_tokens is None

    def test_explicit_total_tokens_not_overwritten(self) -> None:
        # If a caller ever constructs one with total_tokens already
        # set, __post_init__ should not clobber it.
        resp = GeminiResponse(
            text="hi",
            model="gemini-test-model",
            prompt_tokens=10,
            response_tokens=5,
            total_tokens=999,
        )
        assert resp.total_tokens == 999

    def test_is_frozen(self) -> None:
        resp = GeminiResponse(text="hi", model="gemini-test-model")
        with pytest.raises(Exception):
            resp.text = "changed"  # type: ignore[misc]

    def test_repr_truncates_long_text(self) -> None:
        resp = GeminiResponse(text="x" * 200, model="gemini-test-model")
        rendered = repr(resp)
        assert "…" in rendered
        assert "x" * 200 not in rendered

    def test_repr_does_not_truncate_short_text(self) -> None:
        resp = GeminiResponse(text="short", model="gemini-test-model")
        assert "short" in repr(resp)