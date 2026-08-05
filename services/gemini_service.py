"""
services/gemini_service.py

GeminiService — the ONLY component in this project that talks to
Gemini.

Responsibilities (and nothing more)
------------------------------------
* Validate the rendered prompt string it is given.
* Build generation config from settings + per-call overrides.
* Send that string to Gemini.
* Retry on rate limits / transient failures with exponential
  back-off, jitter, and a capped ceiling.
* Enforce a request timeout.
* Detect and surface safety blocks.
* Parse the response into a structured ``GeminiResponse``.
* Optionally parse the response text as JSON (``generate_json``).
* Report basic liveness (``health_check``).

Explicitly OUT of scope
------------------------
* Building or rendering prompts (that's ``PromptService``).
* Evaluating answers, selecting questions, or anything candidate-aware.
* Any database, BigQuery, or repository access.

Callers should catch ``GeminiServiceError`` (or a specific subclass
from ``exceptions.gemini_exceptions``) — this module never lets a raw
SDK exception escape.

Public API
----------
``generate_text()``, ``generate_json()``, and ``health_check()`` are
the three methods described in the original design. ``generate()`` is
ALSO public and documented as such: it's the one method that returns
the full ``GeminiResponse`` (latency, token counts, request_id,
finish_reason, ...), which any caller that needs more than bare text
or parsed JSON should use directly rather than reaching into private
internals. ``generate_text()`` and ``generate_json()`` are both thin
wrappers over it.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from config.settings import settings
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
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from models.gemini_response import GeminiResponse

_module_logger = logging.getLogger(__name__)

# Finish reasons that mean "safety filter blocked this candidate".
_SAFETY_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"})

# Matches a fenced code block, optionally tagged ```json, wrapping the
# whole response. Some models add this even when told to return raw JSON.
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE
)

# Minimal prompt used by health_check() — cheap, deterministic, and
# unlikely to trip safety filters or need more than a few tokens.
_HEALTH_CHECK_PROMPT = "Reply with exactly one word: OK"


class GeminiService:
    """
    Thin, resilient wrapper around the Gemini SDK.

    One instance is safe to reuse across requests — the underlying
    SDK client is created once in ``__init__`` (or injected) and is
    thread-safe for read-only ``generate_content`` calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int | float = 30,
        max_retries: int = 3,
        max_backoff_seconds: float = 20.0,
        max_prompt_chars: int = 32_000,
        default_temperature: float | None = None,
        default_max_output_tokens: int | None = None,
        log_prompts: bool = False,
        client: genai.Client | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Args:
            api_key: Overrides ``settings.GOOGLE_API_KEY`` when given.
                Ignored if ``client`` is injected. Falls back to
                settings otherwise.
            model: Overrides ``settings.GEMINI_MODEL`` when given.
            timeout_seconds: Per-attempt wall-clock timeout, enforced
                both at the SDK/transport level (only when this
                service constructs its own client — see ``client``
                below) and as a thread-based backstop regardless.
            max_retries: Max retry attempts for rate-limit / transient
                errors, on top of the initial attempt.
            max_backoff_seconds: Hard ceiling on the exponential
                back-off delay between retries, regardless of retry
                count or jitter.
            max_prompt_chars: Prompts longer than this are rejected by
                ``_validate_prompt`` before any network call is made.
            default_temperature: Default sampling temperature applied
                by ``_build_generation_config`` when a call doesn't
                override it. ``None`` leaves the SDK default in place.
            default_max_output_tokens: Default output token cap applied
                the same way.
            log_prompts: When ``True``, full prompt/response text is
                included in debug logs. Left ``False`` by default so
                candidate/PII-bearing content never lands in logs
                unless explicitly opted into (e.g. in a dev environment).
            client: An already-constructed ``genai.Client`` to use
                instead of building one from ``api_key``. Intended for
                tests (inject a fake/mock client) and for callers that
                need a client configured beyond what this constructor
                exposes (custom ``http_options``, Vertex AI mode,
                etc.). When given, ``api_key`` is ignored and no
                configuration validation is done on it.
            logger: A ``logging.Logger`` to use instead of this
                module's default logger. Useful for tests that want to
                capture/assert on log output, or callers that want
                request-scoped logger instances.

        Raises:
            GeminiConfigurationError: If ``model`` is unavailable from
                either the argument or ``settings``, or if no
                ``client`` was injected and no API key is available
                from either the argument or ``settings``.
        """
        resolved_model = model or settings.GEMINI_MODEL
        if not resolved_model:
            raise GeminiConfigurationError(
                "GEMINI_MODEL is not set.",
                config_key="GEMINI_MODEL",
            )

        self._model: str = resolved_model
        self._timeout_seconds: int | float = timeout_seconds
        self._max_retries: int = max_retries
        self._max_backoff_seconds: float = max_backoff_seconds
        self._max_prompt_chars: int = max_prompt_chars
        self._default_temperature: float | None = default_temperature
        self._default_max_output_tokens: int | None = default_max_output_tokens
        self._log_prompts: bool = log_prompts
        self._logger: logging.Logger = logger or _module_logger

        if client is not None:
            self._client = client
        else:
            resolved_key = api_key or settings.GOOGLE_API_KEY
            if not resolved_key:
                raise GeminiConfigurationError(
                    "GOOGLE_API_KEY is not set. Configure it in the "
                    "environment (.env), pass api_key explicitly, or "
                    "inject a pre-built client.",
                    config_key="GOOGLE_API_KEY",
                )
            try:
                # NOTE on timeout: the SDK's own http_options.timeout is
                # set here as the first line of defense, but it is known
                # to be unreliable in several released SDK versions
                # (e.g. googleapis/python-genai issues #911, #1330, #4031
                # — timeout silently not honored by the sync transport in
                # some cases). _call_with_timeout() below additionally
                # wraps every call in a bounded thread as a backstop, so
                # a stuck request can never hang this service
                # indefinitely even if the SDK-level timeout fails to
                # fire.
                self._client = genai.Client(
                    api_key=resolved_key,
                    http_options=genai_types.HttpOptions(
                        timeout=int(self._timeout_seconds * 1000)
                    ),
                )
            except Exception as exc:  # SDK init failures are always config problems
                raise GeminiConfigurationError(
                    f"Failed to initialise Gemini client: {exc}",
                    config_key="GOOGLE_API_KEY",
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str, **generation_kwargs: Any) -> str:
        """
        Convenience wrapper returning just the response text.

        Args:
            prompt: Fully rendered prompt string.
            **generation_kwargs: Passed through to ``generate()``.

        Returns:
            The response text.
        """
        return self.generate(prompt, **generation_kwargs).text

    def generate(self, prompt: str, **generation_kwargs: Any) -> GeminiResponse:
        """
        Send ``prompt`` to Gemini and return a structured response.

        Public API: use this directly (rather than ``generate_text()``
        or ``generate_json()``) whenever you need latency, token
        counts, ``finish_reason``, or ``request_id`` alongside the
        text.

        Args:
            prompt: Fully rendered prompt string. This service does
                not template or modify it in any way.
            **generation_kwargs: Extra keyword args forwarded to
                ``_build_generation_config`` to override per-call
                defaults (e.g. ``temperature``, ``max_output_tokens``).

        Returns:
            A populated ``GeminiResponse``.

        Raises:
            GeminiPromptError: The prompt failed pre-flight validation.
            GeminiTimeoutError: The request exceeded the configured
                timeout on every attempt.
            GeminiRateLimitError: The retry budget was exhausted while
                hitting rate limits.
            GeminiSafetyError: The response was blocked by safety
                filters.
            GeminiResponseError: The response could not be parsed
                into usable text.
        """
        self._validate_prompt(prompt)
        config = self._build_generation_config(**generation_kwargs)
        request_id = uuid.uuid4().hex[:12]
        start = time.monotonic()

        if self._log_prompts:
            self._logger.debug(
                "gemini.generate.prompt", extra={"request_id": request_id, "prompt": prompt}
            )

        retry_count = 0
        last_error: GeminiServiceError | None = None

        while retry_count <= self._max_retries:
            try:
                sdk_response = self._call_with_timeout(
                    prompt, config, request_id=request_id
                )
                response = self._parse_response(
                    sdk_response,
                    request_id=request_id,
                    retry_count=retry_count,
                    latency_ms=(time.monotonic() - start) * 1000,
                )
                self._logger.info(
                    "gemini.generate.success",
                    extra={
                        "request_id": request_id,
                        "model": response.model,
                        "retry_count": response.retry_count,
                        "latency_ms": response.latency_ms,
                        "finish_reason": response.finish_reason,
                        "prompt_tokens": response.prompt_tokens,
                        "response_tokens": response.response_tokens,
                        "total_tokens": response.total_tokens,
                    },
                )
                return response

            except GeminiRateLimitError as exc:
                last_error = exc
                if retry_count >= self._max_retries:
                    break
                sleep_for = self._backoff_seconds(retry_count)
                self._logger.warning(
                    "gemini.generate.rate_limited_retrying",
                    extra={
                        "request_id": request_id,
                        "retry_count": retry_count,
                        "sleep_for": sleep_for,
                    },
                )
                time.sleep(sleep_for)
                retry_count += 1
                continue

            # Safety, timeout, and response-parse errors are not
            # retryable — a different prompt or a human is needed.
            except (GeminiSafetyError, GeminiTimeoutError, GeminiResponseError):
                raise

        # Retry budget exhausted on rate limiting.
        raise GeminiRateLimitError(
            f"Gemini rate limit exceeded after {retry_count} retries.",
            retry_count=retry_count,
            request_id=request_id,
        ) from last_error

    def generate_json(self, prompt: str, **generation_kwargs: Any) -> Any:
        """
        Send ``prompt`` to Gemini and parse the response text as JSON.

        Args:
            prompt: Fully rendered prompt string. Callers are
                responsible for instructing the model to return JSON
                — this method does not rewrite the prompt.
            **generation_kwargs: Forwarded to ``generate()``.

        Returns:
            The parsed ``dict`` or ``list``. If metadata about the
            call (latency, token counts, request_id, ...) is also
            needed, call ``generate()`` directly and parse its
            ``.text`` yourself — this method intentionally returns
            only the parsed value, not a response wrapper, since a
            response model should represent exactly one shape of
            "the answer."

        Raises:
            GeminiJsonParseError: The response text was not valid
                JSON (after stripping a wrapping Markdown code fence,
                if present), or the top-level value was not a ``dict``
                or ``list``.
            (All exceptions raised by ``generate()`` propagate too.)
        """
        response = self.generate(prompt, **generation_kwargs)
        candidate_text = self._strip_code_fence(response.text)

        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError as exc:
            raise GeminiJsonParseError(
                "Gemini response was not valid JSON.",
                raw_text=response.text,
                parse_error=exc,
                request_id=response.request_id,
            ) from exc

        if not isinstance(parsed, (dict, list)):
            raise GeminiJsonParseError(
                "Gemini response was valid JSON but the top-level "
                f"value was {type(parsed).__name__}, not an object or array.",
                raw_text=response.text,
                parse_error=None,
                request_id=response.request_id,
            )

        return parsed

    def health_check(self) -> bool:
        """
        Lightweight liveness check.

        Sends a minimal, deterministic prompt and reports whether a
        usable response came back — intended for readiness/liveness
        probes, not for validating any particular model behaviour.

        Returns:
            ``True`` if ``generate()`` returned successfully. ``False``
            if it raised any ``GeminiServiceError`` (timeout, rate
            limit, safety block, malformed response, bad config,
            etc.) — a health check should report "not healthy" rather
            than let the exception propagate and crash whatever is
            polling it.
        """
        try:
            self.generate(_HEALTH_CHECK_PROMPT, max_output_tokens=8)
            return True
        except GeminiServiceError as exc:
            self._logger.warning(
                "gemini.health_check.failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_prompt(self, prompt: str) -> None:
        """
        Pre-flight checks that fire before any network call is made.

        Raises:
            GeminiPromptError: If ``prompt`` is not a non-empty string
                within ``max_prompt_chars``.
        """
        if not isinstance(prompt, str):
            raise GeminiPromptError(
                f"prompt must be a str, got {type(prompt).__name__}."
            )
        if not prompt.strip():
            raise GeminiPromptError("prompt must not be empty or whitespace-only.")
        if len(prompt) > self._max_prompt_chars:
            raise GeminiPromptError(
                f"prompt is {len(prompt)} characters, which exceeds the "
                f"configured limit of {self._max_prompt_chars}."
            )

    def _build_generation_config(
        self, **overrides: Any
    ) -> genai_types.GenerateContentConfig:
        """
        Build a ``GenerateContentConfig`` from instance defaults,
        overridden per-call by ``**overrides``.

        Centralising this means generation parameters flow
        Settings -> defaults -> per-call overrides -> SDK config in
        one place, rather than being scattered across call sites.

        Args:
            **overrides: Any ``GenerateContentConfig`` field to
                override for this call (e.g. ``temperature=0.2``,
                ``max_output_tokens=1024``, ``top_p=0.9``).

        Returns:
            A ``GenerateContentConfig`` ready to pass to the SDK.
        """
        config_kwargs: dict[str, Any] = {}
        if self._default_temperature is not None:
            config_kwargs["temperature"] = self._default_temperature
        if self._default_max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self._default_max_output_tokens
        config_kwargs.update(overrides)
        return genai_types.GenerateContentConfig(**config_kwargs)

    def _call_with_timeout(
        self,
        prompt: str,
        config: genai_types.GenerateContentConfig,
        request_id: str,
    ) -> Any:
        """
        Invoke the SDK in a worker thread and enforce ``timeout_seconds``
        as a backstop on top of the SDK's own ``http_options.timeout``.

        Why both: the SDK-level timeout is the cheaper, more "correct"
        mechanism when it works, but several released versions of
        google-genai have shipped with that timeout not reliably
        honored by the synchronous transport (silent hangs reported in
        googleapis/python-genai #911, #1330, #4031). A single-shot
        ``ThreadPoolExecutor`` guarantees this method returns control
        to the caller within ``timeout_seconds`` regardless of what the
        transport does under the hood — at the cost of leaking one
        blocked worker thread until the underlying call eventually
        completes or the process exits. That trade-off is acceptable
        here: a bounded, bookkept leak beats an unbounded hang in a
        service other components depend on synchronously. Worth
        revisiting if/when the SDK's own timeout becomes reliable
        across supported versions.

        Raises:
            GeminiTimeoutError: The call did not complete in time.
            GeminiRateLimitError: The SDK reported HTTP 429 / quota
                exhaustion.
            GeminiResponseError: Any other SDK-level failure.
        """
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                return future.result(timeout=self._timeout_seconds)

        except FutureTimeoutError as exc:
            raise GeminiTimeoutError(
                f"Gemini request exceeded {self._timeout_seconds}s timeout.",
                timeout_seconds=self._timeout_seconds,
                request_id=request_id,
            ) from exc

        except genai_errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise GeminiRateLimitError(
                    "Gemini rate limit / quota exceeded.",
                    request_id=request_id,
                ) from exc
            raise GeminiResponseError(
                f"Gemini client error: {exc}",
                request_id=request_id,
            ) from exc

        except genai_errors.ServerError as exc:
            raise GeminiResponseError(
                f"Gemini server error: {exc}",
                request_id=request_id,
            ) from exc

    def _parse_response(
        self,
        sdk_response: Any,
        request_id: str,
        retry_count: int,
        latency_ms: float,
    ) -> GeminiResponse:
        """
        Convert a raw SDK response object into a ``GeminiResponse``,
        detecting safety blocks and malformed responses along the way.

        Raises:
            GeminiSafetyError: A candidate was blocked by safety
                filters, or ``prompt_feedback.block_reason`` is set.
            GeminiResponseError: ``candidates`` is missing/empty, or
                no usable text could be extracted.
        """
        prompt_feedback = getattr(sdk_response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            raise GeminiSafetyError(
                f"Prompt was blocked before generation: {block_reason}",
                blocked_categories=[str(block_reason)],
                request_id=request_id,
            )

        candidates = getattr(sdk_response, "candidates", None)
        if not candidates:
            raise GeminiResponseError(
                "Gemini response contained no candidates.",
                raw_text=None,
                request_id=request_id,
            )

        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason_str = str(finish_reason) if finish_reason is not None else None

        if finish_reason_str in _SAFETY_FINISH_REASONS:
            categories = [
                str(getattr(rating, "category", rating))
                for rating in (getattr(candidate, "safety_ratings", None) or [])
                if getattr(rating, "blocked", False)
            ]
            raise GeminiSafetyError(
                f"Response blocked by safety filters (finish_reason={finish_reason_str}).",
                blocked_categories=categories,
                request_id=request_id,
            )

        text = getattr(sdk_response, "text", None)
        if text is None:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            text_parts = [p.text for p in parts if getattr(p, "text", None)]
            text = "".join(text_parts) if text_parts else None

        if not text:
            raise GeminiResponseError(
                "Gemini response had no extractable text.",
                raw_text=None,
                request_id=request_id,
            )

        usage = getattr(sdk_response, "usage_metadata", None)

        if self._log_prompts:
            self._logger.debug(
                "gemini.generate.response_text",
                extra={"request_id": request_id, "text": text},
            )

        return GeminiResponse(
            text=text,
            model=self._model,
            finish_reason=finish_reason_str,
            request_id=request_id,
            retry_count=retry_count,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            response_tokens=getattr(usage, "candidates_token_count", None),
            timestamp=time.time(),
            raw_response=sdk_response,
        )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        Strip a wrapping ```json ... ``` (or bare ``` ... ```) fence
        from ``text``, if present. Some models add this even when the
        prompt asks for raw JSON. Returns ``text`` unchanged if it
        doesn't match the fenced pattern.
        """
        match = _CODE_FENCE_RE.match(text)
        return match.group(1) if match else text

    def _backoff_seconds(self, retry_count: int) -> float:
        """
        Exponential back-off with full jitter, capped at
        ``max_backoff_seconds``: for retry N, sleep a random duration
        in ``[0, min(max_backoff_seconds, 2**N)]``.
        """
        ceiling = min(self._max_backoff_seconds, float(2**retry_count))
        return random.uniform(0, ceiling)


# Create one reusable service object, mirroring the module-level
# singleton pattern used elsewhere in this project.
gemini_service = GeminiService()