"""
models/gemini_response.py

Structured, immutable representation of a single successful call to
Gemini. ``GeminiService`` is the only thing that should construct
this object — callers just read it.

Design notes
------------
* Kept as a plain ``@dataclass(frozen=True)`` rather than a pydantic
  model: this is an internal, in-process value object, not something
  that needs (de)serialisation or validation on the read path.
* ``raw_response`` intentionally holds the original SDK response object
  (not just a dict) so callers who need something not surfaced by this
  wrapper can still get at it — but they should treat it as an escape
  hatch, not a stable contract.
* This model represents exactly one thing: the API response. Parsed
  JSON is deliberately NOT a field here — ``generate_json()`` returns
  the parsed ``dict``/``list`` directly rather than smuggling a second
  shape of "the answer" into this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeminiResponse:
    """
    Result of a single successful Gemini request.

    Attributes:
        text: The concatenated text of the response. Never ``None``
            for a successfully constructed ``GeminiResponse`` — if
            Gemini returned no usable text, ``GeminiService`` raises
            ``GeminiResponseError`` instead of building one of these.
        model: The model name that actually served the request
            (e.g. ``"gemini-3.6-flash"``), taken from configuration
            rather than the SDK response, since not all SDK versions
            echo it back reliably.
        finish_reason: The SDK's terminal-state string for the
            candidate that produced ``text`` (e.g. ``"STOP"``,
            ``"MAX_TOKENS"``). ``None`` if the SDK did not expose one.
        request_id: Correlation ID assigned by ``GeminiService`` at
            the start of the call, threaded through logs and any
            exception raised for this request.
        retry_count: Number of retries that were performed before
            this response was obtained. ``0`` means it succeeded on
            the first attempt.
        latency_ms: Wall-clock time spent on the request (including
            retries), in milliseconds.
        prompt_tokens: Input token count reported by the SDK's usage
            metadata, if available.
        response_tokens: Output token count reported by the SDK's
            usage metadata, if available. Named to match Gemini's own
            ``candidates_token_count`` terminology rather than the
            OpenAI-style ``completion_tokens``.
        total_tokens: ``prompt_tokens + response_tokens`` when both are
            known; ``None`` if either is missing. Computed automatically
            in ``__post_init__`` when not supplied explicitly.
        timestamp: Unix timestamp (seconds) of when the response was
            constructed, set by ``GeminiService``. Useful for
            analytics/reporting without needing external log
            correlation.
        raw_response: The original SDK response object, kept as an
            escape hatch for callers that need something this wrapper
            doesn't surface. Not part of the stable contract.
    """

    text: str
    model: str
    finish_reason: str | None = None
    request_id: str | None = None
    retry_count: int = 0
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    timestamp: float | None = None
    raw_response: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.total_tokens is None
            and self.prompt_tokens is not None
            and self.response_tokens is not None
        ):
            # Frozen dataclass: bypass __setattr__ for this one
            # derived-field computation.
            object.__setattr__(
                self, "total_tokens", self.prompt_tokens + self.response_tokens
            )

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.text[:80] + "…" if len(self.text) > 80 else self.text
        return (
            f"GeminiResponse("
            f"text={preview!r}, "
            f"model={self.model!r}, "
            f"finish_reason={self.finish_reason!r}, "
            f"request_id={self.request_id!r}, "
            f"retry_count={self.retry_count!r}, "
            f"latency_ms={self.latency_ms!r}, "
            f"total_tokens={self.total_tokens!r})"
        )