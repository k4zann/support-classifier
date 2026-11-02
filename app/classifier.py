import asyncio
import logging
import random
import re
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from app.config import Settings
from app.schemas import ClassifyResponse, ModelOutput

logger = logging.getLogger(__name__)

SUMMARY_MAX_LENGTH = 150

class ClassifierError(Exception):
    """Final classification failure. The message is generic, with no provider text;
    the original error is kept as `__cause__` for logging."""


def backoff_delay(attempt: int) -> float:
    # 1s, 2s, 4s... plus up to 1s of jitter so concurrent retries spread out.
    return 2 ** (attempt - 1) + random.uniform(0, 1)


def is_transient(exc: Exception) -> bool:
    # Rate limit, server errors, timeouts and connection errors may pass on retry;
    # other 4xx won't.
    if isinstance(exc, errors.APIError):
        return exc.code == 429 or exc.code >= 500
    return True


def build_client(settings: Settings) -> genai.Client:
    """The Gemini client, built the same way for the service and the eval runner."""
    # google-genai sets no client timeout by default; HttpOptions.timeout is in ms.
    http_options = types.HttpOptions(timeout=int(settings.request_timeout_s * 1000))
    return genai.Client(api_key=settings.gemini_api_key, http_options=http_options)


def trim_summary(summary: str) -> str:
    """Cut at the last word boundary within the limit, or hard-cut with no whitespace."""
    if len(summary) <= SUMMARY_MAX_LENGTH:
        return summary
    cut = summary[:SUMMARY_MAX_LENGTH]
    if not summary[SUMMARY_MAX_LENGTH].isspace():
        # Any whitespace counts as a boundary: NBSP, tab and newline too.
        boundary = re.search(r"\s\S*$", cut)
        if boundary:
            cut = cut[: boundary.start()]
    cut = cut.rstrip() or summary[:SUMMARY_MAX_LENGTH]
    # The eval harness counts trims by this message.
    logger.warning("summary trimmed from %d to %d characters", len(summary), len(cut))
    return cut


class Classifier:
    def __init__(self, client: Any, prompt: str, settings: Settings) -> None:
        # `client` is a google.genai.Client; tests pass a fake with the same
        # `aio.models.generate_content` shape.
        self.client = client
        self.prompt = prompt
        self.settings = settings
        # Running totals from Gemini's `usage_metadata`; the eval harness stores
        # them so the README can estimate cost. Not part of the HTTP response.
        self.token_usage = {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
        }

    def _add_usage(self, response: Any) -> None:
        # Real responses always have the attribute (possibly None); fakes may not.
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        self.token_usage["prompt_tokens"] += usage.prompt_token_count or 0
        self.token_usage["output_tokens"] += usage.candidates_token_count or 0
        self.token_usage["thinking_tokens"] += usage.thoughts_token_count or 0
        self.token_usage["total_tokens"] += usage.total_token_count or 0

    async def classify(self, text: str) -> ClassifyResponse:
        config = types.GenerateContentConfig(
            system_instruction=self.prompt,
            temperature=self.settings.temperature,
            response_mime_type="application/json",
            response_schema=ModelOutput,
        )
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.settings.gemini_model, contents=text, config=config
                )
                break
            # TransportError covers httpx timeouts and connection errors, which
            # google-genai lets through unwrapped.
            except (errors.APIError, httpx.TransportError, TimeoutError) as exc:
                if not is_transient(exc) or attempt == self.settings.max_retries:
                    raise ClassifierError("model call failed") from exc
                await asyncio.sleep(backoff_delay(attempt))

        # Counted before parsing: an invalid answer still used tokens.
        self._add_usage(response)
        try:
            output = response.parsed
            if not isinstance(output, ModelOutput):
                output = ModelOutput.model_validate_json(response.text or "")
            fields = output.model_dump()
            # Stripped first, so a blank summary fails validation as invalid output.
            fields["summary"] = trim_summary(output.summary.strip())
            return ClassifyResponse.model_validate(fields)
        except ValidationError as exc:
            raise ClassifierError("invalid model output") from exc
