import asyncio
import logging
import random
from typing import Any

import httpx
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
    # Rate limit, server errors and timeouts may pass on retry; other 4xx won't.
    if isinstance(exc, errors.APIError):
        return exc.code == 429 or exc.code >= 500
    return True


def trim_summary(summary: str) -> str:
    """Cut at the last word boundary within the limit, or hard-cut with no spaces."""
    if len(summary) <= SUMMARY_MAX_LENGTH:
        return summary
    cut = summary[:SUMMARY_MAX_LENGTH]
    if not summary[SUMMARY_MAX_LENGTH].isspace():
        head, space, _ = cut.rpartition(" ")
        if space:
            cut = head
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
            except (errors.APIError, httpx.TimeoutException, TimeoutError) as exc:
                if not is_transient(exc) or attempt == self.settings.max_retries:
                    raise ClassifierError("model call failed") from exc
                await asyncio.sleep(backoff_delay(attempt))

        try:
            output = response.parsed
            if not isinstance(output, ModelOutput):
                output = ModelOutput.model_validate_json(response.text or "")
            fields = output.model_dump()
            fields["summary"] = trim_summary(output.summary)
            return ClassifyResponse.model_validate(fields)
        except ValidationError as exc:
            raise ClassifierError("invalid model output") from exc
