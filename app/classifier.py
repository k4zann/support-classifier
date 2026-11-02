from typing import Any

from google.genai import types

from app.config import Settings
from app.schemas import ClassifyResponse, ModelOutput


class Classifier:
    def __init__(self, client: Any, prompt: str, settings: Settings) -> None:
        # `client` is a google.genai.Client; tests pass a fake with the same
        # `aio.models.generate_content` shape.
        self.client = client
        self.prompt = prompt
        self.settings = settings

    async def classify(self, text: str) -> ClassifyResponse:
        response = await self.client.aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=self.prompt,
                temperature=self.settings.temperature,
                response_mime_type="application/json",
                response_schema=ModelOutput,
            ),
        )
        output = response.parsed
        if not isinstance(output, ModelOutput):
            output = ModelOutput.model_validate_json(response.text or "")
        return ClassifyResponse.model_validate(output.model_dump())
