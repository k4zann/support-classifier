import asyncio
from types import SimpleNamespace
from typing import Any

from app.classifier import Classifier
from app.config import Settings
from app.schemas import Category, ClassifyResponse, ModelOutput, Urgency

SMS_TEXT = "Не приходит код из SMS, третий день не могу войти в приложение"
PROMPT = "You classify support requests."


class FakeModels:
    """Mimics `client.aio.models` of google-genai and records each call."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.response


def fake_client(models: FakeModels) -> Any:
    return SimpleNamespace(aio=SimpleNamespace(models=models))


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="test-model",
        temperature=0.3,
    )


def test_classify_sends_structured_output_request_and_returns_response() -> None:
    output = ModelOutput(
        category=Category.auth,
        urgency=Urgency.high,
        summary="Клиент не получает SMS-код и не может войти.",
    )
    models = FakeModels(SimpleNamespace(parsed=output, text=None))
    classifier = Classifier(fake_client(models), PROMPT, make_settings())

    result = asyncio.run(classifier.classify(SMS_TEXT))

    assert result == ClassifyResponse(
        category=Category.auth,
        urgency=Urgency.high,
        summary="Клиент не получает SMS-код и не может войти.",
    )
    assert len(models.calls) == 1
    call = models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == SMS_TEXT
    config = call["config"]
    assert config.response_schema is ModelOutput
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0.3
    assert config.system_instruction == PROMPT


def test_classify_falls_back_to_response_text_when_parsed_is_missing() -> None:
    text = '{"category": "payment", "urgency": "low", "summary": "Вопрос по оплате."}'
    models = FakeModels(SimpleNamespace(parsed=None, text=text))
    classifier = Classifier(fake_client(models), PROMPT, make_settings())

    result = asyncio.run(classifier.classify("Как оплатить картой?"))

    assert result == ClassifyResponse(
        category=Category.payment, urgency=Urgency.low, summary="Вопрос по оплате."
    )
