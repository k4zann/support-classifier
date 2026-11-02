import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors

from app.classifier import Classifier, ClassifierError
from app.config import Settings
from app.schemas import Category, ClassifyResponse, ModelOutput, Urgency

SMS_TEXT = "Не приходит код из SMS, третий день не могу войти в приложение"
PROMPT = "You classify support requests."


class FakeModels:
    """Mimics `client.aio.models` of google-genai and records each call.

    Each call takes the next outcome (the last one repeats); an exception
    outcome is raised instead of returned.
    """

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def fake_client(models: FakeModels) -> Any:
    return SimpleNamespace(aio=SimpleNamespace(models=models))


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="test-model",
        temperature=0.3,
        max_retries=3,
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Retry tests must not actually wait between attempts.
    monkeypatch.setattr("app.classifier.backoff_delay", lambda attempt: 0.0)


def api_error(cls: type[errors.APIError], code: int, status: str) -> errors.APIError:
    # Same constructor the SDK uses in APIError.raise_error().
    body = {"error": {"code": code, "message": "provider says no", "status": status}}
    return cls(code, body)


VALID_RESPONSE = SimpleNamespace(
    parsed=None,
    text='{"category": "auth", "urgency": "high", "summary": "Не приходит SMS-код."}',
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


@pytest.mark.parametrize(
    "transient_error",
    [
        api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED"),
        api_error(errors.ServerError, 503, "UNAVAILABLE"),
        httpx.ReadTimeout("timed out"),
    ],
    ids=["429", "503", "timeout"],
)
def test_classify_retries_transient_error_then_returns_result(
    transient_error: Exception,
) -> None:
    models = FakeModels(transient_error, VALID_RESPONSE)
    classifier = Classifier(fake_client(models), PROMPT, make_settings())

    result = asyncio.run(classifier.classify(SMS_TEXT))

    assert result.category == Category.auth
    assert len(models.calls) == 2


def test_classify_gives_up_after_max_retries_on_server_errors() -> None:
    models = FakeModels(api_error(errors.ServerError, 503, "UNAVAILABLE"))
    classifier = Classifier(fake_client(models), PROMPT, make_settings())

    with pytest.raises(ClassifierError) as exc_info:
        asyncio.run(classifier.classify(SMS_TEXT))

    assert len(models.calls) == 3
    # The message is safe to show to clients: no provider text in it.
    assert "provider says no" not in str(exc_info.value)
    assert "UNAVAILABLE" not in str(exc_info.value)


@pytest.mark.parametrize(
    "text",
    [
        '{"category": "weather", "urgency": "high", "summary": "Не приходит код."}',
        '{"category": "auth", "summary": "Не приходит код."}',
    ],
    ids=["invalid-enum", "missing-field"],
)
def test_classify_does_not_retry_invalid_model_output(text: str) -> None:
    models = FakeModels(SimpleNamespace(parsed=None, text=text))
    classifier = Classifier(fake_client(models), PROMPT, make_settings())

    with pytest.raises(ClassifierError):
        asyncio.run(classifier.classify(SMS_TEXT))

    assert len(models.calls) == 1


def test_classify_does_not_retry_non_429_client_error() -> None:
    models = FakeModels(
        api_error(errors.ClientError, 400, "INVALID_ARGUMENT"), VALID_RESPONSE
    )
    classifier = Classifier(fake_client(models), PROMPT, make_settings())

    with pytest.raises(ClassifierError) as exc_info:
        asyncio.run(classifier.classify(SMS_TEXT))

    assert len(models.calls) == 1
    assert "provider says no" not in str(exc_info.value)


def summary_response(summary: str) -> Any:
    output = ModelOutput(category=Category.auth, urgency=Urgency.high, summary=summary)
    return SimpleNamespace(parsed=output, text=None)


def classify_summary(summary: str) -> str:
    models = FakeModels(summary_response(summary))
    classifier = Classifier(fake_client(models), PROMPT, make_settings())
    return asyncio.run(classifier.classify(SMS_TEXT)).summary


def trim_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "summary trimmed" in r.getMessage()
    ]


def test_classify_trims_overlong_summary_at_word_boundary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Cyrillic is 2 bytes per character in UTF-8, so a byte-based cut would land
    # near 75 characters; the boundary here is not at exactly 150.
    summary = ("Клиент третий день не получает код из SMS и не может войти " * 4)[:200]
    assert len(summary) == 200

    with caplog.at_level(logging.WARNING):
        result = classify_summary(summary)

    assert 140 < len(result) <= 150
    assert summary.startswith(result)
    assert summary[len(result)].isspace()  # the last word is whole
    assert result == result.rstrip()
    [warning] = trim_warnings(caplog)
    assert "200" in warning.getMessage()


def test_classify_hard_cuts_overlong_summary_without_spaces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = classify_summary("я" * 200)

    assert result == "я" * 150
    assert len(trim_warnings(caplog)) == 1


def test_classify_keeps_summary_at_limit_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    summary = ("Не приходит код " * 10)[:149] + "."
    assert len(summary) == 150

    with caplog.at_level(logging.WARNING):
        result = classify_summary(summary)

    assert result == summary
    assert trim_warnings(caplog) == []
