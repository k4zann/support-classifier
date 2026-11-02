import logging
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors

from app.classifier import Classifier, ClassifierError
from app.config import Settings
from app.main import app, get_classifier
from app.schemas import Category, ClassifyResponse, Urgency
from tests.fakes import VALID_RESPONSE, FakeModels, api_error, fake_client

SMS_TEXT = "Не приходит код из SMS, третий день не могу войти в приложение"


class FakeClassifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def classify(self, text: str) -> ClassifyResponse:
        self.texts.append(text)
        return ClassifyResponse(
            category=Category.auth,
            urgency=Urgency.high,
            summary="Клиент не получает SMS-код и не может войти.",
        )


@pytest.fixture
def fake() -> Iterator[FakeClassifier]:
    fake = FakeClassifier()
    app.dependency_overrides[get_classifier] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_classify_returns_classification(fake: FakeClassifier) -> None:
    # No `with`: the lifespan (real Settings and prompt) is not needed here.
    client = TestClient(app)

    response = client.post("/classify", json={"text": SMS_TEXT})

    assert response.status_code == 200
    assert response.json() == {
        "category": "auth",
        "urgency": "high",
        "summary": "Клиент не получает SMS-код и не может войти.",
    }
    assert fake.texts == [SMS_TEXT]


@pytest.mark.parametrize(
    ("body", "expected_msg"),
    [
        ({"text": ""}, "must not be empty"),
        ({"text": "   \n "}, "must not be empty"),
        ({"text": "a" * 4001}, "4000"),
        ({}, "Field required"),
        ({"text": 123}, "valid string"),
    ],
    ids=["empty", "whitespace", "too-long", "missing", "not-a-string"],
)
def test_classify_rejects_invalid_text(
    fake: FakeClassifier, body: dict[str, object], expected_msg: str
) -> None:
    client = TestClient(app)

    response = client.post("/classify", json=body)

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "text"]
    assert expected_msg in error["msg"]
    assert fake.texts == []


def test_classify_accepts_text_at_the_limit(fake: FakeClassifier) -> None:
    client = TestClient(app)
    text = "a" * 4000

    response = client.post("/classify", json={"text": text})

    assert response.status_code == 200
    assert fake.texts == [text]


PROVIDER_TEXT = "503 UNAVAILABLE: generativelanguage.googleapis.com key=AIza-secret"


class FailingClassifier:
    async def classify(self, text: str) -> ClassifyResponse:
        raise ClassifierError(PROVIDER_TEXT)


def test_classify_maps_classifier_error_to_502(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app.dependency_overrides[get_classifier] = lambda: FailingClassifier()
    client = TestClient(app)

    try:
        with caplog.at_level(logging.ERROR):
            response = client.post("/classify", json={"text": SMS_TEXT})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"detail": "classification failed"}
    assert PROVIDER_TEXT not in response.text
    assert PROVIDER_TEXT in caplog.text


def use_real_classifier(models: FakeModels) -> None:
    # The real Classifier (retries, error chaining) around a fake Gemini client.
    settings = Settings(_env_file=None, GEMINI_API_KEY="test-key", max_retries=3)
    classifier = Classifier(fake_client(models), "prompt", settings)
    app.dependency_overrides[get_classifier] = lambda: classifier


@pytest.mark.usefixtures("no_backoff")
@pytest.mark.parametrize(
    "error",
    [
        api_error(errors.ServerError, 503, "UNAVAILABLE"),
        httpx.ConnectError("provider says no"),
    ],
    ids=["503", "connect-error"],
)
def test_classify_returns_502_when_model_stays_unavailable(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    models = FakeModels(error)
    use_real_classifier(models)
    client = TestClient(app)

    try:
        with caplog.at_level(logging.ERROR):
            response = client.post("/classify", json={"text": SMS_TEXT})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"detail": "classification failed"}
    assert len(models.calls) == 3
    assert "provider says no" not in response.text
    assert "provider says no" in caplog.text


@pytest.mark.usefixtures("no_backoff")
def test_classify_returns_200_after_rate_limit_then_success() -> None:
    models = FakeModels(
        api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED"), VALID_RESPONSE
    )
    use_real_classifier(models)
    client = TestClient(app)

    try:
        response = client.post("/classify", json={"text": SMS_TEXT})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "category": "auth",
        "urgency": "high",
        "summary": "Не приходит SMS-код.",
    }
    assert len(models.calls) == 2
