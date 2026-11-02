from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_classifier
from app.schemas import Category, ClassifyResponse, Urgency

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
