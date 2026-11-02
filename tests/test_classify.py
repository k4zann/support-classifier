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
