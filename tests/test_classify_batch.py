import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.classifier import Classifier, ClassifierError
from app.config import Settings
from app.main import app, get_batch_semaphore, get_classifier, lifespan
from app.schemas import Category, ClassifyResponse, Urgency
from tests.fakes import FakeModels, fake_client

REPO_PROMPT = Path(__file__).parent.parent / "prompts" / "classify.md"
PROVIDER_TEXT = "503 UNAVAILABLE: generativelanguage.googleapis.com key=AIza-secret"


class FakeClassifier:
    """Echoes the text as the summary; raises for texts in `fail_on`.

    Tracks how many calls are in flight, to check the concurrency limit.
    """

    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        self.fail_on = fail_on
        self.texts: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def classify(self, text: str) -> ClassifyResponse:
        self.texts.append(text)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            if text in self.fail_on:
                raise ClassifierError(PROVIDER_TEXT)
            return ClassifyResponse(
                category=Category.other, urgency=Urgency.low, summary=text
            )
        finally:
            self.in_flight -= 1


def use(fake: FakeClassifier) -> None:
    app.dependency_overrides[get_classifier] = lambda: fake
    # No lifespan in these tests, so the limit is supplied here.
    app.dependency_overrides[get_batch_semaphore] = lambda: asyncio.Semaphore(5)


@pytest.fixture(autouse=True)
def clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def item(text: str) -> dict[str, str]:
    return {"category": "other", "urgency": "low", "summary": text}


def test_batch_returns_results_in_input_order() -> None:
    fake = FakeClassifier()
    use(fake)
    client = TestClient(app)

    response = client.post("/classify/batch", json={"texts": ["один", "два", "три"]})

    assert response.status_code == 200
    assert response.json() == {
        "results": [item("один"), item("два"), item("три")]
    }


def test_batch_failed_item_holds_error_others_succeed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeClassifier(fail_on=("два",))
    use(fake)
    client = TestClient(app)

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/classify/batch", json={"texts": ["один", "два", "три"]}
        )

    assert response.status_code == 200
    assert response.json() == {
        "results": [item("один"), {"error": "classification failed"}, item("три")]
    }
    assert PROVIDER_TEXT not in response.text
    assert PROVIDER_TEXT in caplog.text


@pytest.mark.parametrize(
    "texts",
    [[], ["текст"] * 51, ["ok text", ""], ["ok text", "   \n "]],
    ids=["empty-list", "51-items", "empty-item", "whitespace-item"],
)
def test_batch_rejects_invalid_request_without_calls(texts: list[str]) -> None:
    fake = FakeClassifier()
    use(fake)
    client = TestClient(app)

    response = client.post("/classify/batch", json={"texts": texts})

    assert response.status_code == 422
    assert fake.texts == []


def test_batch_limits_concurrent_calls() -> None:
    fake = FakeClassifier()
    use(fake)
    client = TestClient(app)
    texts = [f"текст {i}" for i in range(50)]

    response = client.post("/classify/batch", json={"texts": texts})

    assert response.status_code == 200
    assert [r["summary"] for r in response.json()["results"]] == texts
    assert len(fake.texts) == 50
    assert 1 < fake.max_in_flight <= 5


def test_concurrency_limit_is_shared_by_parallel_batch_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The real lifespan (it owns the limit) with the fake classifier.
    monkeypatch.chdir(tmp_path)  # keep the real repo .env out
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PROMPT_PATH", str(REPO_PROMPT))
    monkeypatch.setenv("BATCH_CONCURRENCY", "3")
    fake = FakeClassifier()
    app.dependency_overrides[get_classifier] = lambda: fake

    async def two_batches_at_once() -> list[httpx.Response]:
        async with lifespan(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await asyncio.gather(
                    *(
                        client.post(
                            "/classify/batch",
                            json={"texts": [f"{n}-{i}" for i in range(10)]},
                        )
                        for n in ("a", "b")
                    )
                )

    responses = asyncio.run(two_batches_at_once())

    assert [r.status_code for r in responses] == [200, 200]
    assert len(fake.texts) == 20
    assert 1 < fake.max_in_flight <= 3


@pytest.mark.usefixtures("no_backoff")
def test_batch_connection_errors_become_per_item_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The real Classifier around a fake Gemini client that cannot connect.
    settings = Settings(_env_file=None, GEMINI_API_KEY="test", max_retries=3)
    models = FakeModels(httpx.ConnectError("provider says no"))
    classifier = Classifier(fake_client(models), "prompt", settings)
    app.dependency_overrides[get_classifier] = lambda: classifier
    app.dependency_overrides[get_batch_semaphore] = lambda: asyncio.Semaphore(5)
    client = TestClient(app)

    with caplog.at_level(logging.ERROR):
        response = client.post("/classify/batch", json={"texts": ["один", "два"]})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {"error": "classification failed"},
            {"error": "classification failed"},
        ]
    }
    assert "provider says no" not in response.text
    assert "provider says no" in caplog.text
