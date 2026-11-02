"""A fake google-genai client, shared by the classifier and endpoint tests."""

from types import SimpleNamespace
from typing import Any

from google.genai import errors


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


def api_error(cls: type[errors.APIError], code: int, status: str) -> errors.APIError:
    # Same constructor the SDK uses in APIError.raise_error().
    body = {"error": {"code": code, "message": "provider says no", "status": status}}
    return cls(code, body)


VALID_RESPONSE = SimpleNamespace(
    parsed=None,
    text='{"category": "auth", "urgency": "high", "summary": "Не приходит SMS-код."}',
)
