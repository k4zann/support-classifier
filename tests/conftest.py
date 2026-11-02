import pytest


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Retry tests must not actually wait between attempts.
    monkeypatch.setattr("app.classifier.backoff_delay", lambda attempt: 0.0)
