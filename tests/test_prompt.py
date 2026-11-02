from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

REPO_PROMPT = Path(__file__).parent.parent / "prompts" / "classify.md"


@pytest.fixture(autouse=True)
def startup_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Keep the real repo .env out of these tests.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_startup_fails_naming_missing_prompt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompt_path = tmp_path / "missing.md"
    monkeypatch.setenv("PROMPT_PATH", str(prompt_path))

    with pytest.raises(RuntimeError) as exc_info, TestClient(app):
        pass

    assert str(prompt_path) in str(exc_info.value)


@pytest.mark.parametrize("content", ["", "  \n\t "])
def test_startup_fails_naming_empty_prompt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    prompt_path = tmp_path / "classify.md"
    prompt_path.write_text(content)
    monkeypatch.setenv("PROMPT_PATH", str(prompt_path))

    with pytest.raises(RuntimeError) as exc_info, TestClient(app):
        pass

    assert str(prompt_path) in str(exc_info.value)


def test_startup_succeeds_with_repo_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPT_PATH", str(REPO_PROMPT))

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
