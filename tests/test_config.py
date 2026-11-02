from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings

ENV_EXAMPLE = Path(__file__).parent.parent / ".env.example"
ENV_VARS = [
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "TEMPERATURE",
    "MAX_RETRIES",
    "BATCH_CONCURRENCY",
    "PROMPT_PATH",
    "REQUEST_TIMEOUT_S",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_missing_gemini_api_key_fails_naming_it() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_env_example_with_only_key_set_uses_defaults(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        ENV_EXAMPLE.read_text().replace("GEMINI_API_KEY=", "GEMINI_API_KEY=test-key")
    )

    settings = Settings(_env_file=env_file)

    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.temperature == 0.0
    assert settings.max_retries == 3
    assert settings.batch_concurrency == 5
    assert settings.prompt_path == "prompts/classify.md"
    assert settings.request_timeout_s == 30.0


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("MAX_RETRIES", "0"),
        ("BATCH_CONCURRENCY", "0"),
        ("TEMPERATURE", "9"),
        ("TEMPERATURE", "-0.1"),
        ("REQUEST_TIMEOUT_S", "0"),
    ],
)
def test_out_of_range_setting_fails_naming_it(
    monkeypatch: pytest.MonkeyPatch, env_var: str, value: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv(env_var, value)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert env_var.lower() in str(exc_info.value)
