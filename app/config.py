from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: blank lines copied from .env.example keep the defaults.
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, validate_by_name=True
    )

    # Alias so the "missing field" error names the env variable to set.
    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-3.5-flash"
    temperature: float = 0.0
    max_retries: int = 3
    batch_concurrency: int = 5
    prompt_path: str = "prompts/classify.md"


def load_prompt(path: str) -> str:
    # Fail fast at startup instead of sending an empty system prompt to the model.
    prompt_file = Path(path)
    if not prompt_file.is_file():
        raise RuntimeError(f"Prompt file not found: {path}")
    prompt = prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise RuntimeError(f"Prompt file is empty: {path}")
    return prompt
