"""The real google-genai client as the service builds it, against local sockets only."""

import asyncio
import socket
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.classifier import Classifier, ClassifierError, build_client
from app.config import Settings
from app.main import app

REPO_PROMPT = Path(__file__).parent.parent / "prompts" / "classify.md"


def test_built_client_times_out_a_hanging_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> BaseException | None:
        async def never_answer(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.read()  # until the client gives up and disconnects

        server = await asyncio.start_server(never_answer, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{port}")
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
            request_timeout_s=0.2,
            max_retries=1,
        )
        client = build_client(settings)
        classifier = Classifier(client, "prompt", settings)
        try:
            with pytest.raises(ClassifierError) as exc_info:
                # Without a client timeout this guard fails the test instead of hanging.
                await asyncio.wait_for(classifier.classify("текст"), 5)
        finally:
            await client.aio.aclose()
            server.close()
        return exc_info.value.__cause__

    assert isinstance(asyncio.run(run()), httpx.TimeoutException)


def test_shutdown_closes_the_gemini_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A port nothing listens on: an open client would get a connection error.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.chdir(tmp_path)  # keep the real repo .env out
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PROMPT_PATH", str(REPO_PROMPT))

    with TestClient(app):
        client = app.state.classifier.client

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(client.aio.models.generate_content(model="m", contents="текст"))
