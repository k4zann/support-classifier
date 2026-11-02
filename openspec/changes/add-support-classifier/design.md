## Context

This is a greenfield repo. So far it has only `pyproject.toml` (uv, Python ≥3.12, no dependencies), `docs/spec.md` (the assignment) and OpenSpec scaffolding. The assignment fixes the stack (uv, FastAPI, Pydantic v2, an LLM SDK with structured output) and the repo layout (`app/`, `prompts/`, `evals/`). Requirements are in `specs/`. Motivation is in proposal.md.

Constraints that shape the design:
- The whole eval set, run for at least 3 prompt versions plus a second model, must fit the Gemini free tier, which has per-minute rate limits.
- Tests must not call the real LLM. TDD goes through the public interface (HTTP endpoints, the eval CLI), and only the external model boundary is replaced.
- The student must be able to explain every line, so the code stays plain and avoids clever abstractions.

## Goals / Non-Goals

**Goals:**
- One classification path, shared by `/classify`, `/classify/batch` and `run_eval.py`, so eval numbers describe what the service actually does.
- A single seam for faking the model in tests.
- Reproducible eval runs, traceable to the exact prompt, model and dataset.

**Non-Goals:**
- Caching, auth, rate limiting of clients, or persistence beyond result JSON files.
- Supporting LLM providers other than Gemini (the config allows other Gemini models only).
- Parallel eval runs: the runner is deliberately sequential.

## Decisions

### Layout: flat `app/` package, not a `src/` layout
`uv run uvicorn app.main:app` is the command the assignment requires, so `app/` sits at the repo root and the project is not installed as a package (no build backend). Tests and `evals/run_eval.py` import `app.*` with the repo root on the path. pytest gets this from `pythonpath = ["."]` in pyproject; `run_eval` gets it by running as `uv run python -m evals.run_eval`.
- *Alternative:* the `src/support_classifier` layout that uv generates. Rejected because it breaks the required uvicorn command and the recommended structure.

### Modules
- `app/config.py`: `Settings` built on `pydantic-settings`, reading `.env`. Fields: `gemini_api_key` (required), `gemini_model`, `temperature=0.0`, `max_retries=3`, `batch_concurrency=5`, `prompt_path="prompts/classify.md"`, `request_timeout_s=30` (passed to the Gemini client, which has no timeout by default).
- `app/schemas.py`: `Category` and `Urgency` as `str` Enums, `ClassifyRequest` (text validated: stripped non-empty, ≤4000 characters), `ClassifyResponse` (summary ≤150 characters), `BatchRequest` (1–50 `texts`, each checked by the same validator), `BatchItem` (result or error), and `ModelOutput`, the schema sent to Gemini (see below).
- `app/classifier.py`: a `Classifier` class holding the Gemini async client, the loaded prompt and the settings. Its public method is `async classify(text) -> ClassifyResponse`, which raises `ClassifierError` on final failure. It also contains the retry loop and the summary trimming.
- `app/main.py`: the FastAPI app. A lifespan handler builds the `Classifier` once (failing fast on a missing key or prompt) and closes the Gemini client at shutdown. The routes depend on `get_classifier()`, and a `ClassifierError` becomes a 502.

### Model seam for tests: FastAPI dependency override of `get_classifier`
Tests swap in a fake classifier object with the same `classify()` method, through `app.dependency_overrides`. This replaces only the external boundary (the LLM). Request validation, error mapping, batch ordering and concurrency all run for real.
- *Alternative:* patching `google.genai` internals. Rejected because it is brittle and couples tests to SDK details.
- Retry, trimming and output validation inside `Classifier` are tested by giving it a fake client that has the SDK's `aio.models.generate_content` shape, constructor-injected. That is still the external boundary, not an internal.

### Structured output: `ModelOutput` sent as `response_schema`
The call uses `response_mime_type="application/json"` and `response_schema=ModelOutput`, then reads `response.parsed`, falling back to `ModelOutput.model_validate_json(response.text)`. `ModelOutput.summary` is a plain `str` with no max length, so an overlong summary still parses and can be trimmed (spec: Summary length guarantee). Enums are `str` Enums because Gemini's schema conversion maps them reliably to `enum` (Literal support is less consistent across SDK versions). Any parse or validation failure is an "invalid model output" error.
- *Alternative:* putting `maxLength=150` in the schema and failing on violations. Rejected because of the grill decision to trim and keep the classification.

### Retries: small hand-written loop, no new dependency
Up to `max_retries` attempts with exponential backoff plus jitter (1s, 2s, 4s…). It retries on `google.genai.errors.APIError` with code 429 or ≥500, and on timeouts and connection errors (`httpx.TransportError`, which google-genai lets through unwrapped). Invalid output and 4xx errors other than 429 are not retried. About 15 lines, easy to explain.
- *Alternative:* `tenacity`. Rejected as an extra dependency for one loop.

### Batch: `asyncio.gather` under an `asyncio.Semaphore(batch_concurrency)`
Each item runs `classify()` inside the semaphore. A `ClassifierError` becomes `{"error": "classification failed"}` for that item. `gather` keeps the input order.

### Eval runner
- `evals/validate_dataset.py`: `validate(path) -> list[str]` (violations) plus a `__main__` that prints counts. `run_eval` imports and calls it before doing anything else.
- `evals/run_eval.py`: argparse (`--prompt`, `--model`, `--temperature`, `--dataset`). It builds a `Classifier` from `Settings` with the overrides applied and runs every row in a single `asyncio.run`, awaiting rows one by one. It catches `ClassifierError` (and any unexpected `Exception`) per row. It computes the metrics in plain Python (no pandas/sklearn), prints them as text tables, and writes `evals/results/<YYYY-MM-DD_HH-MM-SS>.json`. Prompt and dataset hashes are SHA-256 of the file bytes (the first 12 hex characters are shown).
- The result file stores token usage from `usage_metadata` when present. Cost for the model comparison is worked out in the README from the stored token counts and the provider's published prices, which are not hard-coded.

### Model names
The default model in `.env.example` and the comparison model are picked during apply from Google's current Gemini model list (a Flash-Lite-class default, the same model the evals ran on, and another Flash-Lite-class comparison, both with free-tier access), not from memory. They live only in config.

### Docker
A multi-stage-free single stage: `python:3.12-slim`, the `uv` binary copied from the official `ghcr.io/astral-sh/uv` image, then `uv sync --frozen --no-dev`, and finally `uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`. `.dockerignore` excludes `.env`, `.venv`, `.git`, `evals/results`. The compose file uses `env_file: .env` and maps port `8000:8000`.

### Tooling
`ruff` (default rules plus `I` for imports) and `pytest` as dev dependencies, configured in `pyproject.toml`. All functions carry type hints. `ruff check` must pass with no errors.

## Risks / Trade-offs

- [Free-tier limits or quota shrink during iterations] → Sequential eval, retries with backoff, and failures counted as wrong rather than hidden. If quota runs out, results show it as errors, and the run can be repeated later.
- [Model is non-deterministic even at temperature 0, so numbers wobble between runs] → Store every run. The README states the run each number comes from, and can note the variance if seen.
- [Gemini schema conversion rejects some Pydantic constructs] → Keep `ModelOutput` flat (two enums plus a string). Cover it with a test that builds the request config. Verify against the real API once during apply with a smoke call.
- [The student's eval set might accidentally overlap v3's few-shot examples] → The validator also checks that no v3 example text appears in the dataset (exact match after normalising case and whitespace).
- [Trimming hides a model that ignores the length rule] → Every trim is logged as a warning, and the eval result file counts how many summaries were trimmed.
- [Committed `openspec/` and `.claude/` reveal AI assistance] → AI assistance is allowed by the assignment. The student keeps both by choice.
