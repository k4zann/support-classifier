## Why

The course assignment (`docs/spec.md`) asks for a small Python service that classifies customer support requests with an LLM, plus an honest, reproducible measurement of its quality on a hand-written eval set. The repo is empty, so this change delivers the whole assignment, including three bonus tasks (with a fourth as a backup), in one planned change.

## What Changes

- New FastAPI service (uv, Python 3.12, Pydantic v2) with `GET /health`, `POST /classify` and `POST /classify/batch`.
- Classification calls Google Gemini through the `google-genai` SDK, using Gemini's own structured output (`response_schema` from the Pydantic model). There is no regex parsing.
- Fixed enums: `category` ∈ {auth, payment, delivery, technical, other}, `urgency` ∈ {high, medium, low}. `summary` is one Russian sentence of at most 150 characters.
- Error contract: invalid input → 422. Model unavailable or invalid response after retries → 502, logged.
- The system prompt lives in `prompts/classify.md`, and frozen versions are kept in `prompts/versions/v1..v3.md`. Model name, temperature, retries and batch concurrency come from `.env` / config.
- Eval tooling: a dataset validator and `evals/run_eval.py`. The script reports category, urgency and hard-only accuracy, a confusion matrix and the error list, and saves each run to `evals/results/<datetime>.json`. Flags: `--prompt`, `--model`, `--temperature`.
- Bonuses: pytest endpoint tests with a mocked model, Dockerfile + `docker compose up`, a comparison of two Gemini models, and the batch endpoint.
- README: setup in 3–4 commands, model choice, final metrics, iteration log, error analysis, conclusions.

## Capabilities

### New Capabilities
- `classify-api`: HTTP API for health, single and batch classification, the input validation and error codes, and how model output is constrained, retried and trimmed.
- `eval-harness`: dataset format and validation rules, the eval runner, the metrics it reports, saved run results, and the flags used for prompt iterations and the model comparison.
- `prompt-versioning`: where the live prompt and frozen versions live, how the service loads the prompt, and how a run is traced back to the exact prompt, model and dataset.
- `container-run`: running the service with `docker compose up`, with config taken from `.env`.

### Modified Capabilities
<!-- None: the repo has no existing specs. -->

## Impact

- New code: `app/` (main, schemas, classifier, config), `prompts/`, `evals/`, `tests/`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `README.md`.
- New dependencies (checkpoint before adding): `fastapi`, `uvicorn`, `pydantic-settings`, `google-genai`; dev: `pytest`, `httpx`, `ruff`.
- External system: the Gemini API, using a free-tier API key kept only in `.env` (gitignored).
- Out of scope: the eval set's content (written and labeled by hand by the student), the README error analysis and conclusions (written by the student), any UI, the SSE `/reply` endpoint, and creating or pushing the GitHub repo.
