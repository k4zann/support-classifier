## 1. Health endpoint end-to-end (walking skeleton)

- [x] 1.1 Human checkpoint: confirm adding deps `fastapi`, `uvicorn`, `pydantic-settings`, `google-genai` and dev deps `pytest`, `httpx`, `ruff`, then `uv add` them. Verify: `uv.lock` exists and `uv sync` succeeds
- [x] 1.2 Configure ruff (default rules + `I`) and pytest (`pythonpath = ["."]`) in `pyproject.toml`, and extend `.gitignore` with `.env` and `evals/results/*.tmp`. Verify: `uv run ruff check .` and `uv run pytest` both run
- [x] 1.3 Write a failing test `GET /health → 200 {"status":"ok"}`, then add `app/main.py` with the route. Verify: the test passes, and `uv run uvicorn app.main:app` serves `/health` and `/docs`

## 2. Classify happy path with structured output

- [x] 2.1 Add `app/config.py` (Settings from `.env`) and `.env.example` listing all variables without values. Pick the default Gemini model from Google's current model list. Verify: a test shows that a missing `GEMINI_API_KEY` fails with a message naming it
- [x] 2.2 Add `prompts/classify.md`, loaded once at startup. Verify: a test shows startup fails with a message naming the file when it is missing or empty
- [x] 2.3 Write a failing endpoint test: `POST /classify` with a fake classifier (dependency override) returns 200 with exactly `category`, `urgency`, `summary`. Then add `app/schemas.py` (enums, request/response, `ModelOutput`) and the route. Verify: the test passes
- [x] 2.4 Write a failing test for `Classifier` with an injected fake Gemini client: it sends `response_schema=ModelOutput`, `response_mime_type="application/json"`, the configured model and temperature, and returns a `ClassifyResponse`. Then implement `app/classifier.py`. Verify: the test passes
- [x] 2.5 Smoke-test against the real Gemini API with a local `.env` (manual, needs the student's key). Verify: `curl -X POST /classify` with the spec's SMS example returns a valid classification

## 3. Input validation errors (422)

- [x] 3.1 Write failing tests for empty, whitespace-only, 4001-character, missing and non-string `text` → 422 with readable messages, and exactly 4000 characters → accepted, with no classifier call on rejection. Then add the validator to `ClassifyRequest`. Verify: all tests pass

## 4. Model failure handling (retry, 502, trimming)

- [x] 4.1 Write failing tests with a fake client: a 429 followed by success → result; 5xx on every attempt → `ClassifierError` after `max_retries`; invalid enum or missing field → `ClassifierError` with no retry; a non-429 4xx → no retry. Then implement the retry loop with backoff (backoff patched to 0 in tests). Verify: tests pass
- [x] 4.2 Write a failing endpoint test: a classifier raising `ClassifierError` → 502 with generic `detail`, provider text absent, and the error logged (checked with `caplog`). Then map it in `app/main.py`. Verify: the test passes
- [x] 4.3 Write a failing test: a 200-character model summary → returned summary ≤150 characters, ending on a whole word, with a warning logged; and the hard-cut case with no spaces. Then implement trimming. Verify: tests pass

## 5. Eval harness: dataset validation

- [x] 5.1 Write failing tests for `evals/validate_dataset.py` using small temp JSONL files: valid set passes and reports counts; <30 rows, a category <4, <8 hard, duplicate id, bad JSON line, bad label and a v3 few-shot text present in the dataset each produce a named violation. Then implement it plus its `__main__`. Verify: tests pass, and `uv run python -m evals.validate_dataset` prints counts or violations
- [ ] 5.2 Student task: write and label `evals/dataset.jsonl` by hand (30+ rows, ≥4 per category, ≥8 `hard`) before any model run. Verify: `uv run python -m evals.validate_dataset` passes

## 6. Eval harness: runner, metrics and saved results

- [ ] 6.1 Write failing tests for `evals/run_eval.py` with a fake classifier on a tiny dataset: correct category, urgency and hard-only accuracy; confusion matrix cells; error list entries; a failing row counted as wrong in every denominator without stopping the run; an invalid dataset → non-zero exit with no classifier calls. Then implement the runner. Verify: tests pass
- [ ] 6.2 Write a failing test: the run writes `evals/results/<timestamp>.json` with model, temperature, prompt path and hash, dataset hash and row count, metrics, per-row predictions, errors, latency, token usage and the trimmed-summary count; and `--prompt`, `--model`, `--temperature` override config. Then implement it. Verify: the test passes, and a real run on the student's dataset produces a result file

## 7. Prompt versions and iterations

- [ ] 7.1 Draft `prompts/versions/v1.md` (basic instruction), copy it to `prompts/classify.md`, and run the eval. Verify: a result file exists for v1. Student commits the prompt change separately
- [ ] 7.2 Draft `v2.md` (adds category and urgency descriptions from the spec), update `classify.md`, run the eval. Verify: a v2 result file exists. Student commits separately
- [ ] 7.3 Draft `v3.md` (adds 3 few-shot examples not in the dataset), update `classify.md`, run the eval. Verify: a v3 result file exists, and the validator's few-shot overlap check passes. Student commits separately
- [ ] 7.4 Student task: inspect the errors from each run and decide on any further versions. If the dataset changes, re-run v1–v3 on it

## 8. Batch endpoint (bonus)

- [ ] 8.1 Write failing endpoint tests with a fake classifier: 3 texts → 3 results in order; a failing middle item → `{"error": ...}` in place while the others succeed; `[]`, 51 items and one invalid item → 422 with no calls; with concurrency 5 and 50 items, the fake records ≤5 in flight. Then implement `BatchRequest`/`BatchItem` and the `POST /classify/batch` route with a semaphore. Verify: tests pass

## 9. Docker (bonus)

- [ ] 9.1 Add `Dockerfile`, `.dockerignore` (excludes `.env`, `.venv`, `.git`, `evals/results`) and `docker-compose.yml` (`env_file: .env`, port 8000). Verify: `docker compose up --build`, then `curl localhost:8000/health` → `{"status":"ok"}`, and `docker compose run --rm app ls -a` shows no `.env`

## 10. Model comparison (bonus)

- [ ] 10.1 Pick the comparison model from Google's current list and run the eval with the final prompt on both models. Verify: two result files with different `model` values exist, and a comparison table (accuracy, mean latency, tokens and estimated cost) is drafted for the README

## 11. README and final checks

- [ ] 11.1 Write the README: setup in 3–4 commands (local and Docker), model and provider choice with reasons, final metrics, the iteration log table filled from result files, the model comparison, and placeholders for the student's error analysis (3–5 errors) and honest conclusion. Verify: every number in the README matches a file in `evals/results/`
- [ ] 11.2 Student task: write the error analysis and the conclusion in the README
- [ ] 11.3 Final check: `uv run ruff check .` and `uv run pytest` pass; a fresh clone works with `cp .env.example .env` + key, then `uv sync` and `uv run uvicorn app.main:app`; `git log -p | grep -i "api_key="` shows no secret. Verify: all commands succeed
