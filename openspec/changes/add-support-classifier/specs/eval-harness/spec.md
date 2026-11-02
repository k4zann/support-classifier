## Purpose

Measures classification quality on a fixed, hand-labeled dataset and saves comparable results for every run, so prompt and model changes are judged by numbers.

## ADDED Requirements

### Requirement: Dataset format and validation
The eval set SHALL be `evals/dataset.jsonl`, with one JSON object per line holding `id` (unique integer), `text` (non-empty string), `category` and `urgency` (allowed values only), and an optional boolean `hard`. A validation command SHALL fail with a list of every violation when the set has fewer than 30 rows, any category has fewer than 4 rows, fewer than 8 rows have `hard: true`, an id repeats, a line is not valid JSON, or a label is outside the allowed values. The runner SHALL refuse to run on an invalid dataset.

#### Scenario: Valid dataset
- **WHEN** the dataset has 30+ rows, every category at least 4 times, and 8+ hard rows with valid labels
- **THEN** validation passes and reports the counts per category, per urgency and for hard rows

#### Scenario: Category under-represented
- **WHEN** the `other` category has only 3 rows
- **THEN** validation fails and names `other` and its count

#### Scenario: Invalid label
- **WHEN** a row has `"category": "billing"`
- **THEN** validation fails and names the row's id and the bad value

#### Scenario: Runner on invalid dataset
- **WHEN** the eval runner starts and the dataset fails validation
- **THEN** it exits non-zero without calling the model

### Requirement: Eval run and metrics
The eval runner SHALL classify every dataset row through the same classification logic the service uses, one row at a time, and SHALL print: category accuracy, urgency accuracy, category accuracy and urgency accuracy on hard rows only, a category confusion matrix (expected × predicted), and every error with id, text, expected labels and predicted labels. A row whose model call fails after retries SHALL count as wrong for both category and urgency, appear in the error list with the error message, and be included in every accuracy denominator.

#### Scenario: Metrics printed
- **WHEN** the runner completes on a valid dataset
- **THEN** it prints category accuracy, urgency accuracy, hard-row accuracy for both, the confusion matrix and the error list

#### Scenario: Failed call counts as wrong
- **WHEN** one of 30 rows fails after retries and the other 29 are correct
- **THEN** category accuracy is 29/30, and the failed row appears in the error list marked as an error

#### Scenario: Runner never crashes on a row
- **WHEN** any single row raises an error during classification
- **THEN** the runner records it and continues to the next row

### Requirement: Saved run results
Each eval run SHALL save a JSON file to `evals/results/<YYYY-MM-DD_HH-MM-SS>.json` with: timestamp, model name, temperature, prompt file path and content hash, dataset hash and row count, the retry count, batch concurrency and request timeout used, all metrics, the confusion matrix, per-row predictions with the number of model call attempts, the error list, and the per-row and total latency. The hashes SHALL describe the files as they were when classification started. A run SHALL NOT overwrite an existing result file.

#### Scenario: Result file written
- **WHEN** a run finishes
- **THEN** a new file exists in `evals/results/` holding the model, temperature, prompt hash, dataset hash and all metrics

#### Scenario: Runs are comparable
- **WHEN** two runs use different prompt files on the same dataset
- **THEN** their result files have the same dataset hash and different prompt hashes

#### Scenario: Two runs in the same second
- **WHEN** a run is saved with the same timestamp as an existing result file
- **THEN** both files exist and the first is unchanged

### Requirement: Run parameters for iterations and model comparison
The eval runner SHALL accept `--prompt <path>`, `--model <name>` and `--temperature <float>`, which override the configured values for that run only. Without these flags it SHALL use the service's configured prompt, model and temperature.

#### Scenario: Evaluate a frozen prompt version
- **WHEN** the runner is started with `--prompt prompts/versions/v1.md`
- **THEN** classification uses that prompt, and the result file records that path and its hash

#### Scenario: Compare two models
- **WHEN** the runner is run once with `--model A` and once with `--model B`
- **THEN** each result file records its model, accuracy and latency, so the README can compare them
