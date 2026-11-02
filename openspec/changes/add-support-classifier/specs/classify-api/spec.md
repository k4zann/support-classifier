## Purpose

HTTP API that classifies Russian-language customer support requests into a fixed category and urgency, with a short summary, and reports model failures as explicit errors.

## ADDED Requirements

### Requirement: Health check
The service SHALL expose `GET /health`, which returns HTTP 200 with body `{"status": "ok"}` without calling the model.

#### Scenario: Service is up
- **WHEN** a client sends `GET /health`
- **THEN** the response is 200 with body `{"status": "ok"}`

### Requirement: Classify a single request
The service SHALL expose `POST /classify`, which accepts `{"text": <string>}` and returns HTTP 200 with `{"category", "urgency", "summary"}`. `category` MUST be one of `auth`, `payment`, `delivery`, `technical`, `other`. `urgency` MUST be one of `high`, `medium`, `low`. `summary` MUST be a non-empty string of at most 150 characters.

#### Scenario: Valid request
- **WHEN** a client posts `{"text": "Не приходит код из SMS, третий день не могу войти в приложение"}` and the model answers with a valid classification
- **THEN** the response is 200 and the body has exactly the fields `category`, `urgency` and `summary`, with values from the allowed sets

#### Scenario: Model output is constrained to the allowed values
- **WHEN** the model is asked to classify any text
- **THEN** the request to the model carries the response JSON schema with the enum values, so free-text parsing is never needed

### Requirement: Input validation
The service SHALL reject a `text` that is empty, made up only of whitespace, longer than 4000 characters, missing, or not a string, with HTTP 422 and a readable error message. The model MUST NOT be called for rejected input.

#### Scenario: Empty text
- **WHEN** a client posts `{"text": ""}`
- **THEN** the response is 422 with a message saying the text must not be empty, and no model call is made

#### Scenario: Whitespace-only text
- **WHEN** a client posts `{"text": "   \n "}`
- **THEN** the response is 422

#### Scenario: Text too long
- **WHEN** a client posts a text of 4001 characters
- **THEN** the response is 422 with a message naming the 4000-character limit

#### Scenario: Text at the limit
- **WHEN** a client posts a text of exactly 4000 non-whitespace characters
- **THEN** the request is accepted and classified

### Requirement: Model failure handling
The service SHALL retry transient model failures (rate limit, server error, timeout) up to a configured number of attempts with backoff. If the model is still unavailable, or returns output that does not match the schema (missing field, value outside the enums), the service SHALL respond with HTTP 502 and a generic `detail` message, and SHALL log the underlying error. The 502 body MUST NOT include provider error text or secrets.

#### Scenario: Transient failure then success
- **WHEN** the first model call fails with a rate-limit error and the retry succeeds
- **THEN** the response is 200 with the classification

#### Scenario: Model stays unavailable
- **WHEN** every attempt fails with a server error
- **THEN** the response is 502 with a generic `detail` message, and the error is logged

#### Scenario: Invalid model output
- **WHEN** the model returns a category outside the allowed set
- **THEN** the response is 502 and the error is logged

### Requirement: Summary length guarantee
The service SHALL never return a `summary` longer than 150 characters. If the model returns a longer summary along with a valid category and urgency, the service SHALL cut it at the last word boundary within 150 characters (or hard-cut if there is none), return 200, and log a warning.

#### Scenario: Overlong summary
- **WHEN** the model returns a valid category and urgency with a 200-character summary
- **THEN** the response is 200, the summary is at most 150 characters and ends on a whole word, and a warning is logged

### Requirement: Batch classification
The service SHALL expose `POST /classify/batch`, which accepts `{"texts": [<string>, ...]}` with 1 to 50 items, each following the same rules as single classification. If the list is empty, holds more than 50 items, or any item is invalid, the whole request SHALL be rejected with 422 and no model calls. Otherwise the service SHALL return HTTP 200 with `{"results": [...]}` in the same order as the input. Each item is either `{"category", "urgency", "summary"}` or `{"error": <generic message>}`. The number of concurrent model calls SHALL be limited by a configured value (default 5).

#### Scenario: All items succeed
- **WHEN** a client posts three valid texts
- **THEN** the response is 200 with three results in input order, each holding a classification

#### Scenario: One item fails
- **WHEN** a client posts three valid texts and the model call for the second one fails after retries
- **THEN** the response is 200, items 1 and 3 hold classifications, and item 2 holds an `error`

#### Scenario: Invalid item rejects the batch
- **WHEN** a client posts `["ok text", ""]`
- **THEN** the response is 422 and no model call is made

#### Scenario: Too many items
- **WHEN** a client posts 51 texts
- **THEN** the response is 422

#### Scenario: Concurrency limit
- **WHEN** a client posts 50 texts with the concurrency limit set to 5
- **THEN** no more than 5 model calls are in flight at any moment

### Requirement: Configuration and secrets
The service SHALL read the API key, model name, temperature (default 0), retry count and batch concurrency from environment variables or a `.env` file. None of these may be hard-coded. The repository MUST NOT contain the API key: `.env` is gitignored and `.env.example` lists the variables without values.

#### Scenario: Missing API key
- **WHEN** the service starts without an API key configured
- **THEN** startup fails with a clear message naming the missing variable

#### Scenario: Model switched by config
- **WHEN** the model name variable is changed and the service is restarted
- **THEN** classifications are requested from the new model without any code change
