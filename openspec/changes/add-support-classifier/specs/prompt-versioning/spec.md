## Purpose

Keeps the system prompt outside the code and keeps every prompt version reproducible, so each iteration can be re-run on the same dataset at any time.

## ADDED Requirements

### Requirement: Prompt loaded from file
The service SHALL read its system prompt from `prompts/classify.md` once at startup and SHALL fail at startup with a clear message if the file is missing or empty. The prompt text MUST NOT be embedded in code.

#### Scenario: Prompt file missing
- **WHEN** the service starts and `prompts/classify.md` does not exist
- **THEN** startup fails with a message naming the missing file

#### Scenario: Prompt change needs no code change
- **WHEN** the contents of `prompts/classify.md` change and the service restarts
- **THEN** classification uses the new prompt without any code change

### Requirement: Frozen prompt versions
Every prompt version that is evaluated SHALL be kept as its own file, `prompts/versions/v<N>.md`. The live `prompts/classify.md` SHALL match the newest version. At least three versions SHALL exist: v1 a basic instruction, v2 adding category and urgency descriptions, v3 adding three few-shot examples whose texts do not appear in the eval set.

#### Scenario: All versions can be re-run
- **WHEN** the eval set changes
- **THEN** every version can be re-evaluated by pointing the runner at `prompts/versions/v1.md`, `v2.md` and `v3.md`

#### Scenario: Few-shot examples kept apart from the eval set
- **WHEN** v3's few-shot example texts are compared with the dataset texts
- **THEN** none of them match a dataset row
