# Eval runs per prompt version

Every number below is copied from the result file named in its row. All runs use
`evals/dataset.jsonl` (42 rows, 18 hard, dataset hash `de3da1692a4c`) and temperature 0.

Command: `MAX_RETRIES=7 uv run python -m evals.run_eval --prompt prompts/versions/vN.md --model gemini-3.5-flash-lite`
(`MAX_RETRIES=7` lets the retry backoff wait out the free tier's 15 requests/minute limit.)

## Valid runs

| Version | What changed | Result file | Prompt hash | category | urgency | hard category | hard urgency | failed calls | mean latency, s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Basic instruction, no descriptions or examples | `2026-11-02_17-53-00.json` | `3bc91e416af3` | 39/42 (92.9%) | 34/42 (81.0%) | 16/18 (88.9%) | 14/18 (77.8%) | 0 | 3.9 |
| v2 | Adds category and urgency descriptions from the spec | `2026-11-02_18-10-57.json` | `2231334c5bed` | 41/42 (97.6%) | 37/42 (88.1%) | 17/18 (94.4%) | 15/18 (83.3%) | 0 | 3.2 |
| v3 | Adds 3 few-shot examples (not in the dataset) | `2026-11-02_18-15-17.json` | `dc8529b602a2` | 41/42 (97.6%) | 33/42 (78.6%) | 17/18 (94.4%) | 14/18 (77.8%) | 0 | 2.7 |

## Invalid runs (quota errors, kept only as a record)

| Result file | Prompt | Model | Failed calls | Why |
| --- | --- | --- | --- | --- |
| `2026-11-02_17-45-49.json` | v1 | gemini-3.5-flash | 26 | Free tier: 20 requests/day |
| `2026-11-02_17-51-14.json` | v1 | gemini-3.5-flash-lite | 10 | Free tier: 15 requests/minute, before `MAX_RETRIES=7` |
