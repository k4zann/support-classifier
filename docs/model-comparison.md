# Model comparison (draft for the README, task 10.1)

Both runs use the final prompt `prompts/classify.md` (= `prompts/versions/v3.md`, same bytes,
prompt hash `dc8529b602a2`), `evals/dataset.jsonl` (42 rows, 18 hard, dataset hash
`de3da1692a4c`) and temperature 0. Every number below comes from the two result files.

Command: `MAX_RETRIES=7 uv run python -m evals.run_eval --model <model>`
(the v3 Flash-Lite run was started with `--prompt prompts/versions/v3.md`, which has the same hash).

| | gemini-3.5-flash-lite | gemini-3.1-flash-lite |
| --- | --- | --- |
| Result file | `2026-11-02_18-15-17.json` | `2026-11-02_19-17-16.json` |
| Category accuracy | 41/42 (97.6%) | 42/42 (100.0%) |
| Urgency accuracy | 33/42 (78.6%) | 31/42 (73.8%) |
| Hard category accuracy | 17/18 (94.4%) | 18/18 (100.0%) |
| Hard urgency accuracy | 14/18 (77.8%) | 12/18 (66.7%) |
| Failed calls | 0 | 0 |
| Mean latency per row, s | 2.72 | 5.62 |
| Median latency per row, s | 0.88 | 3.64 |
| Input tokens (42 rows) | 17,465 | 17,465 |
| Output tokens (42 rows, no thinking tokens) | 1,564 | 1,484 |
| Price, input / output per 1M tokens | $0.30 / $2.50 | $0.25 / $1.50 |
| Estimated cost, 42 rows | $0.0092 | $0.0066 |
| Estimated cost per 1,000 requests | $0.22 | $0.16 |

Notes:
- **Cost** = input tokens × input price + output tokens × output price, using the paid-tier
  Standard prices from <https://ai.google.dev/gemini-api/docs/pricing> (page "Last updated
  2026-10-30 UTC", checked 2026-11-02). Both runs used the **free tier**, where both models are
  "Free of charge", so the real cost of these runs was $0.
- **Latency** is wall time per row and includes retry backoff. The classifier retries 429s
  (the free tier's per-minute limit, 15 requests per minute for gemini-3.5-flash-lite) and 503s,
  and gemini-3.1-flash-lite returned 503 "high demand" to a probe call just before its run. The
  slowest rows (max 37.9 s and 39.0 s) are most likely retry waits, not model time; the result
  files do not record retries, so this is not verified. The median is
  shown because it is less affected by those waits.
- Summaries trimmed: 0 in both runs.

## Why these two models

The design asked for Flash vs Flash-Lite. That was not possible on the free tier:

- `gemini-3.6-flash`: the free tier allows **20 requests per day** for this model (quota error
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue: 20`), and 42 rows need 42.
  The attempt is kept as an invalid run: `2026-11-02_18-29-23.json` (41 of 42 calls failed with 429).
- `gemini-3.8-flash`: answered a single probe call with 503 "high demand" on 2026-11-02 (earlier it
  gave 504s on this key).

So the comparison is between two Flash-Lite generations from the current model list
(<https://ai.google.dev/gemini-api/docs/models>, "Last updated 2026-10-28 UTC", checked 2026-11-02):
the newer gemini-3.5-flash-lite (the model used for the prompt iterations) and the older, cheaper
gemini-3.1-flash-lite. Both are stable, free-tier models and both returned valid structured output
in every row.
