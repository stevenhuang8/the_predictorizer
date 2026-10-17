# AI Task 18: the Claude forecaster

Done on 2026-10-16, except **the one real API call the task asks for, which hasn't been made.** `ANTHROPIC_API_KEY` is commented out in `.env` and `.env.local`. Every test mocks the API, and nothing has been spent. No migration, and nothing written to the live database.

## Quick reference

```python
from eco_prediction.ai.llm.claude_forecaster import ClaudeForecaster, register_prompt_version
from eco_prediction.ai.llm.context import build_context_bundle, load_question

question = load_question(conn, 2585)                         # any AI question id
bundle = build_context_bundle(conn, question, date(2026, 10, 15))   # cutoff: the day before
forecaster = ClaudeForecaster(conn=conn)                     # logs to llm_calls; reuses identical calls
forecaster.predict_binary(question, bundle)                  # {'probability': .., 'rationale': .., 'call_id': .., 'cost_usd': ..}
forecaster.predict_numeric(question, bundle, question.horizon_months)
conn.commit()
```

```sh
uv run pytest tests/test_claude_forecaster.py
```

---

## Decisions (my defaults, not asked)

### The model: `claude-opus-5-5`, not the task's `claude-opus-4-20250514`
- **The task's model is long retired.** Opus 5.5 is the current Opus ($4 / $20 per million tokens). It's `MODEL`, and `ClaudeForecaster(model=...)` overrides it.
- **Adaptive thinking at `effort="high"`.**
  - Opus 5.5 can't turn thinking off, and its default effort is medium.
  - Forecasting is reasoning-heavy, so I set high explicitly. Task 20 can compare medium against high on cost and Brier.

### Structured output, rationale first
- **`output_config.format` constrains the reply to a JSON schema:**
  - `{rationale, point_forecast, lower_80, upper_80}` for numeric questions.
  - `{rationale, probability}` for binary ones.
- **`rationale` comes first** so the model explains before it commits to numbers.
- **The schema can't enforce coherence,** so the code checks it:
  - lower ≤ point ≤ upper, all finite.
  - A probability in [0, 1].
  - Anything else raises.

### Refusals: server-side fallback is on
- **`fallbacks: "default"`** (beta `server-side-fallback-2026-07-01`) re-runs a request refused by a safety classifier on Anthropic's recommended model, inside the same call.
- **AI-progress forecasting is close to the `frontier_llm` category,** so this matters more here than usual.
- **Cost is priced per attempt** from `usage.iterations`, so a fallback is billed at its own model's rates.
- **The answering model is recorded:** if a fallback answered, the result carries `served_by`.
- **A refusal that survives the fallback** is logged and raises.
- **`fallbacks=False`** turns it off.

### Errors
- **Every failure is logged to `llm_calls` with its `error`, then raises `ForecastError`.** The failures covered:
  - API errors.
  - A refusal.
  - `max_tokens` truncation.
  - Unparseable or incoherent JSON.
- **`retryable`** is true for rate limits, 5xx/529 overload, timeouts and connection errors. The SDK has already retried those 4 times (`max_retries`), so Task 21's job decides whether to try again later.
- **A numeric question with no series values known by the cutoff raises before any call.** This was added after I saw the live database (below).

### The context bundle (`ai/llm/context.py`)
- **Recent values:** the series' last 12 months, as known on the cutoff. That's for numeric and threshold questions. Curated questions have no series.
- **Baselines:**
  - Numeric questions: random walk, 12-month linear trend and ARIMA, fit point-in-time.
  - Threshold questions: the historical base rate.
  - Curated questions: the market price on the cutoff, plus the category base rate.
  - A model that can't answer is left out.
- **Showing the market price is a choice.** It's the strongest baseline, and the LLM may lean on it. Task 20 should also compare against a variant without it (`baselines=False` drops all of them).
- **Documents** are retrieved with a quota per type (4 arXiv, 3 Polymarket, 2 HF cards, 2 releases), from the last 365 days. That's the fix for Polymarket crowding seen in Task 17.
  - The query is the question text for curated questions, and a per-series description for the rest (`SERIES_QUERIES`).
  - Each document is cut to 1,500 characters in the prompt.

### The prompt
- **The system prompt says** the date, to reason only from what's given, to start from base rates and adjust, and to be calibrated.
- **The user message holds** the question, resolution rule, values, baselines and documents, in XML tags, then the task.
- **The system prompt can't remove what the model learned in training.** A backtest before Opus 5.5's training cutoff is contaminated whatever the prompt says. That's Task 19's problem ("leakage-safe backtest"), and it's the main reason live forecasts are the real test.
- **No prompt caching:** at about 3k tokens, a request is too short to reach the minimum cacheable prefix.

### Logging and reuse
- **Every call writes an `llm_calls` row** (Task 16's table): prompt, raw and parsed reply, all token counts including cache reads and writes, `cost_usd`, `prompt_version`, `as_of_date` and `question_id`.
- **The context hash covers** the model, system prompt, prompt, every request parameter and the prompt version.
  - A matching earlier *successful* call is returned instead of paying again. A backtest re-run then costs nothing.
  - Failures are never reused. `reuse=False` forces a new call.
- **The key** is read from the environment, then `.env.local` (as the task says), then `.env` (`anthropic_api_key`). A commented-out line counts as unset. Added after the user asked why the key wasn't found: both files have it commented out.
- **`register_prompt_version(conn, forecaster)`** records `claude_llm` / `<model>:<prompt version>:<effort>` in `model_versions`, with a hash of the prompt and schemas as `code_hash`. That's for Task 21's stored forecasts. Changing the prompt means bumping `PROMPT_VERSION`.

---

## What I did
- **`pyproject.toml` / `uv.lock`:** added `anthropic` (1.15.0).
- **`src/eco_prediction/ai/llm/context.py` (new):** `LLMQuestion`, `load_question`, `Baseline`, `ContextBundle` and `build_context_bundle`.
- **`src/eco_prediction/ai/llm/claude_forecaster.py` (new):**
  - `ClaudeForecaster` (`predict_numeric`, `predict_binary`) and `ForecastError`.
  - `build_prompt`, `usage_and_cost` and `PRICES`.
  - `register_prompt_version`.
- **`tests/test_claude_forecaster.py`: 16 tests, not tagged `economy`. The API is never called.**
  - **The prompt:**
    - Every part of the bundle appears, rendered exactly.
    - It's deterministic.
    - The binary version asks for a probability.
  - **The request:**
    - Model, system prompt, adaptive thinking, effort and schema.
    - `fallbacks: "default"` plus its beta, and the version without them.
  - **Guards:** wrong question type, wrong horizon, and no data (no request is made).
  - **Malformed replies:** not JSON, not an object, an incoherent interval, a non-numeric value, a probability out of range.
  - **Errors:**
    - A refusal (with its category) and `max_tokens` raise.
    - A rate limit and a connection error are retryable; a 400 isn't.
  - **Cost:**
    - Input, output, cache-write and cache-read rates.
    - A fallback priced per attempt.
    - An unknown model gives no cost.
  - **The database:**
    - A failed call is logged and not reused. A success is logged and reused with no new request.
    - `reuse=False` and a changed prompt both make new calls.
    - An API error is logged without a response.
  - **Prompt versions:** registered once per model and effort.
  - **Bundles:**
    - Numeric: 12 months as known on the cutoff, the three baselines, and documents within quota, cutoff and the 365-day window.
    - Curated: the market price as of the cutoff, not a later one.
  - **The real SDK over a mock HTTP transport:**
    - The wire request is checked: `/v1/messages`, the beta header, `fallbacks` in the body.
    - A real `BetaMessage` is parsed and priced.

## Checks
- `uv run pytest`: **625 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **Real bundles from the live database** (read-only, cutoff 2026-10-15):
  - **Question 82 (arXiv, Nov 2026):** about 3,400 tokens of prompt. It has documents from every type in the corpus so far, but **no values and no baselines.**
  - **Question 2585 (curated, "OpenAI announces it has achieved AGI before 2027?"):** about 2,400 tokens, with the market price and category base rate.

### Resolved the same day: AI series data loaded
- **All four backfills were run live on 2026-10-16:** arXiv (201 months), GitHub (117), Hugging Face (54) and Epoch (four series).
- **Every numeric question's bundle now has 12 recent values and its baselines.** ARIMA is missing only for SWE-bench (23 months) and ARC-AGI-2 (19), which are under its 24-month minimum.
- **Question 82's prompt** now shows arXiv's last 12 months (11,245 in 2026-09) and three baselines (random walk 11,245; trend 11,156; ARIMA 10,254).
- **Baseline values are now plain floats.** NumPy floats from the trend model had leaked into the bundle.

### What it looked like before the backfills
- **0 AI observations:** the arXiv, Hugging Face, GitHub and Epoch backfills (Tasks 4–7) were never run live, though the 21 numeric questions exist.
- **Effect:** numeric questions can't be forecast yet (the new guard refuses them), and threshold questions get no base rate.
- **Fix:** run each source's backfill, `uv run python -m eco_prediction.ai.ingest_arxiv` and so on (see AI_NOTES/Task4–7.md for each command and its run time). Curated questions are unaffected.

### Not done: the real API call
- **The task's manual check** (one question, real API, judge the output) needs `ANTHROPIC_API_KEY` set.
- **What it would show:**
  - The real cost per forecast. My estimate is about $0.05–0.10: roughly 3k input tokens at $4 per million, plus 2–4k output and thinking tokens at $20 per million. That's unmeasured.
  - The quality of the rationale.
- **Budgeting:** at that rate, one pass over the 2,440 binary questions would be about $120–250. Task 19's backtest needs a budget decision before it runs.

## Where I changed the task's plan
- **Model:** `claude-opus-5-5` with adaptive thinking and refusal fallbacks (above).
- **`ClaudeForecaster(model, api_key)`** also takes `conn`, `client`, `effort`, `max_tokens`, `fallbacks` and `reuse`.
- **`build_context_bundle(conn, question, as_of_date)`** takes an `LLMQuestion` (from `load_question`) and lives in `ai/llm/context.py`, not the forecaster module.
- **Added:** per-type document quotas, the no-data guard, call reuse by context hash, and `served_by`.
- **Prompt version is tracked** in `model_versions` with the model and effort in the tag, plus a hash of the prompt.

## Tradeoffs
- **Training-data contamination is the big one.** The model may simply know how 2024–2025 questions resolved. Only Task 19's design (forecasts after the training cutoff, or a check that the model doesn't recall outcomes) can tell skill from memory.
- **The market price in the prompt** makes the LLM partly a market-follower on curated questions. That's good for accuracy, but it means less independent signal for the ensemble.
- **Retrieval is lexical with fixed per-series queries.** It works for the benchmarks; "growth of AI research papers" is a weak query for arXiv counts. Documents may matter less than the numbers for count series.
- **One sample per question.** No self-consistency (several samples, then averaged), which would multiply the cost. It's a candidate for Task 19 if calibration is poor.
- **Prices are hard-coded** (`PRICES`, as of 2026-10). A price change makes logged costs drift until the table is updated. Unknown models log `cost_usd` as NULL rather than a wrong number.
