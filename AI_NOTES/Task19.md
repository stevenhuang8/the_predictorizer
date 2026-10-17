# AI Task 19: leakage-safe LLM backtest and the ensemble

Done on 2026-10-16, except **the real run, which hasn't happened.** It needs `ANTHROPIC_API_KEY` (commented out in `.env.local`) and a budget, both the user's decision. No migration. The only live database change is the AI series backfills run the same day (AI_NOTES/Task18.md).

## Quick reference

```sh
uv run python -m eco_prediction.ai.llm.backtest --dry-run                          # counts + cost estimate, no API calls
uv run python -m eco_prediction.ai.llm.backtest --budget 50                        # Opus 5.5: ~588 forecasts, est. ~$44
uv run python -m eco_prediction.ai.llm.backtest --model claude-haiku-5-5 --budget 3   # est. ~$1.10
uv run python -m eco_prediction.ai.llm.backtest --binary-only --budget 10 --no-save
uv run pytest tests/test_llm_backtest.py
```

```python
from eco_prediction.ai.llm.backtest import run_llm_backtest
result = run_llm_backtest(conn, budget_usd=50)
result.summary["binary"]    # Brier: llm, market, base_rate, ensemble on the common questions
result.summary["numeric"]   # relative RMSE: llm, random_walk, arima, lightgbm, ensemble
result.summary["cost"]      # new calls, reused calls, spend, per call, failures
```

---

## Decisions (my defaults, not asked)

### The leakage rule: nothing forecast before 2026-07-01
- **Every current Claude model has a training data cutoff of June 2026.** That's Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5, per platform.claude.com/docs/en/about-claude/models/overview, fetched 2026-10-16. So `TRAINING_CUTOFFS` maps each to 2026-07-01.
- **The rule is on the forecast date, not the question's resolution date.** The task said to drop questions with `target_date < cutoff`, but that alone isn't enough:
  - A question asked in May that resolves in August still leaks. The model knows May–June news its May data doesn't.
  - Requiring the forecast date to be ≥ the cutoff means the model knows nothing the forecast's own data doesn't, and the outcome is necessarily later still.
- **Dropped forecasts are counted** (`dropped_before_cutoff`), whatever `start_date` asks for.
- **The harness gets `model_training_cutoff`.** It's optional and defaults to `None`; forecast dates before it are skipped. All 317 economy tests still pass.
  - The statistical models run with the same cutoff, so the comparison is on the same window. That's the task's "separate evaluation window". Their long backtests (Tasks 13–15) stay separate.

### What's evaluated (from the live database on 2026-10-16)
- **Binary: 574 curated Polymarket questions,** asked from 2026-07-01 to yesterday, whose market resolved yes or no. Each gets four forecasts:
  - The LLM.
  - The market's own price the day before (Polymarket's forecast; Metaculus is deferred).
  - The category base rate.
  - The ensemble of the three.
- **Outcomes come from the market when the resolutions table has none.** The AI resolution job (Task 11) hasn't run live, so it has no rows yet.
- **Numeric: 14 forecasts.** 7 series × the 2026-07-01 and 08-01 forecasts at 1 month ahead are the only targets published so far. 3- and 6-month targets resolve from December on.
  - The LLM adapter skips unpublished targets before calling. Without that check the dry run counted 84 numeric calls, 70 of them unscorable.
  - The check reads only whether a value exists, never the value.
  - Each forecast is scored against random walk, ARIMA and LightGBM.
- **The numeric set is too small to conclude anything.** It grows by 7 forecasts a month, plus 3- and 6-month ones once those resolve. The binary set is where the comparison means something now.

### Cost controls
- **`--budget` is required for a real run.**
  - The run stops before the next call once new spending reaches it.
  - Forecasts made so far are kept. A test caught an early version that lost them on the stop.
- **Task 18's reuse by context hash makes re-runs free.** A test shows the second run makes 0 calls and stores nothing twice.
- **`--dry-run` estimates cost** from the real prompts' length (mean about 3,900 tokens) and an assumed 3,000 output and thinking tokens:

  | Model | Per forecast | 588 forecasts |
  |---|---|---|
  | `claude-opus-5-5` | ~$0.076 | ~$44 |
  | `claude-haiku-5-5` | ~$0.002 | ~$1.10 |

  The output-token figure is a guess. The first real calls' `llm_calls.output_tokens` will replace it.
- **No default switch to Haiku.** The task says "use cheaper Haiku for bulk backtests"; its model id, `claude-haiku-3-20240320`, doesn't exist, and Haiku 5.5 is the current one.
  - The CLI defaults to Opus 5.5, the model live forecasts use, because backtesting a different model says less about the live one.
  - Haiku is one flag away (`--model claude-haiku-5-5`), and has the same cutoff, so the same window.

### Saving
- **Binary LLM forecasts are stored** in `forecasts` as backtests (`is_backtest = true`), under the forecaster's model version, linked to their `llm_calls` row. `--no-save` skips that.
- **Numeric backtest targets aren't questions in the database,** and making question rows for past months would leave them for the live resolution job to trip over. So those forecasts stay in the result frame and in `llm_calls`.

### The ensemble (`ai/llm/ensemble.py`)
- **`LLMEnsembleForecaster(method)`** offers three methods:
  - `equal`.
  - `inverse_variance`: weights proportional to 1 / past MSE.
  - `learned`: the 0.05-step grid point on the weight simplex with the lowest past error (up to 3 components; inverse variance beyond).
- **Walk-forward:** a forecast's weights use only outcomes resolved before its ask date (binary), or actuals published by its data cutoff (numeric).
- **How components combine:**
  - Binary forecasts are a linear pool.
  - Numeric points and bounds are averaged separately (quantile averaging).
  - Errors are relative (forecast / actual − 1), so series pool.
  - A missing component drops out and the weights renormalize.
- **Cold start:** until 10 outcomes are known it uses equal weights.
  - A test shows the cost. On 60 synthetic questions the ensemble's Brier is 0.028, against 0.006 for the best component alone, because the first questions pay for the noisy one.
  - Once warmed up it matches the best component, and it beats equal weights throughout.
  - **For the 14 numeric forecasts it will stay at or near equal weights.** "The ensemble doesn't degrade performance" can only be judged on the binary set.

---

## What I did
- **`backtest/harness.py`:** `model_training_cutoff` (optional; economy behaviour unchanged).
- **`src/eco_prediction/ai/llm/backtest.py` (new):**
  - `run_llm_backtest` and `backtest_questions`.
  - `LLMNumericForecaster`, the harness adapter.
  - `BacktestBudget` / `BudgetExceeded`, `TRAINING_CUTOFFS` and `summarize_backtest`.
  - The CLI.
- **`src/eco_prediction/ai/llm/ensemble.py` (new):** `LLMEnsembleForecaster` and `simplex_grid`.
- **`tests/test_llm_backtest.py`: 12 tests, not tagged `economy`. The API is never called.**
  - **The cutoff:**
    - The harness skips dates before it, and nothing changes without one.
    - A mid-month cutoff starts the next month.
    - Per-model lookup; an unknown model raises.
  - **The budget:** counts new spending only, and raises past its limit.
  - **The ensemble:**
    - The simplex grid.
    - Each method's weights, and equal weights under `min_history`.
    - Renormalization when a component is missing.
    - Binary walk-forward: history only from earlier resolutions, and the cold-start and warmed-up claims above.
    - Numeric walk-forward: weights only from published actuals, and the blended interval is ordered.
  - **Full runs on a seeded database with a fake client.** The fake answers from the prompt, well when the question hints at the outcome.
    - **Binary:**
      - The question asked before the cutoff is dropped.
      - Every prompt is as of the day before the ask date.
      - Cost is exact ($0.036 a call).
      - The LLM beats both the base rate and the market, and the ensemble is no worse than the worst.
      - Forecasts are saved as backtests and linked to their calls.
      - A re-run makes no calls and no duplicates.
    - **The budget stop:** after two calls the third is refused, a malformed reply is counted as a failure, and the forecasts made are kept.
    - **Numeric:**
      - Only published targets are called (2 of 4 dates).
      - All four models are scored.
      - The ensemble covers the common forecasts.
    - **The adapter** returns NaN for an unpublished target without calling.

## Checks
- `uv run pytest`: **639 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **A dry run on the live database** (no calls, no writes, about 50 s): 574 binary and 14 numeric LLM forecasts, with prompts averaging about 3,900 tokens. Costs as in the table above.

## Not done: the real run
- **Needed first:** `ANTHROPIC_API_KEY` uncommented in `.env.local`, and a choice of model and budget.
- **What it would answer** is the PRD's success criterion: does the LLM beat the base rate after its training cutoff? Here that's measured on about 574 binary questions.
- **What if it doesn't:** the PRD says to report that and keep the LLM out of live use.

## Where I changed the task's plan
- **The cutoff applies to the forecast date,** not only the target date (above).
- **Model:** `claude-opus-5-5` by default with Haiku 5.5 as an option. The task's model ids don't exist.
- **Benchmarks:** Polymarket only, since Metaculus is deferred.
- **Numeric forecasts aren't saved as `forecasts` rows** (no question rows for past months).
- **Added:** the budget, the dry-run estimate, re-run reuse, and skipping unscorable numeric targets.

## Tradeoffs
- **Showing the market price to the LLM** makes the "LLM vs market" comparison partly circular: the LLM can lean on the price. A `baselines=False` run (no market price in the prompt) would measure independent skill. It's not wired to the CLI yet; a candidate for Task 20.
- **Polymarket questions cluster:** dozens of legs of one event ("which company has the best model"), and several leads per market (90/30/7 days). The 574 questions are far from independent, so Brier differences need a per-market bootstrap before they mean much. That's for Task 20.
- **A June 2026 cutoff for every model is the published month,** taken as ending 2026-06-30. If a model saw some July data the window would leak slightly. Moving `TRAINING_CUTOFFS` a month later costs about a fifth of the binary set.
- **Reconstructed data:** numeric actuals and features over the window are backfilled values (hindsight), as in Tasks 12–15. Live pulls replace them going forward.
