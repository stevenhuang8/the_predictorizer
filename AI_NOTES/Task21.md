# AI Task 21: the AI forecast and resolution jobs

Done on 2026-10-16. **Only dry runs have touched the live database.** Each did its work in one transaction and rolled it back, and the row counts were identical after. The first real run, and installing the launchd agent, are the user's to do. The LLM is off unless asked for; it's never been run (budget deferred).

## Quick reference

```sh
uv run python -m eco_prediction.ai.jobs.ai_forecast_job --dry-run                   # what a run would store; ~1 min, no writes
uv run python -m eco_prediction.ai.jobs.ai_forecast_job                             # a real run (ingest + resolve + forecast)
uv run python -m eco_prediction.ai.jobs.ai_forecast_job --once-per-month            # what launchd runs
uv run python -m eco_prediction.ai.jobs.ai_forecast_job --llm claude-haiku-5-5 --llm-budget 2   # with the LLM, capped
uv run python -m eco_prediction.ai.jobs.ai_resolution_job [--include-backtest] [--dry-run]
uv run eco-forecast ai-forecast ...    # the same, through the CLI
uv run eco-forecast ai-resolve ...
uv run pytest tests/test_ai_forecast_job.py
```

**Scheduling: launchd, like the economy job.** It isn't installed yet.

```sh
cp deploy/launchd/com.eco-prediction.ai-forecast-job.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.eco-prediction.ai-forecast-job.plist
launchctl print gui/$(id -u)/com.eco-prediction.ai-forecast-job      # status
```

- **The plist runs daily at 09:30** with `--once-per-month`, half an hour after the economy job, and logs to `logs/launchd-ai.log`.
- **Launchd, not cron:**
  - This machine has no crontab. The economy job runs as the `com.eco-prediction.forecast-job` launch agent.
  - Launchd runs a job missed while the Mac slept as soon as it wakes; cron doesn't.
  - A first draft of these notes suggested a cron line, copied from `NOTES/TASK22.md`, which predates the switch to launchd.

---

## Decisions (my defaults, not asked)

### The LLM is opt-in, with a hard per-run budget
- **The task said** "use Haiku for live forecasts to control costs, or make model configurable". Spending is deferred until all AI tasks are done (the user's call), so the job never calls Claude unless `--llm MODEL --llm-budget USD` is given.
- **`--llm` without a budget is refused,** and so is `--llm` with `--dry-run`.
- **When the budget runs out,** the LLM is switched off for the rest of the run, and every other model carries on.
  - A test caught a first version where running out during the numeric questions skipped all binary questions, the free models' forecasts included.
  - The next run fills in the missing LLM forecasts; stored ones are skipped.
- **LLM forecasts are stored** under the forecaster's registered model version (`claude_llm`, `<model>:<prompt version>:<effort>`), with the rationale, linked to their `llm_calls` row.

### Separate from the economy job, sharing its conventions
- **New package `ai/jobs/`,** with `ai_forecast_job.py` and `ai_resolution_job.py`. `scheduler/forecast_job.py` stays economy-only.
- **Reused from it:** `today()` (US Eastern), `setup_logging`, `code_version`, `feature_dict` and `model_parameters`.
- **Its own advisory lock** (`LOCK_KEY = 2_202_212`; the economy's is 2_202_210):
  - An AI run never waits on an economy run, and vice versa.
  - The AI resolution job shares the AI lock, so it never overlaps an AI forecast run.
  - Tests: the AI lock held elsewhere stops a run, and the economy lock held elsewhere doesn't.
- **Each forecast commits on its own,** and one already stored for the same question, model version and date is skipped. Runs are safe to repeat; a test reruns and stores nothing new.
- **`--once-per-month`** reuses the month's first numeric forecast date, as the economy job does.

### What a run does
1. **Resolves and scores** with Task 11's `resolve_and_score_ai`. The dry run on the live database showed **2,276 AI questions waiting to resolve**, mostly curated markets that have closed. The first real run will resolve them.
2. **Refreshes data** by calling each source's own live CLI: arXiv, Hugging Face, GitHub counts and the stars snapshot, Epoch, and Polymarket. A failing source is logged, and the run carries on with what's stored; the exit status is 1.
3. **Creates questions** with Tasks 9–10's generators: numeric and threshold questions for the month, and curated questions whose ask date has come.
4. **Forecasts:**

   | Questions | Models |
   |---|---|
   | Numeric (7 series × 3 horizons) | `ai_random_walk`, `ai_linear_trend`, `ai_arima`, `ai_ets`, `ai_lightgbm` (pooled across benchmarks for percent series; stores SHAP values and its feature row) |
   | Threshold | `ai_base_rate` (`ThresholdBaseRate`), `ai_threshold_classifier` (Task 15, fit once per cutoff) |
   | Curated | `ai_market_price`, `ai_base_rate` (`CuratedBaseRate`) |
   | All, opt-in | `claude_llm` |

### Binary questions are forecast as of their ask dates
- **Ask dates:** a curated question is asked `lead_days` before its market closes; a threshold question on the day it was created.
- **The forecast is dated at the ask date,** with data through the day before, even if the run is late. A missed day catches up with the same cutoff.
- **A catch-up more than 3 days late** (`CATCHUP_DAYS`) is stored as a backtest, because it wasn't made in real time. With `--date` in the past, everything is a backtest.
- **A market that resolved before its ask date isn't forecast.**

### "Unavailable" isn't a failure
- **Some forecasts can't be made yet:**
  - ARIMA and ETS need 24 months, so SWE-bench (23) and ARC-AGI-2 (19) can't be fit until 2027.
  - A market may have no price on its ask date.
  - A series may have no data.
- **Counting those as failures would make every run exit 1.** They're counted as `unavailable` and logged at info level.
- **Anything else that raises is a failure:** logged with its traceback, rolled back, and the exit status is 1. A test shows one broken model doesn't stop the others.

### Dry run
- **`--dry-run` runs resolution, question creation and every forecast in one transaction,** using savepoints so a failed forecast doesn't abort the rest, then rolls it all back. It skips the data refresh and the LLM.
- **It's what found the two bugs below.**

### Bugs found by running against real data
1. **NumPy floats can't be stored.**
   - The running-max baselines (random walk and trend on the benchmark and compute series) returned `np.float64`. psycopg2 wrote that as the literal `np.float64(97.5)`, which failed with `schema "np" does not exist`: 17 failed forecasts per run.
   - The synthetic tests used only arXiv, a count series, which takes another code path.
   - **Fixed at both ends:** `ai/baselines._bounded` returns Python floats, and the job converts every forecast (`plain`) before storing it. There's a regression test with a percent series.
   - Task 18's context builder had hit the same leak (fixed then); this is its database form.
2. **Time zone of the ask date.**
   - A threshold question is asked "on the day it was created", read as `created_at::date` in the database's time zone (UTC). The job's clock is US Eastern.
   - In the evening, a question created "today" is tomorrow in UTC, so the job thought it wasn't asked yet and skipped it. An evening scheduled run would silently skip every new threshold question.
   - **Fixed:** creation dates are read in Eastern time, both in the job and in `ai/baselines.binary_questions` (Task 13), which had the same expression.

### The resolution job
- **It wraps `resolve_and_score_ai`** with the AI lock, `--date`, `--dry-run` and a scorecard (`ai_score_summary`):
  - **Numeric**, by series, horizon and model: n, RMSE, relative RMSE (comparable across series), bias and interval coverage.
  - **Binary**, by kind (threshold or curated) and model: n, Brier and a calibration gap (the count-weighted |predicted − observed| over 10 bins).
- **Binary probabilities are read from `forecasts.probabilities`.** I first recovered them from the Brier score and replaced that.
- **The forecast job already resolves before forecasting,** so this job is for checking the scorecard or running resolution on its own schedule.

### CLI
- **`eco-forecast ai-forecast` and `eco-forecast ai-resolve`** pass every option through to the jobs, like `eco-forecast backtest`.
- **Task 23's `--domain ai`** will build on this.

---

## What I did
- **`src/eco_prediction/ai/jobs/ai_forecast_job.py` (new):**
  - `run_ai_forecast_job` and `due_binary_questions`.
  - `refresh_ai_data`, `month_forecast_date`, the lock helpers, and the CLI.
- **`src/eco_prediction/ai/jobs/ai_resolution_job.py` (new):** `scored_ai_forecasts`, `ai_score_summary`, `calibration_gap`, and the CLI.
- **`src/eco_prediction/cli.py`:** the `ai-forecast` and `ai-resolve` subcommands.
- **`src/eco_prediction/ai/baselines.py`:** plain floats from `_bounded`, and Eastern creation dates in `binary_questions`.
- **`tests/test_ai_forecast_job.py`: 13 tests, not tagged `economy`.** They run on a migrated database with arXiv history and two markets.
  - **A full run:**
    - The questions created, and every model's forecasts.
    - Unavailable counted apart from failures.
    - LightGBM's SHAP values and feature snapshot.
  - **Binary dates:** a live forecast at yesterday's ask date; a 14-day catch-up stored as a backtest.
  - **Idempotence, and the resolved-before-ask rule.**
  - **A broken model** is a failure and doesn't stop the run.
  - **The LLM:**
    - Opt-in, capped at its budget: three calls, then off.
    - Forecasts and calls are linked.
    - The free models still forecast everything.
    - The next run fills in the rest.
    - The CLI refuses `--llm` without a budget, or with `--dry-run`.
  - **Locks:** the AI lock blocks a run; the economy lock doesn't.
  - **A dry run changes nothing,** and `--once-per-month` reuses the month's date.
  - **Resolution:**
    - The dry run is rolled back.
    - A real run resolves the numeric question and the resolved market, but not the open one.
    - The scorecard: Brier 0.09 for a 0.3 forecast of a "no", a calibration gap of 0.3, and live-only by default.
  - **CLI passthrough,** and the NumPy-float regression.

## Checks
- `uv run pytest`: **661 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **Dry runs on the live database** (no ingest, no LLM, about 1 minute):
  - **The first** found the two bugs above (17 failures).
  - **The second:** 64 questions created, 541 forecasts, 14 unavailable (all expected: ARIMA and ETS on short series, two markets with no price yet), 0 failed.
  - **Before and after both,** the live counts were the same: 2,461 AI questions, 22 forecasts, 0 resolutions, 0 LLM calls.

## Not done
- **The first real run** writes about 2,276 resolutions, 64 questions and 541 forecasts to the live database. Run it by hand first (`--no-ingest` is fine, the backfills are fresh), then install the launchd agent (`deploy/launchd/com.eco-prediction.ai-forecast-job.plist`).
- **LLM runs:** deferred with the budget.

## Where I changed the task's plan
- **The LLM is off unless asked for,** with a budget cap, rather than Haiku by default.
- **Binary forecasts are dated at their ask dates,** not the run date.
- **"Unavailable" is counted apart from failures.**
- **Both jobs have `--dry-run`;** the resolution job also has `--date`.
- **The resolution job shares the AI forecast job's lock,** not the economy's.

## Tradeoffs
- **Ingest runs each source's CLI in turn:** about 7 minutes, mostly Hugging Face and Polymarket. They're sequential and simple; parallelizing would share rate limits anyway.
- **LightGBM refits every run** (21 fits plus the classifier, under a minute). That's fine monthly; `--once-per-month` keeps the daily scheduled runs from refitting numeric models.
- **The first real run's catch-up:** every curated question whose ask date has passed but whose market is open gets a forecast, most of them as backtests (asked more than 3 days ago). That's useful track record, and only the opt-in LLM would cost anything.
- **Two locks mean the AI and economy jobs can write at once.** They touch different questions, and each forecast is its own small transaction, so they don't conflict.
