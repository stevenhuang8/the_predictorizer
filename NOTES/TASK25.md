# Task 25: `eco-forecast` CLI

Done on 2026-10-08.

## Quick reference

```sh
uv run eco-forecast forecast --target cpi --horizon 3          # every model, data through yesterday
uv run eco-forecast forecast --target unemployment --horizon 1 --model lightgbm
uv run eco-forecast forecast --target fomc                     # next scheduled meeting
uv run eco-forecast forecast --target fomc --meeting 2027-01-27
uv run eco-forecast forecast --target cpi --date 2025-06-01    # as of a past date (point-in-time)
uv run eco-forecast inspect                                    # list questions
uv run eco-forecast inspect --question-id 7                    # forecasts, scores, post-mortems
uv run eco-forecast scores [--include-backtest]                # scorecard
uv run eco-forecast backtest [--refresh] [--start 2018-01-01 ...]   # Task 19's report
uv run pytest tests/test_cli.py
```

---

## What I did

### `src/eco_prediction/cli.py`, installed as `eco-forecast` (`[project.scripts]`)
- **`forecast`** refits the models on the database's data (point-in-time, through the day before `--date`, default today) and prints each model's forecast.
  - **CPI and unemployment:** random walk, ARIMA and LightGBM (the scheduler's `DEFAULT_NUMERIC`). It shows the point forecast and 80% interval, LightGBM's top 5 SHAP drivers, and the latest value known.
  - **FOMC:** persistence and the FOMC LightGBM, for the next scheduled meeting or `--meeting`. It shows cut/hold/hike probabilities and the forecaster's top features. Dates that aren't decision days, and past meetings, are refused.
  - `--model` limits it to one model.
- **`inspect`:**
  - Without an id, it lists every question: target, date, horizon or lead, live and backtest forecast counts, and the actual value once resolved.
  - With `--question-id`, it shows the question's rule, its status (open, or resolved with the value and publication date), and every forecast with its score and post-mortem, plus post-mortem notes.
- **`scores`:** Task 23's scorecard (live only; `--include-backtest` adds backtests).
- **`backtest`:** runs Task 19's comparison and passes every option through, including `--help`.

### Where I changed the task's plan
- **argparse, not Click.** Every other entry point in the project (ingest, backfill, migrate, the jobs, the comparison) uses argparse, and Click isn't a dependency. Subcommands cover everything the sketch did.
- **`forecast` doesn't store anything.** The task's sketch didn't say. Storing manual forecasts would let a hand run add to the live track record outside the schedule, with whatever code happened to be on disk. The daily job stays the only writer of live forecasts. The CLI is for looking.
- **Targets are `cpi`, `unemployment` and `fomc`**, as in the task. They map to the database names (`cpi_yoy`, `fomc_decision`).
- **Added `scores`, plus `--date` and `--model` on `forecast`.** They're cheap and are what you'd reach for when inspecting.

---

## Real data (2026-10-08)
- **`forecast --target cpi --horizon 3`** (about 40 s, mostly ARIMA and LightGBM fitting) gave, for January 2027: random walk 3.35, ARIMA 3.75, LightGBM 3.54. These **match the live forecasts the daily job stored this morning** exactly, so the CLI and the job run the same pipeline. LightGBM's drivers: oil month-on-month +0.07, claims +0.06, PPI year-on-year +0.05.
- **`forecast --target fomc`** (October 28 meeting, 20 days out): persistence 59% hike, LightGBM 88% hike (top features: the decision before last, the T-bill's 1-month change, the 2-year spread).
- **`inspect`** lists the 8 live questions. `inspect --question-id 7` shows the three April 2027 unemployment forecasts, all still open.
- **`scores`** prints "No scored forecasts yet."
- **`backtest`** regenerated the Task 19 report from its cache.
- **Bad input:** an unknown question id, a non-decision date, or an unknown model each exit with a one-line error.

## Tradeoffs
- **Refitting on every `forecast`** takes about 40 s for CPI and less for FOMC. Loading the job's fitted models instead would need model artifacts on disk, which the project doesn't store (Task 22 refits monthly by design).
- **Plain-text tables** (pandas `to_string`), no colour or rich formatting. That keeps the output greppable and avoids a dependency.

---

## Checks
- `uv run pytest`: 312 passed (9 new in `tests/test_cli.py`). They cover:
  - parsing: defaults, an invalid target, a missing `--target`, unknown options refused except after `backtest`
  - `backtest` forwarding its options to the comparison script
  - `--horizon 0` refused
  - a numeric forecast on the synthetic MICH/unemployment store: header, cutoff, both models, SHAP drivers naming the signal
  - `--model` filtering, and an unknown model refused
  - FOMC defaulting to the next meeting (2019-07-31, 11 days out) with both models' probabilities, and refusing a non-decision date and a past meeting
  - on a throwaway database: an empty listing, then a question with live and backtest forecasts shown open, then resolved, scored and post-mortemed, and an unknown id refused
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
