# Task 22: live forecasting scheduler

Done on 2026-10-08.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate                                # applies 005 and 006
uv run python -m eco_prediction.scheduler.forecast_job                    # forecast as of today
uv run python -m eco_prediction.scheduler.forecast_job --once-per-month   # what cron runs
uv run python -m eco_prediction.scheduler.forecast_job --no-ingest        # skip the FRED refresh
uv run python -m eco_prediction.scheduler.forecast_job --date 2026-09-01  # stored as a backtest
uv run python -m eco_prediction.scheduler.forecast_job --no-fomc
uv run pytest tests/test_forecast_job.py
```

**Not installed yet.** To schedule it, add this with `crontab -e`:

```cron
0 9 * * * cd /Users/stevenhuang/dev/AI_Projects/eco_prediction && /opt/homebrew/bin/uv run python -m eco_prediction.scheduler.forecast_job --once-per-month >> logs/cron.log 2>&1
```

The job also logs to `logs/forecast_job.log`. It exits with 1 if a model or a series' ingest failed, 0 otherwise.

---

## What I did

### `src/eco_prediction/scheduler/forecast_job.py`
A run on date D:
1. **Takes a Postgres advisory lock.** If another run holds it, the job exits.
2. **Refreshes data:** an incremental ingest of every `V1_SERIES` (16 series). A failed series is logged and the run carries on with the data it has.
3. **Creates questions** for a forecast made in month M, matching the backtest harness:
   - CPI YoY and unemployment for months M+1, M+3 and M+6;
   - the FOMC decision at the scheduled meeting in each of those months, if there is one. About two-thirds of months have one.
4. **Runs every model on every question of its kind**, with data through D − 1 (`store.until(D − 1)`):
   - **CPI and unemployment:** random walk, ARIMA, LightGBM. LightGBM's forecasts store SHAP values (Task 17) and a feature snapshot of the exact row the model saw.
   - **FOMC:** Persistence and `FOMCForecaster`. Each model is trained for that meeting's lead time, the days from D to the meeting. The forecaster stores a feature snapshot.
   - Baselines run too, so every question has a reference forecast in the track record.
5. **Stores a model version per fitted model:** model type, a tag like `cpi_yoy-h3-20261007` or `fomc-lead111-20261007`, its parameters (including the tuned ones), `training_end`, and `code_hash` (`git describe --always --dirty`).

### Safe to rerun
- **A forecast that already exists is skipped before its model is fit:** same question, model type and tag, forecast date, and live/backtest flag. A rerun the same day took 5 s and inserted nothing.
- **Migration 005** adds a unique index on `forecasts (question_id, model_version_id, forecast_date, is_backtest)`, so a duplicate fails even if something bypasses the check. With 006 below, Task 20's post-mortem migration will be 007, not 005 as Task 10's notes said.
- **Each forecast commits on its own.** A model that throws is rolled back, logged and listed in `failures`; the others are kept, and the next run fills in only the missing ones.
- **`--once-per-month`:** if live forecasts already exist this month, the run reuses their forecast date and only fills gaps. Otherwise it forecasts as of today.

### `src/eco_prediction/db/forecasts.py`
- New functions: `get_or_create_question`, `get_or_create_model_version`, `save_feature_snapshot`, `forecast_exists`, `save_probability_forecast` (for FOMC).
- JSON values are cleaned before storing: NaN becomes null, numpy numbers become Python numbers, dates become ISO strings.

### Speed fixes found along the way
- **`data.fomc.value_at` uses a binary search** instead of filtering the whole daily series on every call. An FOMC fit went from 20 s to 4 s, which also speeds up Task 21's backtests.
- **The job's models share one `FeatureEngineer`,** so the three LightGBM horizons and the FOMC model reuse feature rows within a run. Its cache is keyed by store, and each run has its own cut-off store, so nothing carries over between runs. The job's tests went from 5 min to 70 s.

### Where I changed the task's plan
- **Cron runs daily with `--once-per-month`, not `0 9 1 * *`.** This is a laptop, and cron doesn't run jobs that were missed while the machine was asleep (launchd would, but it's more setup). A daily run that does nothing once the month's forecasts exist covers that. The forecast date is then the first day the machine was awake that month, and the data cutoff moves with it.
- **No business-day logic.** The backtest's forecasts are made on the 1st of the month with data through the last day of the previous month. Running on the calendar 1st matches that exactly, and FRED doesn't publish on weekends anyway.
- **"Today" is US Eastern time,** the clock FRED and the Fed publish on. This only matters for a run near midnight.
- **A run with a past `--date` is stored as `is_backtest = true`.** Its data is cut off correctly, but it wasn't made in real time, and the track record must not count it as live. A `--once-per-month` catch-up run is the exception: it reuses this month's date with data cut off at that date, so it stays live.
- **ARIMA instead of the task's model list:** I used random walk + ARIMA + LightGBM for CPI and unemployment. ETS and the historical mean are left out; each is one line in `DEFAULT_NUMERIC`.

### Follow-up: pre-meeting FOMC questions (migration 006)
Task 21 found LightGBM's edge over persistence comes almost entirely from the last week before a meeting, when the T-bill rate has priced the decision. The monthly run asks about meetings 1 to 7 months out (here 111 and 202 days). So the job now also asks about every meeting **7 days before its decision day**, as a separate question.

- **Separate questions, not revisions** (your call). Accuracy can then be scored per lead time. If a 7-day forecast revised the monthly one and only the latest forecast were scored, the record would show week-ahead skill under every meeting. If every revision were scored, week-ahead and months-ahead results would be averaged together.
- **Schema change (`006_question_lead_days.sql`):**
  - `questions.lead_days` is new, and `horizon_months` becomes nullable.
  - A check requires exactly one of the two.
  - Uniqueness is now `UNIQUE NULLS NOT DISTINCT (target, target_date, horizon_months, lead_days)`, so the same meeting can have a monthly question at h=1 and a 7-day question.
  - `get_or_create_question(..., lead_days=7)` creates them.
- **`run_pre_meeting_job(conn, store, meeting)`** forecasts the meeting as of 7 days before it (data cut off the day before that) with both FOMC models. The tag looks like `fomc-lead7-20230912`. The monthly run and this one share `forecast_questions`.
- **Scheduling:** the same daily cron run. `pre_meeting_due(today)` returns any meeting from 7 days before it until the day before the decision.
  - On the due day, the forecast is made then.
  - If the machine was asleep, a later run catches up, still dated and cut off 7 days before the meeting. This is the same policy as the monthly catch-up.
  - Once the meeting has happened, it's too late. Called directly for a past meeting, the run defaults to a backtest.
  - With `--date D`, only a meeting exactly 7 days after D is included.
- **`--once-per-month` ignores pre-meeting forecasts** when looking for this month's monthly run, so they can't make it reuse the wrong date.
- **Real database:** 006 applied; the existing 22 forecasts were all found again. The first live pre-meeting forecast is due **2026-10-21**, for the 2026-10-28 meeting.

---

## First live run (2026-10-08, data through 2026-10-07)

8 questions, 22 forecasts, 0 failures, 3.5 minutes including the ingest. Of that, LightGBM's grid search on CPI took about 50 s, and ARIMA on CPI about 20 s per horizon.

| Question | Random walk | ARIMA | LightGBM |
|---|---|---|---|
| CPI YoY, Nov 2026 | 3.35 | 3.72 | 3.62 |
| CPI YoY, Jan 2027 | 3.35 | 3.75 | 3.54 |
| CPI YoY, Apr 2027 | 3.35 | 2.69 | 3.09 |
| Unemployment, Nov 2026 | 4.20 | 4.20 | 4.09 |
| Unemployment, Jan 2027 | 4.20 | 4.20 | 4.07 |
| Unemployment, Apr 2027 | 4.20 | 4.20 | 4.12 |

| FOMC meeting | Persistence (cut / hold / hike) | LightGBM |
|---|---|---|
| 2027-01-27 (111 days) | 0.02 / 0.40 / 0.59 | 0.04 / 0.37 / 0.59 |
| 2027-04-28 (202 days) | 0.02 / 0.40 / 0.59 | 0.08 / 0.62 / 0.29 |

- **Persistence says hike for both meetings** because the last decision (2026-09-16) was a hike. Task 21 showed LightGBM adds little over persistence beyond a few weeks out, so the January numbers agreeing is expected.
- **These are real live forecasts in the `eco_forecast` database.** To remove them: `DELETE FROM questions WHERE created_at::date = '2026-10-08';` (forecasts cascade), then delete the orphaned `model_versions` and `feature_snapshots`.

## Tradeoffs and open questions
- **Every model is refit every month.** That costs about 3.5 minutes, which is fine monthly. It is also the simplest way to keep `model_versions` honest: one row per fitted model, with the data cutoff in its tag. The alternative, saving models to disk between runs, adds artifact storage for little gain.
- **Ingest failures don't stop the run.** The forecasts use whatever data is in the database, and the exit code is 1 so it shows up in the cron log. Stopping instead would mean a FRED outage on the 1st costs that month's forecasts.
- **Data cutoff vs. the moment of the run.** A forecast made on the 8th at 18:30 uses data through the 7th, never same-day releases. That matches the backtest, at the cost of ignoring a CPI release on the morning of the forecast.
- **Untyped shap import warnings** (3 `PendingDeprecationWarning`s) appear in the job log via stderr. They're harmless.

---

## Checks
- `uv run pytest`: 268 passed (16 new in `tests/test_forecast_job.py`, on a migrated throwaway database). They cover:
  - question planning: dates for each horizon, FOMC questions only in months with a meeting (checked against the real October 2026 calendar), and mid-month dates using the calendar month
  - a full run: question types and resolution rules, one forecast per model per question, forecast date, live flag, model-version tag, cutoff and code hash, interval or probabilities as appropriate, SHAP and feature snapshots only for the LightGBM models, and FOMC lead times in the tags
  - the stored SHAP explanation adding up to the stored LightGBM prediction
  - a rerun the same day inserting nothing and building no models
  - the next month adding new questions with no duplicate (question, model, date)
  - a failing model: no half-written rows; the other models stored; a rerun after the fix filling only the gap
  - past dates defaulting to backtest without colliding with live rows
  - the unique index rejecting a direct duplicate insert
  - finding this month's live run date, ignoring backtests
  - the advisory lock blocking a second connection
  - pre-meeting: when a forecast is due (from 7 days before until the day before the decision); a 7-day question beside the monthly h=1 question for the same meeting, with its own dates and tag; a rerun adding nothing; not counted as the monthly run; past meetings defaulting to backtest
  - exactly one of horizon/lead, enforced in code and by the check constraint; lead questions unique and reused
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
- **Manual run on the real database:** migration 005 applied; the live run and the rerun are described above.
