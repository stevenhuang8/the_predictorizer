# Task 23: resolving questions and scoring forecasts

Done on 2026-10-08.

## Quick reference

```sh
uv run python -m eco_prediction.scheduler.resolution_job                     # refresh, resolve, score, print scores
uv run python -m eco_prediction.scheduler.resolution_job --no-ingest
uv run python -m eco_prediction.scheduler.resolution_job --include-backtest  # backtest rows too
uv run python -m eco_prediction.scheduler.forecast_job --no-resolve          # forecast without resolving
uv run pytest tests/test_resolution_job.py
```

**No cron change needed.** The daily forecast job now resolves and scores right after its data refresh, before forecasting.

```python
from eco_prediction.scheduler.resolution_job import resolve_and_score, score_summary
from eco_prediction.db.resolutions import scored_forecasts

resolve_and_score(conn, store, as_of=date(2026, 12, 15))   # ResolutionResult(resolved=[...], still_open=7, scored=12)
score_summary(scored_forecasts(conn))                      # live scores by target / horizon / model
```

---

## What I did

### `src/eco_prediction/scheduler/resolution_job.py`
- **`resolve_and_score(conn, store, as_of)`** works only from data published by `as_of` (`store.until(as_of)`). It:
  1. resolves every open question whose answer is out, committing each resolution as it goes;
  2. scores every unscored forecast of a resolved question, live and backtest alike. This includes forecasts added after their question resolved, such as a later backtest.
- **How questions resolve:**
  - **CPI YoY and unemployment:** the target month's **first release**, using `harness.actual_value`. That function is now shared with the backtest harness, so live and backtest scoring can't disagree. CPI YoY uses the year-earlier value as known at that release. A later revision doesn't change a resolution. `actual_as_of` records the vintage used, which Task 20's "bad data" check will need.
  - **FOMC:** the decision (`data.fomc.decision`), once the target for the day after the meeting is published. `actual_as_of` is that publication date.
- **Scores:**
  - Numeric forecasts: `error` (prediction − actual) and `in_interval` (bounds included).
  - FOMC forecasts: `brier_score`, the multi-category version (0–2) used everywhere else.
  - `scored_at` is stamped on each.
  - These are the columns Task 10 created for this.
- **`score_summary(scored_forecasts(conn))`** produces one row per (target, horizon, model, live/backtest): n, RMSE, MAE, bias, coverage, interval score, Brier.
  - The horizon reads "3 months" for monthly questions and "7 days" for pre-meeting FOMC questions. **Week-ahead and months-ahead FOMC accuracy are separate rows**, as agreed in Task 22.
  - Interval scores use each model version's stored `coverage` parameter, 0.8 if it has none.
  - Live only by default; `--include-backtest` adds backtest rows.
- **Safe to rerun:** resolved questions and scored forecasts are skipped. The command uses the same advisory lock as the forecast job, so a run by hand can't overlap the cron run.

### Other changes
- **`src/eco_prediction/db/resolutions.py`** (new): `unresolved_questions`, `save_resolution`, `unscored_forecasts`, `save_score`, `scored_forecasts`.
- **`backtest/harness.py`:** `WalkForwardBacktest.get_actual_value` now calls a module-level `actual_value(store, target, period, resolve_with)`. The behaviour is the same, and the 25 harness tests pass unchanged.
- **`scheduler/forecast_job.py`:** resolves and scores before forecasting, with a `--no-resolve` flag to skip it.

### Where I changed the task's plan
- **No separate cron entry.** The task wanted the resolution job run daily on its own, and the forecast job already runs daily with a data refresh. Folding resolution into it means one cron line, one data refresh, and no two jobs competing for the lock. The standalone command is for runs by hand.
- **Resolution by first release, not the latest data.** "Fetch actual value from database" could mean the latest vintage. Backtests are scored on first releases (Task 11), and live scores have to be comparable. Revisions can be compared later using `actual_as_of`.
- **No "resolution date" column.** The task's sketch looks for questions "past resolution date". A question resolves when its answer appears in the data, which is checked daily. That copes with delayed releases (e.g. a government shutdown) without hard-coding release calendars.

---

## Real data
- **Your database:** all 8 live questions are still open (the earliest, November 2026 CPI, is first released in mid-December), so nothing is scored yet. Both jobs ran cleanly against it.
- **Smoke test in a throwaway database** (created, then dropped): backtest forecasts for 2026-06-01, made with the random walk and persistence from the real data, plus a 7-day pre-meeting forecast for 2026-07-29, all resolved as of today.
  - **6 resolved:**
    - July CPI YoY 3.30, first released 2026-08-12;
    - July unemployment 4.1 (2026-08-07) and September unemployment 4.2 (2026-10-02);
    - the July meeting, hold, as both its monthly and 7-day questions;
    - the September meeting, hike.
  - **4 still open:** September CPI isn't out yet, and the December questions are still in the future.
  - **The scorecard** listed the FOMC "1 months", "3 months" and "7 days" questions as separate rows. Persistence scored 1.47 on the September hike it didn't expect.

## Tradeoffs and open questions
- **A question that's never answered stays open forever.** For example, October 2025 CPI was never published. That's harmless: it is skipped each day for the cost of one lookup. A cutoff such as "give up after 6 months" could be added if it ever matters.
- **Scores are stored per forecast, not aggregated.** `score_summary` recomputes the scorecard from the forecasts each time, which is cheap at this scale and can't go stale. Task 24's track-record page can call it, or query `forecasts` directly.
- **No significance tests in the live scorecard.** With a handful of resolved questions per month, Diebold–Mariano p-values would mean little for a long time. Task 19's report has them for backtests.

---

## Checks
- `uv run pytest`: 285 passed (7 new in `tests/test_resolution_job.py`, on a migrated throwaway database with a hand-built store). They cover:
  - scoring a single forecast: error, interval, missing interval, Brier
  - a numeric question resolving to its first release (3.6, not the later 3.5 revision), with `actual_as_of`; good and bad forecasts scored
  - CPI YoY of 5.0% (105/100, not the revised 101)
  - answers published after the run date ignored, and questions resolving on the day the data appears
  - FOMC: a hike resolved for both the monthly and 7-day questions (Brier 0.26 each), June left open until its decision is published, then resolved as hold
  - reruns changing nothing, and a backtest forecast added after resolution scored on the next run
  - the scorecard: RMSE, bias, coverage and interval score computed by hand, FOMC rows split by horizon, backtests excluded unless asked for
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
