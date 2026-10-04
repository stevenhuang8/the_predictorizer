# Task 11: walk-forward backtesting harness

Done on 2026-10-04.

## Quick reference

```python
from datetime import date
from eco_prediction.backtest.data import PointInTimeData
from eco_prediction.backtest.harness import Forecast, WalkForwardBacktest, summarize

class LastValue:                                   # any object with fit + predict
    def fit(self, data: PointInTimeData) -> None: ...
    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        return Forecast(float(data.target_history().iloc[-1]))

bt = WalkForwardBacktest(date(2020, 1, 1), date(2023, 12, 1), "quarterly")
results = bt.run(LastValue(), "cpi_yoy", horizon_months=3)   # one row per forecast
summarize(results)        # n_forecasts, n_resolved, rmse, mae, bias, interval_coverage
```

```sh
uv run pytest tests/test_backtest_harness.py
```

---

## What I did

### `src/eco_prediction/backtest/data.py`: the only way models see data
- **`VintageStore`** loads every vintage of a series from the database the first time it's needed and keeps it in memory. A backtest then reads each series from the database once, not once per forecast date. You can also pass it data frames directly, which is what the tests do.
- **`PointInTimeData(store, target, cutoff, start=None)`** is a read-only view of the data as it stood at the end of `cutoff`:
  - `.series("T10Y2Y")` returns any series. `.target_history()` returns the forecast quantity, for example CPI year-over-year %.
  - For each period it returns the latest vintage published on or before the cutoff. Nothing later is reachable.
  - `max_as_of_seen` records the newest vintage it actually handed out.
  - `start` is the rolling-window limit. For CPI year-over-year, the change is computed before the limit is applied, so the first month in the window still has its year-earlier value.
- **`TARGETS`**:
  - `cpi_yoy` is CPIAUCSL, 12-month % change.
  - `unemployment` is UNRATE, the level.
  - The names match the `forecast_target` enum from Task 9.

### `src/eco_prediction/backtest/harness.py`
- **`WalkForwardBacktest(start, end, retrain_frequency, *, window, window_months, train_start, resolve_with, store)`**.
- **Forecast dates** are the first of each month in `[start, end]`.
- **No lookahead.**
  - A forecast made on day D sees data published by the end of D − 1. `as_of` is only a date, and a release on day D may come after the forecast is made (see the Task 7 note).
  - After every `fit` and every `predict`, the harness checks the view's `max_as_of_seen` against its cutoff and raises `LookaheadError` if it's later.
  - `max_as_of_seen` is also returned with each result row, so you can audit it.
- **The target period** is D's month plus the horizon, matching the task's `forecast_date + horizon_months`. On 2020-03-01 at horizon 3 that's June 2020. The model gets `target_period` and works out its own step count from the last period it knows: `months_between` and `add_months` are exported.
- **Retraining** happens on the first forecast date and then every month, quarter or year. Training uses either all history (`expanding`, optionally from `train_start`) or the last `window_months` (`rolling`).
- **Resolution.**
  - The default scores against the **first release**, the number the forecast would have been judged on at the time. `resolve_with="latest"` uses the newest vintage instead.
  - CPI year-over-year is resolved with the year-earlier value as known on the same release date, the way BLS computes the headline number.
  - A target that hasn't been published yet gets no actual and isn't scored.
- **Result columns:**
  - dates: `forecast_date`, `data_as_of`, `trained_as_of`, `max_as_of_seen`
  - the forecast: `target_period`, `horizon_months`, `prediction`, `lower`, `upper`
  - the outcome: `actual`, `actual_as_of`
  - the scores: `error` (prediction − actual), `abs_error`, `squared_error`, `in_interval`
- **`summarize(results)`** gives RMSE, MAE, bias and interval coverage over the resolved rows. Task 13 will add the full set of metrics.

### Choices and limits
- **The model interface is `fit(data)` / `predict(data, target_period) -> Forecast(point, lower, upper)`.** Models pull whatever series they need from the view, so the feature pipeline in Task 14 can take a `PointInTimeData` too. That way nothing outside the view can leak into a model. The task's sketch had `model.predict(features)`, with the harness building the features. That would make the harness depend on Task 14 and on each model's choice of features.
- **Training data comes from the latest vintages as known at fit time.** For example, a fit on 2020-03-31 sees the 2019 data as revised by March 2020, not as first printed. That's what a forecaster would actually have had then.
- **Numeric targets only.** FOMC decisions (cut/hold/hike) need a categorical forecast and a Brier score, which belong with Task 21. Adding a third target is one more `Target` entry plus a scorer.
- **Nothing is saved to `questions` or `forecasts` yet.** The `forecasts` table comes in Task 10, which still has to be done. For now, `run` returns a DataFrame.

---

## Checks
- `uv run pytest`: 88 passed (21 new). The new tests use made-up monthly series: each month is first printed on the 15th of the next month and revised by +0.5 a month after that. They check:
  - **No leakage:** models never see a period or a revision before its publish date, including one published on the forecast date itself.
  - **Training:** the retrain schedule and its cutoffs, plus the expanding, `train_start` and rolling windows.
  - **Resolution:** first release versus latest, CPI year-over-year using the year-earlier value as known at the same release, and targets not yet published.
  - **Scores:** hand-computed scores and summary metrics, and point-only forecasts.
  - **Setup:** invalid settings are rejected, and loading from a real test database works.
- `ruff check src tests` and `mypy src tests`: no problems.
- **Real data, 2020–2023, random walk (last known value), quarterly retrain.** 48 forecasts each; under 0.5 s per run.

  | Target | h=1 RMSE | h=3 RMSE | h=6 RMSE |
  |---|---|---|---|
  | cpi_yoy | 1.27 | 1.85 | 2.60 |
  | unemployment | 2.67 | 3.17 | 2.52 |

  In every row `max_as_of_seen` < `forecast_date`. The 2020-01-01 forecast saw data up to 2019-12-11, the November CPI release. The unemployment errors are mostly the April 2020 spike.
