# Task 15: statistical models (AutoARIMA, AutoETS)

Done on 2026-10-08.

## Quick reference

```python
from eco_prediction.models.statistical import ARIMAModel, ETSModel

bt.run(ARIMAModel(), "cpi_yoy", horizon_months=3)
bt.run(ETSModel(season_length=1), "unemployment", horizon_months=6)
ARIMAModel(coverage=0.95)          # 95% interval instead of the default 80%
```

```sh
uv run pytest tests/test_statistical.py
```

---

## What I did

### `src/eco_prediction/models/statistical.py`
- **`StatsForecastModel`** is the shared base. It implements the harness's `Forecaster` protocol (`fit(data)` / `predict(data, target_period)`), like the baselines. Subclasses only provide `build()`, which returns an unfitted statsforecast model.
- **`ARIMAModel`** wraps `AutoARIMA` (orders chosen by stepwise AICc search at each fit).
- **`ETSModel`** wraps `AutoETS` (error, trend and season components chosen by AICc at each fit).
- **`monthly_history(data, start)`** returns the target on a complete monthly index, with interior gaps filled by linear interpolation.

### Where I changed the task's code
- **The harness interface, not `fit(train_df)` / `predict(h)`.** The task's sketch used `StatsForecast(...)` on a `unique_id/ds/y` frame. I used the model objects directly on a numpy array, which avoids the frame plumbing for a single series and fits the `Forecaster` protocol, so these models run in `WalkForwardBacktest` unchanged.
- **Predict starts from the latest known month, without a refit.** `fit` selects and estimates the model. `predict` keeps those parameters but runs the model over the target history known at predict time (statsforecast's `forward`). Between quarterly retrains, a forecast therefore starts from the latest published month and uses revised values, rather than forecasting further and further out from the training cutoff. The history given to `forward` starts where training did, so a rolling window lines up with the fit.
- **The step count comes from the data.** Steps = months from the last known period to `target_period`, so a delayed release just means one more step.
- **Intervals** are the model's own central `coverage` interval (80% by default), the same convention as the baselines.
- **Gaps are interpolated.** ARIMA and ETS need an unbroken series. October 2025 CPI, never published, is the real case.
- **Needs 24 months of history** to fit; fewer raises `ValueError`.

### Tradeoffs
- **`forward` vs. refitting every month.** `forward` is much cheaper than refitting and still uses the newest data, but the parameters (orders, smoothing weights) stay at the last fit's. A monthly retrain frequency gets you full refits if you want them.
- **ETS reacts slowly to level shifts.** With smoothing weights fit on smooth data, ETS moved only about 0.7 of a +5 level shift in the test series. That's the model working as designed, so the test only checks that newer data moves the forecast.
- **Linear interpolation of gaps** invents a value. It's only used for interior gaps (never the latest month) and the alternative, dropping the model, is worse.
- **Seasonality default is 12**, which assumes monthly seasonality. CPI YoY and seasonally adjusted unemployment have little left, so `season_length=1` may be the better choice on real data; worth comparing in the backtest.
- **mypy:** statsforecast has no type stubs, so the import has `# type: ignore[import-untyped]`, matching how the repo handles other untyped imports.

---

## Checks
- `uv run pytest`: 200 passed (20 new in `tests/test_statistical.py`, each run for both models). They cover:
  - recovering a trend plus alternating up/down pattern within 0.05, with the forecast inside its interval
  - predict starting from the latest known month: a +5 shift after training moves the forecast, which a forecast from the training cutoff would not
  - intervals wider at higher coverage and at longer horizons
  - a missing month interpolated, and both models fitting across it
  - errors for under 24 months, a target not after the latest data, predict before fit, and coverage outside 0–1
  - a full walk-forward run: 24 complete rows, errors under 1.0
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
- **Not done:** the task's test plan asked for a fit on 10 years of real CPI. The tests use synthetic data so they don't need the database; the models haven't been backtested on real data yet.
