# Task 16: LightGBM model with hyperparameter tuning

Done on 2026-10-08.

## Quick reference

```python
from eco_prediction.models.lightgbm_model import LightGBMModel

model = LightGBMModel(horizon_months=3)                 # must match the harness horizon
bt.run(model, "cpi_yoy", horizon_months=3)

model.best_params            # {'max_depth': 3, 'learning_rate': 0.05, 'min_child_samples': 20, 'n_estimators': 77}
model.cv_rmse                # mean validation RMSE of the winning grid point
model.feature_importance()   # total split gain per feature, largest first

LightGBMModel(3, predict_change=False)                  # predict levels instead of changes
LightGBMModel(3, param_grid={"max_depth": [3], "learning_rate": [0.1], "min_child_samples": [10]})
```

```sh
uv run pytest tests/test_lightgbm_model.py
```

---

## What I did

### `src/eco_prediction/models/lightgbm_model.py`
`LightGBMModel` implements the harness's `Forecaster` protocol, like the baselines and the Task 15 models.

- **Training rows.** One row per published target period. The features are the Task 14 `FeatureEngineer` vector as known at a month end; the label is the target `horizon_months` after the following month starts. That is exactly the gap a harness forecast made the next day spans. `training_frame(data)` returns `(X, y)` so the alignment can be checked.
- **Labels** are the target as known at the training cutoff. Rows whose target period hasn't been published by then are left out.
- **Features** missing in more than `max_missing` (50%) of training rows are dropped with `drop_sparse`; other gaps stay NaN, which LightGBM handles natively.
- **Tuning.** A grid over `max_depth` [3, 5, 7], `learning_rate` [0.01, 0.05, 0.1] and `min_child_samples` [5, 10, 20], scored with `TimeSeriesSplit(n_splits=5, gap=horizon)`. Each candidate trains with early stopping (50 rounds, cap 500 trees) on its validation fold. The lowest mean validation RMSE wins; the final model is refit on every row with the mean best iteration count as `n_estimators`.
- **Intervals.** Empirical: the central `coverage` quantiles of the winning candidate's out-of-fold residuals, added to the point forecast. Same idea as the random walk's intervals.
- **`feature_importance()`** returns split gain per feature.

### Where I changed the task's code
- **The harness interface, not `fit(X, y)` / `predict(X)`.** The model builds its own matrix from the `PointInTimeData` view, so it runs in `WalkForwardBacktest` and every training row uses the vintages known on its own date (no lookahead).
- **Direct forecasting, one model per horizon.** `fit(data)` doesn't receive the horizon, so it's a constructor argument; `predict` raises if `target_period` isn't that many months out.
- **`n_estimators` is chosen by early stopping, not gridded.** The task's grid had `n_estimators` [100, 200, 500]. Early stopping finds the count per candidate for the cost of one fit, and the task asked for early stopping anyway.
- **The CV folds have a `horizon`-row gap.** Without it, a training label could be published after a validation row's as-of date, which leaks the future into tuning.
- **Predicts the change, not the level (`predict_change=True`, the default).** The label is the target minus the latest value known at the row's as-of date; `predict` adds the latest known value back. See the real-data results for why.

---

## Results on real data

Walk-forward 2018-01 to 2024-12, 3-month horizon, annual refits, `VintageStore.from_db(backfill=True)`, full default grid. RMSE, lower is better:

| Target | LightGBM, levels | LightGBM, changes (default) | Random walk |
|---|---|---|---|
| CPI YoY | 1.75 | 1.54 | **1.43** |
| Unemployment | 2.72 | 2.49 | **2.40** |

- **LightGBM doesn't beat the random walk yet**, on either target. The task's test plan asked for the comparison, not a win, so I marked it done; it's the obvious thing to come back to.
- **Why changes beat levels.** Trees can't predict outside the range of labels they were trained on. The window holds the 2020 unemployment spike and the 2021–22 inflation surge, both near or beyond anything in training, so the level model was capped. The change model only needs the size of moves to be familiar. It cut RMSE by about 10% and brought average bias to about zero (−0.08 and −0.01, from −0.22 and +0.95).
- **Intervals are too narrow:** 61–63% coverage against 80%. Out-of-fold residuals come from calmer years than the test window.
- **`trend_months` ranks among the top features** for both targets. A linear time trend lets trees learn "which era is this" rather than economic relationships. Dropping it is the first thing to try.
- **Runtime:** about 40–55 s per 7-year backtest with annual refits (27 grid points × 5 folds per refit). Monthly refits would take roughly 12 times as long; pass a smaller `param_grid` for those.

## Tradeoffs and open questions
- **Grid search vs. random or Bayesian search.** The grid is small (27 points) and exhaustive, which keeps results deterministic and easy to reason about. A bigger search space would want `RandomizedSearchCV` or Optuna.
- **Early stopping on the fold it's scored on** makes CV RMSE slightly optimistic. It's used only to rank candidates, so the bias matters less than it would for reporting.
- **One model per horizon** means fitting a model per horizon you backtest. The alternative, horizon as a feature in one model, shares data across horizons but complicates the training frame.
- **Labels use the target as known at the training cutoff** (revised values), while features use the vintage known on each row's date. That's what a forecaster could actually train on at the time, but it means labels can be revised between refits.
- **`n_jobs=1`.** The matrices are a few hundred rows; thread start-up cost more than it saved.
- **mypy:** sklearn has no type stubs, so `TimeSeriesSplit` is imported with `# type: ignore[import-untyped]`.

---

## Checks
- `uv run pytest`: 210 passed (10 new in `tests/test_lightgbm_model.py`). The tests use a synthetic store where MICH is white noise and unemployment is 5 + MICH from 5 months earlier, so there's a real signal a random walk can't use. They cover:
  - training rows lining up with harness forecasts (label = 5 + the latest `mich`), and labels stopping at the latest published period
  - change labels equal to the level minus the latest known unemployment
  - `mich` and `unrate` as the top two features, a sensible `best_params`, and the forecast within 0.5 of the true value inside its interval
  - all-NaN features (oil, claims) dropped, target lags kept
  - intervals wider at higher coverage
  - errors for a mismatched horizon, under 36 labelled months, use before fitting, and bad constructor arguments
  - a walk-forward run where LightGBM's RMSE is under half the random walk's
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
- One fixture bug found along the way: MICH published on the 10th of the next month means a target at m+4 needs a 5-month lead, not 4. The model's alignment was right; the test data was wrong.
