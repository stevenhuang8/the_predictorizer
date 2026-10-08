# Task 17: SHAP explainability, stored with forecasts

Done on 2026-10-08.

## Quick reference

```python
from eco_prediction.models.explainability import SHAPExplainer
from eco_prediction.db.forecasts import save_forecast, get_explanation

model.fit(train)
explainer = SHAPExplainer(model)                 # once per fit
explanation = explainer.explain(view, target_period)
explanation.top(5)        # [('claims_4wk_yoy', 0.563), ('ppi_3m', 0.366), ...]
explanation.prediction    # offset + expected_value + sum(contributions) == model.predict(...).point

forecast_id = save_forecast(conn, question_id, model_version_id, forecast_date,
                            model.predict(view, target_period), explanation=explanation)
conn.commit()
get_explanation(conn, forecast_id)               # Explanation(...), or None if none was stored
```

```sh
uv run pytest tests/test_explainability.py
```

---

## What I did

### `src/eco_prediction/models/explainability.py`
- **`SHAPExplainer(model)`** wraps `shap.TreeExplainer` around a fitted `LightGBMModel`. TreeSHAP is exact for tree models, so the contributions add up to the model's output with no sampling error.
- **`explain(data, target_period)`** explains exactly the forecast `model.predict(data, target_period)` makes: same feature row, same offset.
- **`Explanation`** is a frozen dataclass:
  - `expected_value`: the model's mean output over its training rows
  - `offset`: the latest known target value when the model predicts changes (the default since Task 16); 0 for a level model
  - `contributions`: one value per feature the model was trained on, in the target's units (percentage points)
  - `prediction`, `top(n)` (largest by absolute size, signed), `to_json()` / `from_json()`

### `src/eco_prediction/db/forecasts.py` (new)
- **`save_forecast(conn, question_id, model_version_id, forecast_date, forecast, *, explanation, feature_snapshot_id, is_backtest, rationale)`** inserts a numeric forecast and returns its id. The explanation goes in `forecasts.shap_values` as `{"expected_value", "offset", "contributions": {...}}`.
- **`get_explanation(conn, forecast_id)`** reads it back as an `Explanation`, `None` if the forecast has none, and `KeyError` if there's no such forecast.
- Like the rest of `db`, they don't commit.

### `src/eco_prediction/models/lightgbm_model.py`
- Split **`prediction_inputs(data, target_period)`** out of `predict`. It returns the feature row and the offset, and both `predict` and the explainer use it, so an explanation can't drift from the forecast it explains.

### Where I changed the task's code
- **No "forecasting pipeline" existed to modify.** Nothing wrote to `forecasts` yet; the backtest harness keeps its results in a DataFrame. I added `save_forecast` / `get_explanation` so that whatever writes forecasts later (live forecasts, or persisting backtests) stores explanations with them in one call.
- **The stored JSON includes `offset`.** The model predicts changes, so SHAP explains the change. Without the latest value the contributions wouldn't add up to the stored prediction. The task's test plan, "SHAP values sum to prediction − base_value", holds with base = `offset + expected_value`.
- **One forecast at a time.** The task's sketch also handled multi-row inputs with per-feature lists. A forecast row is always one row, and a flat `{feature: value}` dict is what's useful in JSONB (`shap_values->'contributions'->>'mich'` is queryable).
- **The explainer refuses a refit model.** `TreeExplainer` holds the booster it was built on; after `fit` runs again, `explain` raises instead of silently explaining the old trees with the new model's inputs.

---

## Tradeoffs
- **shap library vs. LightGBM's built-in `pred_contrib=True`.** LightGBM computes the same TreeSHAP values natively, without the dependency. I used `shap` as the task specified, and because it gives a path to its plots and to non-tree explainers later. Switching would be a few lines if import time or the dependency becomes a problem.
- **Contributions for NaN features.** A feature missing in the forecast row still gets a contribution: LightGBM sends missing values down a learned branch, and SHAP attributes the effect of that. Not a bug, but it can look odd.
- **Correlated features split credit.** `claims`, `claims_4wk` and `claims_4wk_yoy` move together, and TreeSHAP divides their effect between them somewhat arbitrarily. Summing contributions by group (claims, prices, the target's own lags) would read more reliably than single features.
- **Every feature is stored** (about 40 floats per forecast), not just the top few. Small, and it keeps the sum exact.

## Real-data explanations
Fit on everything known at the cutoff (`VintageStore.from_db(backfill=True)`, full default grid), 3-month horizon:

| Forecast | Prediction = latest + expected + features | Top contributions |
|---|---|---|
| CPI YoY, made 2022-07-01 for Oct 2022 | 9.90 = 8.52 + 0.04 + 1.35 | `claims_4wk_yoy` +0.56, `ppi_3m` +0.37, `claims` +0.16, `trend_months` +0.16, `ppi_yoy` +0.15 |
| Unemployment, made 2020-04-01 for Jul 2020 | 4.29 = 3.50 − 0.01 + 0.80 | `claims_4wk_yoy` +0.58, `trend_months` +0.19, the rest under 0.02 |

- **Unemployment, COVID:** exploding jobless claims are almost the whole push upward, which is right. The model still forecast 4.3% against an actual of about 10.8% for June; no tree trained on earlier data has seen a move that large.
- **CPI, 2022:** PPI momentum pushing inflation up makes sense. Falling claims year-over-year (pandemic base effects) being the largest term is plausible as "tight labor market", but probably partly spurious.
- **`trend_months` appears again**, as in Task 16. More evidence it should be dropped.

---

## Checks
- `uv run pytest`: 218 passed (8 new in `tests/test_explainability.py`; the 2 database tests ran against Postgres). They cover:
  - contributions adding up to `model.predict` within 1e-6, for both change and level models, with one contribution per trained feature and the right offset
  - `mich` and `unrate` as the top two contributions on the synthetic leading-indicator data, and `top()` sorted by size
  - JSON round trip
  - errors for an unfitted model and for explaining after a refit
  - saving a forecast with its explanation and reading it back, prediction, interval and backtest flag stored, and a JSONB path query on one contribution
  - a forecast with no explanation returning `None`, and an unknown id raising `KeyError`
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems. shap has no type stubs, so it's imported with `# type: ignore[import-untyped]`.
- shap emits 3 `PendingDeprecationWarning`s from its own plotting module on import; harmless.
