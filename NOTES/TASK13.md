# Task 13: scoring metrics

Done on 2026-10-05.

## Quick reference

```python
from eco_prediction.backtest import metrics

metrics.rmse(pred, actual); metrics.mae(...); metrics.bias(...)
metrics.interval_coverage(zip(lower, upper), actual)
metrics.brier_score({"cut": 0.2, "hold": 0.7, "hike": 0.1}, "hold")   # 0.14
metrics.mean_brier_score(prob_dicts, outcomes)

probs, hits = metrics.one_vs_rest(prob_dicts, outcomes)
curve = metrics.calibration_curve(probs, hits, n_bins=10)
ax = metrics.plot_calibration_curve(curve, label="model A")
metrics.plot_calibration_curve(other_curve, ax=ax, label="model B")
```

```sh
uv run pytest tests/test_metrics.py
```

---

## What I did

### `src/eco_prediction/backtest/metrics.py`
- **Point:** `rmse`, `mae`, `bias` (prediction − actual, the same sign as the harness's `error` column).
- **Intervals:** `interval_coverage`. Both bounds count as inside, the same as the harness's `in_interval`.
- **Categorical:** `brier_score` for one forecast, `mean_brier_score` for a set of forecasts.
- **Calibration:** `calibration_curve` and `plot_calibration_curve`, plus `one_vs_rest`, which turns cut/hold/hike forecasts into binary (probability, happened) pairs so they can be plotted.
- **`summarize` in `harness.py` now calls these functions** instead of computing the numbers itself. Its output hasn't changed.

### Where I changed the task's code
- **Brier is the multi-category form.** It sums the squared error over every category, so it runs from 0 to 2, not 0 to 1. This is the formula in the task, and the value Task 10's `forecasts.brier_score` column will store. For a yes/no event, pass `{"yes": p, "no": 1 - p}`. That gives twice the usual binary Brier.
- **Inputs are checked:**
  - Brier needs probabilities in [0, 1] that sum to 1 (within 1e-6), and an outcome that is one of the categories.
  - The metrics refuse empty input, mismatched lengths, NaN, and an interval with lower > upper.
  - They don't skip unresolved rows silently. Filter those out first, the way `summarize` does.
- **`calibration_curve` is written with numpy and pandas instead of wrapping sklearn's.** It returns a DataFrame with `bin_lower`, `bin_upper`, `mean_predicted`, `observed_frequency` and `count`. sklearn's version returns two arrays without bin counts, and you need counts to see how much each point means. Empty bins are left out, as in sklearn.
- **Plot:** marker size shows the bin count, a dashed y = x line is drawn, and you pass `ax` back in to put several models on one chart. `matplotlib` is imported inside the function, so importing `metrics` doesn't load it.
- **New dependency:** `matplotlib` (added with `uv add`, so `pyproject.toml` and `uv.lock` changed).

---

## Checks
- `uv run pytest`: 158 passed (23 new in `tests/test_metrics.py`). They cover:
  - hand-computed RMSE, MAE, bias, coverage and Brier
  - perfect forecasts scoring 0, and Brier's bounds of 0 and 2
  - 80% normal intervals on 10,000 draws covering 80% ± 1.5%
  - calibrated synthetic forecasts lying on y = x within 0.02 in every bin, and overconfident ones falling below it
  - bin edges, with 1.0 going in the last bin and empty bins left out
  - `one_vs_rest` flattening
  - overlaying two models on one plot
  - each kind of bad input being refused
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
