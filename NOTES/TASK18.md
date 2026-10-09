# Task 18: calibrating intervals and probabilities

Done on 2026-10-08.

## Quick reference

```python
from eco_prediction.models.calibration import (
    IntervalCalibrator, ProbabilityCalibrator, calibrate_intervals,
)

# Backtest: conformal intervals refit at every model refit, on errors published by then
results = bt.run(model, "cpi_yoy", horizon_months=3)
calibrated = calibrate_intervals(results, coverage=0.8, max_residuals=36)
summarize(calibrated)["interval_coverage"]
calibrated[["lower", "upper", "raw_lower", "raw_upper", "n_calibration"]]

# By hand
cal = IntervalCalibrator(0.8).fit(actuals - predictions)    # or method="signed"
cal.predict_interval(2.4); cal.apply(forecast)              # -> Forecast with new bounds

pc = ProbabilityCalibrator("isotonic")                      # or "sigmoid" (Platt)
pc.fit(probs, outcomes); pc.calibrate(new_probs)
pc.fit_categorical(prob_dicts, outcomes)
pc.calibrate_categorical({"cut": 0.2, "hold": 0.7, "hike": 0.1})   # sums to 1
```

```sh
uv run pytest tests/test_calibration.py
```

---

## What I did

### `src/eco_prediction/models/calibration.py`
- **`IntervalCalibrator(coverage, method)`**: split conformal prediction.
  - `"absolute"` (default, the task's version): half-width = a quantile of |actual − prediction|, so the interval is symmetric.
  - `"signed"`: lower and upper quantiles of the signed errors, so it also corrects a biased point forecast.
  - `fit(residuals)`, `predict_interval(point)`, `apply(forecast)`.
- **`ProbabilityCalibrator(method)`**:
  - `"isotonic"` (default) or `"sigmoid"` (Platt scaling: a logistic regression on the log-odds).
  - Binary: `fit` / `calibrate`.
  - Categorical: `fit_categorical` / `calibrate_categorical`, which take the same `{"cut": ..., "hold": ..., "hike": ...}` dicts as `metrics.brier_score`.
- **`calibrate_intervals(results, coverage, method, max_residuals)`**: the backtest integration. It takes a `WalkForwardBacktest.run` frame and returns a copy:
  - **Fit per fold.** For each fold (rows sharing `trained_as_of`), it fits on every earlier forecast whose actual was published by that fold's cutoff (`actual_as_of <= trained_as_of`). This is "fit on fold N, apply to fold N+1" with no lookahead.
  - **New intervals.** `lower`, `upper` and `in_interval` are replaced, so `summarize` scores the calibrated intervals.
  - **Extra columns.** The model's own bounds stay in `raw_lower` / `raw_upper`, and `n_calibration` records how many errors each row was fit on.
  - **Folds with too few errors** (including the first fold) get NaN bounds, so coverage is scored only on the rows that were calibrated.
  - **`max_residuals`** keeps only the most recent N errors (a rolling window).

### Where I changed the task's code
- **Finite-sample conformal quantile.** I used the `ceil((n+1)·coverage)`-th smallest error, not `np.quantile(..., coverage)`. That is the version that guarantees coverage of at least the target. With few residuals (a backtest fold may have 8), the plain quantile under-covers. `fit` raises if there are too few residuals for a finite interval (`min_residuals`: 4 for 80% absolute, 9 for 80% signed).
- **Sigmoid (Platt) is implemented.** The task's sketch set the calibrator to `None` for anything other than isotonic.
- **Categorical calibration pools all categories into one calibrator, then renormalizes.** FOMC data is about 8 meetings a year, and one curve per category would be very thin. sklearn's `CalibratedClassifierCV` wasn't used because it wraps a classifier. Here the inputs are probabilities already produced, possibly by a model that isn't an sklearn estimator.
- **Backtest integration is a post-processing step, not a harness change.** The harness already records `actual_as_of` and `trained_as_of`, which is everything needed to know which errors were known at each refit. Doing it afterwards means any results frame can be recalibrated without re-running the models (some take a minute).
- **Only intervals are hooked into the backtest.** The harness produces numeric forecasts only. Nothing produces categorical forecasts yet, so `ProbabilityCalibrator` is ready for Task 21 (the FOMC forecaster, which depends on this task) to use in the same fold-by-fold way.

---

## Results on real data

Walk-forward 2015-01 to 2024-12, 3-month horizon, annual refits, `VintageStore.from_db(backfill=True)`, `method="absolute"`. Everything is scored on the same 84 forecasts per row, 2018-01 to 2024-12. 2015–17 are only there to give the calibrator errors to fit on.

**Interval score** (`metrics.interval_score`, in percentage points, lower is better). It is width plus 10× the distance any actual lands outside its 80% interval, so widening intervals can't game it the way it games coverage. Coverage is in brackets.

| Target | Model | Model's own | All past errors | Last 24 | Last 36 | Last 60 |
|---|---|---|---|---|---|---|
| CPI YoY | Random walk | **5.57** (0.71) | 6.17 (0.62) | 5.91 (0.73) | 6.08 (0.68) | 6.21 (0.62) |
| CPI YoY | LightGBM | 6.83 (0.63) | 6.72 (0.64) | 6.22 (0.70) | **6.13** (0.74) | 6.92 (0.68) |
| Unemployment | Random walk | **8.80** (0.71) | 9.45 (0.71) | 12.06 (0.85) | 11.77 (0.85) | 10.13 (0.71) |
| Unemployment | LightGBM | **10.08** (0.61) | 10.96 (0.69) | 12.95 (0.82) | 12.77 (0.82) | 11.56 (0.69) |

**Calibration doesn't help on this data, except for LightGBM on CPI.**
- An earlier version of these notes judged the settings by coverage alone and recommended the last 36 errors. By interval score, the model's own intervals are best in three of four cases. The exception is LightGBM on CPI, where the last 36 errors are about 10% better.
- **Unemployment's higher coverage came from making the intervals much wider.** Coverage by year (LightGBM; the random walk looks the same):

  | | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
  |---|---|---|---|---|---|---|---|
  | Model's own | 1.0 | 0.92 | 0.0 | 0.0 | 0.92 | 0.75 | 0.67 |
  | Last 36 | 1.0 | 0.75 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |

  - **2020 is 0 for every setting.** Nothing fit only on past errors can anticipate COVID.
  - **From 2021 on, the 2020 errors are in the window.** The interval widens to about 5 points and covers everything, including the calm years 2022–24, where narrow intervals would have done.
  - **The coverage was right only on average.** The intervals were too narrow when it mattered and too wide afterwards.
- **CPI with LightGBM** gains from the rolling window mainly in 2022 (coverage 0.58 → 1.0). Inflation errors built up gradually, which a window can follow; the unemployment shock was a single jump.
- **The sample is small and the errors cluster.** One year can decide a row. With 84 forecasts, the standard error of a coverage figure is about 0.044 even before the clustering. Differences of a few tenths in interval score shouldn't be read as real.
- **I kept the default `max_residuals=None`, and I don't recommend a window.** For Task 19, I'd report interval score and coverage by year alongside overall coverage. The next thing to try is conditional widths: scaling the interval by recent volatility, so it widens in a shock and narrows again after.

### Added along the way: `metrics.interval_score`
`interval_score(intervals, actuals, coverage)` is the mean Winkler/Gneiting–Raftery score: width + 2/(1 − coverage) × distance outside. It is a proper scoring rule: the true quantiles minimize it, so it can rank interval methods, which coverage can't. It isn't in `summarize`, because a results frame doesn't record the interval's nominal coverage.

## Tradeoffs and open questions
- **Symmetric vs. signed.** Signed needs about twice as many errors for the same coverage (two tails). I compared the two on coverage only (signed was further from 80%), not on interval score, so absolute stays the default without strong evidence either way.
- **One interval width per fold.** Every forecast in a fold gets the same width. Conditional widths (e.g. conformalized quantile regression, or scaling by the model's own interval width) would let calm and volatile periods differ within a fold.
- **Folds are refit points, not forecast dates.** A forecast late in a fold doesn't use errors resolved after the fold started. Recalibrating at every forecast would use slightly more data, but it would no longer line up with "fit on fold N, apply to N+1".
- **Isotonic can output exactly 0 or 1.** That's fine for Brier, but infinite for log-loss. `calibrate_categorical` falls back to the raw probabilities if every category maps to 0. Platt never outputs exactly 0 or 1.

---

## Checks
- `uv run pytest`: 238 passed (17 new in `tests/test_calibration.py`, 3 in `tests/test_metrics.py` for `interval_score`: computed by hand, lowest at the true quantiles versus half and double width, bad input refused). The calibration tests cover:
  - the conformal quantile computed by hand, and `min_residuals` at 80%/90% and absolute/signed, with one residual fewer refused
  - 80% intervals fit on 1,000 heavy-tailed (t, 4 d.f.) errors covering 80% ± 3% of 10,000 new ones, for both methods
  - signed intervals recentring a forecast biased by 2
  - overconfident probabilities: the reliability-curve gap from y = x falls from over 0.08 to under 0.03 (isotonic and sigmoid), and Platt recovers the true probabilities within 0.03
  - categorical: calibrated forecasts sum to 1, keep category order, and improve mean Brier by more than 0.02
  - a walk-forward backtest with an overconfident model (±0.1 intervals on N(0, 1) noise): raw coverage under 15%, calibrated 80% ± 8%, half-width about z₀.₉ = 1.28, point forecasts unchanged
  - no lookahead: the first fold uncalibrated, each fold's `n_calibration` equal to the forecasts resolved by its refit (8 at the 2009 refit, counted by hand), and the rolling window capping it
  - bad input refused
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems. sklearn is imported with `# type: ignore[import-untyped]`, as in Task 16.
