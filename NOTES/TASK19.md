# Task 19: model comparison report

Done on 2026-10-08. **The success criterion was not met:** LightGBM doesn't beat the random walk on 3-month CPI RMSE (details below).

## Quick reference

```sh
uv run python -m eco_prediction.scripts.model_comparison            # 2020-2023; reuses cached runs
uv run python -m eco_prediction.scripts.model_comparison --refresh  # rerun every backtest (~10 min)
uv run python -m eco_prediction.scripts.model_comparison --start 2018-01-01 --end 2024-12-01 --out results/2018_2024
open results/model_comparison.html
uv run pytest tests/test_model_comparison.py
```

---

## What I did

### `src/eco_prediction/scripts/model_comparison.py`
- **Backtests:** every model (LightGBM, random walk, ARIMA, ETS, historical mean) on both targets at 1, 3 and 6 months.
  - Walk-forward, forecasting on the 1st of each month from 2020-01 to 2023-12 (48 forecasts per run), refit annually on all data published by each refit, scored against first releases.
  - Intervals are 80%. At 3 months, every model also runs at 50% and 95% for the calibration curves.
  - 50 backtests in total, about 10 minutes, each cached in `results/cache/<window>/`, so regenerating the report takes seconds. `--refresh` reruns them.
- **Metrics per (target, horizon, model):**
  - RMSE, MAE, bias, coverage, interval score (Task 18);
  - RMSE relative to the random walk;
  - a **Diebold–Mariano test** against the random walk.
- **Charts** (static PNGs inside the HTML), each with a table of the same numbers:
  - RMSE by horizon;
  - interval calibration curves (nominal vs. actual coverage);
  - forecasts vs. actuals at 3 months;
  - LightGBM's mean |SHAP| by feature. SHAP is recorded as each backtest forecast is made, so there is no extra fitting.
  - Colors follow the dataviz reference palette's validated first five slots, and each model keeps its color in every chart.
- **Tables:** coverage by year, and the FOMC models' Brier scores (Task 21) over the same window at a 7-day lead.
- **Findings:** short statements generated from the table. They name the RMSE winner per target and horizon, say when MAE ranks a different model first, and show the success-criterion verdict.
- **Outputs:**
  - `results/model_comparison.html`: self-contained, about 630 KB.
  - `results/model_comparison.json`: every number in the report.
  - `results/cache/` is gitignored; the report and JSON can be committed.

### Also added
- **`metrics.diebold_mariano(errors_a, errors_b, horizon)`** tests whether two models' squared errors differ.
  - It allows for the overlap between h-step forecasts (autocorrelation up to lag h − 1) and applies the Harvey–Leybourne–Newbold small-sample correction.
  - Why it's needed: with 48 overlapping forecasts, a lower RMSE alone can be noise.
- **`scipy`** is now a direct dependency. It was already installed through scikit-learn; Diebold–Mariano needs its t distribution.

### Where I changed the task's plan
- **Training window.** The task says "2015–2019 for initial training, 2020–2023 for evaluation". I read that as: the first fit uses everything published by the end of 2019 (an expanding window, as in every earlier backtest), and forecasts are scored over 2020–2023. Training on 2015–2019 data only would leave ARIMA and LightGBM about 60 months, too little for a fair comparison. `--start` and `--end` change the window.
- **Significance testing.** The task's criterion is a raw RMSE comparison, which the report gives. Diebold–Mariano p-values sit beside it, so "beats" can be read with its uncertainty.
- **Interval score and coverage by year.** Task 18 showed overall coverage hides when the misses happen; these sit alongside the calibration curves.
- **The FOMC section** wasn't in the task, which predates Task 21.

---

## Results (2020-01 to 2023-12, 48 forecasts per cell)

### Success criterion: not met
LightGBM's 3-month CPI RMSE is **1.98** against the random walk's **1.84** (Diebold–Mariano p = 0.24, so not significantly different either way). Task 16 found the same over 2018–2024 (1.54 vs. 1.43). I marked the task done because the deliverable, the comparison and report, is complete. Making LightGBM win is modelling work for later, and the report says plainly that it doesn't. Reopen the task if you'd rather it stay open until the criterion passes.

### RMSE (percentage points; lower is better)
| | CPI 1m | CPI 3m | CPI 6m | Unemp 1m | Unemp 3m | Unemp 6m |
|---|---|---|---|---|---|---|
| LightGBM | 1.04 | 1.98 | 2.82 | 2.84 | 3.28 | 4.18 |
| Random walk | 1.27 | 1.84 | 2.60 | 2.67 | 3.17 | 2.52 |
| **ARIMA** | **0.83** | **1.24** | **1.87** | 3.35 | 4.40 | 4.74 |
| ETS | 1.16 | 1.68 | 2.37 | 3.05 | 3.86 | 3.78 |
| Historical mean | 2.86 | 2.85 | 2.74 | **2.62** | **2.62** | **1.90** |

- **CPI: ARIMA is best at every horizon**, 28–35% below the random walk. That is significant at 1 month (p < 0.01) and suggestive at 3 months (p = 0.07). It followed the 2021–22 surge and decline more closely than the others.
- **Unemployment: the April 2020 spike decides RMSE.**
  - Every model forecast about 3.5% for the months when unemployment hit 14.8%, then overshot to 15–23% as it fell. ARIMA and ETS extrapolated the spike hardest.
  - The flat historical mean (about 5.8%) "wins" on RMSE only because it never moves.
  - **By MAE, the random walk is best at every horizon** (1.23 / 1.71 / 1.58). The report's findings say so.
  - No difference on unemployment is significant (all p > 0.05).
- **LightGBM** beats the random walk only at 1 month on CPI (1.04 vs. 1.27, p < 0.01). At 6 months on unemployment it's the worst model, with a bias of +2.6: it learned "unemployment falls fast after a spike" and kept predicting it.

### Intervals (3-month; share of actuals inside nominal 50% / 80% / 95% intervals)
| | CPI | Unemployment |
|---|---|---|
| LightGBM | 0.19 / 0.35 / 0.62 | 0.19 / 0.42 / 0.50 |
| Random walk | 0.25 / 0.50 / 0.85 | 0.42 / 0.50 / 0.67 |
| ARIMA | 0.44 / 0.69 / 0.85 | 0.54 / 0.58 / 0.65 |
| ETS | 0.42 / 0.60 / 0.85 | 0.58 / 0.75 / 0.81 |
| Historical mean | 0.44 / 0.79 / 1.00 | 0.23 / 0.73 / 0.92 |

- **Every model's intervals cover too little in this window.**
- **LightGBM's are the worst:** its 80% interval covers 35–42%. It builds intervals from out-of-fold residuals of calmer years (Task 16), and this window is the most volatile in the data.
- **By year (80%, 3-month):** for unemployment, coverage is close to 0 for every model in 2020 except the historical mean, and 0.92–1.0 in 2022–23. This is the same pattern as Task 18: intervals too narrow when it matters.

### What drives LightGBM (mean |SHAP|, 3-month)
- **The top feature for both targets is `trend_months`**, a linear time trend (CPI 0.26, unemployment 0.40 percentage points). Next come the jobless-claims features (`claims_4wk_yoy`, `claims`) and the targets' own lags.
- **Experiment: LightGBM without the trend** (3-month RMSE; exploratory, model defaults unchanged):

  | | 2015–19 CPI | 2015–19 unemployment | 2020–23 CPI | 2020–23 unemployment |
  |---|---|---|---|---|
  | Random walk | 0.618 | 0.253 | 1.845 | 3.167 |
  | LightGBM (with trend) | **0.618** | **0.223** | 1.977 | 3.284 |
  | LightGBM, no trend | 0.752 | 0.231 | **1.763** | 3.259 |

  - **The trend helps in calm years and hurts in the volatile window.**
  - **Dropping it would pass the success criterion on 2020–23** (1.76 < 1.84). But choosing it because it wins on the evaluation window is tuning on the test set, and it loses clearly on 2015–19. I left the feature in. Tasks 16 and 17 suggested dropping it; this experiment doesn't support doing that as a blanket change.
  - Side note: on 2015–19 CPI, LightGBM and the random walk score the same (0.6180 vs. 0.6181) by coincidence. Their forecasts differ by 0.39 points on average; I checked.

### FOMC (7 days before each meeting, 30 meetings, 11 of them cuts or hikes)
| | Brier | Brier on cuts/hikes | Accuracy |
|---|---|---|---|
| **LightGBM** | **0.149** | **0.367** | 0.87 |
| Persistence | 0.295 | 0.592 | 0.87 |
| Climatology | 0.520 | 1.162 | 0.63 |
| Always hold | 0.733 | 2.000 | 0.63 |

This matches Task 21's longer backtest: a week out, the FOMC model halves persistence's Brier score.

## Takeaways
1. **For CPI, ARIMA is the model to beat, not the random walk.** It should be the reference in future comparisons, and is a candidate to weight most in a combined forecast.
2. **No model forecasts unemployment better than "no change" over this window.** The 2020 shock dominates any scoring that squares errors. Whether a model handles unemployment better should be judged on a window without 2020, or by MAE.
3. **All the intervals need work**, LightGBM's most. Task 18 found conformal calibration doesn't fix it here; widening intervals with recent volatility is the next thing to try.
4. **LightGBM's case currently rests on 1-month CPI and the FOMC model.** For numeric targets at 3 to 6 months, it isn't earning its complexity yet.

## Tradeoffs
- **One evaluation window.** 2020–2023 is the task's window, and it is extreme: COVID plus the largest inflation surge in 40 years. Rankings can flip in calm years (as with the trend feature). `--start`/`--end` makes rerunning on 2015–2019 or 2018–2024 one command.
- **Annual refits** keep the full run at about 10 minutes. Monthly refits would be roughly 12 times slower for LightGBM and ARIMA.
- **Static PNG charts** rather than interactive ones: the report has to be one self-contained file that opens without a server. Every chart has a table with its numbers.
- **The cache is keyed by window and refit frequency, not by code version.** After changing a model, run with `--refresh`, or old results are reused.

---

## Checks
- `uv run pytest`: 278 passed (10 new):
  - 7 in `tests/test_model_comparison.py`, on the synthetic store where MICH leads unemployment by 5 months, with cheap stand-ins for ARIMA and ETS. They cover:
    - every model, horizon and calibration level backtested, with SHAP recorded per forecast
    - the metrics table, with LightGBM significantly beating the random walk when there is real signal
    - Diebold–Mariano left blank when two models' errors are identical
    - SHAP ranking the signal features first
    - wider nominal intervals never covering less
    - coverage-by-year columns
    - HTML with 4 inline charts, and JSON contents
    - the success criterion passing, failing, or not applying
    - cached runs reused, and recomputed with `refresh`
  - 3 in `tests/test_metrics.py` for `diebold_mariano`:
    - detects a clearly better forecast, and is symmetric when the two are swapped
    - rejects equal forecasts under 10% of the time at the 5% level
    - refuses bad input
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
- **Real run:** 50 backtests plus 4 FOMC runs. I looked at the rendered charts: no label collisions or overflow, and the legend and colors are consistent.
