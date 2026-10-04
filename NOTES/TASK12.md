# Task 12: baseline models and expert benchmarks

Done on 2026-10-04.

## Quick reference

```python
from datetime import date
from eco_prediction.backtest.harness import WalkForwardBacktest, summarize
from eco_prediction.models.baselines import HistoricalMeanModel, RandomWalkModel
from eco_prediction.models.benchmarks import ExpertBenchmark, SPFForecaster

bt = WalkForwardBacktest(date(2020, 1, 1), date(2023, 12, 1), "quarterly")
summarize(bt.run(RandomWalkModel(), "cpi_yoy", 3))
summarize(bt.run(SPFForecaster(), "unemployment", 3))

bench = ExpertBenchmark()
bench.get_cpi_nowcast(date(2024, 3, 11), date(2024, 2, 1))        # 3.12
bench.get_unemployment_spf(date(2020, 5, 31), date(2020, 6, 1))   # 16.1
```

```sh
uv run pytest tests/test_baselines.py tests/test_benchmarks.py
```

---

## What I did

### `src/eco_prediction/models/baselines.py`
Both baselines follow the harness's `fit(data)` / `predict(data, target_period)` interface. Both give 80% intervals by default (`coverage=`).
- **`RandomWalkModel`** predicts the last known value.
  - Its interval is empirical. Say the last known period is k months before the target. The model takes every k-month change in the training history and uses the 10th and 90th percentiles. So the interval matches the real step size, which is about h + 2 months given release lags.
  - The task sketch called this `naive_interval()` without defining it.
- **`HistoricalMeanModel`** predicts the mean of the training history, with mean ± 1.28σ as the interval.
  - Pair it with a rolling window (for example `window_months=120`). Otherwise an expanding window averages back to 1948 for unemployment.

### `src/eco_prediction/models/benchmarks.py`
- **`SPF`**: the Philadelphia Fed's Survey of Professional Forecasters medians (unemployment and CPI).
  - **Point-in-time.** Each survey is visible from its real news release date, taken from the Philly Fed's release-date file. That file starts at 1990Q2, so earlier surveys are assumed to come out on the 15th of the quarter's third month.
  - **Unemployment.** SPF forecasts the quarterly average rate, which is used for every month in that quarter.
  - **CPI year-over-year.** SPF forecasts annualized quarter-over-quarter rates. To turn those into year-over-year, I take the quarterly-average CPI actually known at the cutoff, chain SPF's rates onto it for the quarters that aren't known yet, and take the change from the same quarter a year earlier. That's an approximation, since the target is the monthly year-over-year figure.
- **`SPFForecaster`** wraps SPF as a harness model. It raises `BenchmarkUnavailable` for a target or horizon SPF doesn't cover.
- **`ClevelandNowcast`**: the Cleveland Fed's daily CPI year-over-year nowcasts since 2013-08. `nowcast(as_of, target_month)` returns the latest nowcast made on or before `as_of`.
- **`ExpertBenchmark`** puts these behind the methods in the task's sketch: `get_cpi_nowcast`, `get_unemployment_spf`, plus `get_cpi_spf`.
- **`CachedDownloader`** handles all downloads.
  - Files are cached in `data/cache/benchmarks/` (added to `.gitignore`) and refreshed after a day.
  - If a refresh fails, it uses the stale copy and logs a warning. It only raises when there is no cached copy at all.

### Choices and limits
- **The Cleveland nowcast isn't a harness model.** It only covers the current and previous month. The harness's horizon 1 on day D targets month D + 1, and at that point no nowcast exists yet. So it's available for direct lookups (and for a nowcast comparison later), but not wired into backtests.
- **CME FedWatch isn't implemented.** CME doesn't offer a free historical archive, and the harness doesn't do FOMC targets yet anyway. For Task 21, fed funds futures implied rates are one possible substitute.
- **New dependency: `openpyxl`**. The SPF files are .xlsx and pandas needs it to read them.

---

## Checks
- `uv run pytest`: 119 passed (31 new). The new tests run offline, using canned files and a fake HTTP session. They cover:
  - **Random walk:** it returns the last value, and its interval is exact on a known zigzag series.
  - **Historical mean:** it is the training mean with a 1.28σ interval, and stays constant between retrains.
  - **Fitting errors:** predicting before `fit` and fitting on too little data are both rejected.
  - **Downloads:** a fresh cache is served without a request, a stale one is refreshed, a stale copy is used when offline, and having no copy at all raises.
  - **SPF release dates:** the parser carries the year forward and ignores footnotes, and surveys switch over exactly on their release day.
  - **SPF forecasts:** months map to the right quarter, CPI rates are chained onto known actuals, and SPF works end to end in the harness.
  - **Nowcast:** parsing handles the December-to-January turn and skips empty values, and lookups are point-in-time.
- `ruff check src tests` and `mypy src tests`: no problems.
- **Real data, 2020–2023, quarterly retrain.** RMSE in points (interval coverage of the 80% band in brackets):

  | Target | h | Random walk | 10-year mean | SPF median |
  |---|---|---|---|---|
  | cpi_yoy | 1 | 1.27 (54%) | 3.64 (48%) | 1.28 |
  | cpi_yoy | 3 | 1.84 (50%) | 3.69 (46%) | 1.83 |
  | cpi_yoy | 6 | 2.60 (48%) | 3.73 (50%) | 2.63 |
  | unemployment | 1 | 2.67 (58%) | 2.39 (90%) | 2.74 |
  | unemployment | 3 | 3.17 (50%) | 2.36 (92%) | 3.00 |
  | unemployment | 6 | 2.52 (50%) | 1.68 (98%) | 1.91 |

  - 2020–2023 is a hard test: the COVID unemployment spike and the 2021–22 inflation surge. The random walk's intervals cover only about half the outcomes, well short of the 80% they aim for.
  - SPF is about level with the random walk on CPI. Its April 2022 forecast was 4.2% against an actual of 8.2%.
  - The SPF lookups switch exactly on release dates: the June 2020 unemployment forecast was 3.5% from the survey released before COVID and 16.1% from the survey released on 2020-05-15.
