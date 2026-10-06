# Task 14: feature engineering pipeline

Done on 2026-10-05.

## Quick reference

```python
from datetime import date
from eco_prediction.backtest.data import VintageStore
from eco_prediction.features.engineering import FeatureEngineer, drop_sparse

fe = FeatureEngineer()
store = VintageStore.from_db()
fe.build_features_as_of(store, date(2020, 3, 31))     # {'t10y2y': 0.47, 'oil_mom': -32.6, ...}
X = fe.build_matrix(store, [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)])
X, dropped = drop_sparse(X, max_missing=0.5)

# Inside a harness model, pass the PointInTimeData view instead of the store:
fe.build_matrix(data, dates)      # LookaheadError if any date is after data.cutoff
```

```sh
uv run pytest tests/test_features.py
```

---

## What I did

### `src/eco_prediction/features/engineering.py`
`FeatureEngineer.build_features(series_data, as_of_date)` returns 41 features. The names are always the same. A value is a float, or `None` when it can't be computed:

| Group | Features |
|---|---|
| Yield curve | `t10y2y`, `t10y3m` (latest daily value) |
| Prices | `oil_*`, `gas_*`, `ppi_*` with `_mom`, `_3m`, `_yoy` (% changes) |
| CPI (target) | `cpi_yoy`, `cpi_yoy_lag1..3`, `cpi_mom` |
| Unemployment (target) | `unrate`, `unrate_diff1`, `unrate_ma3`, `unrate_lag1..3` |
| Jobless claims | `claims` (latest week), `claims_4wk`, `claims_4wk_yoy` |
| Other V1 series | `wages_yoy` (CES0500000003), `rent_yoy` (CUSR0000SEHA), `mich` |
| Calendar | `month_1..12` (as-of month), `trend_months` (months since 1990-01) |

Other functions:
- **`build_features_as_of(source, date)`** reads the series from a `VintageStore` or `PointInTimeData` view and builds one feature vector.
- **`build_matrix(source, dates)`** builds one row per date, in a DataFrame indexed by `as_of`.
- **`drop_sparse(matrix, max_missing)`** removes columns that are missing too often and returns the names it removed.
- **`to_monthly`, `monthly_mean` and `pct_change`** are the building blocks, and are public so they can be tested.

### Where I changed the task's code
- **Point-in-time rows.** Each row of `build_matrix` uses the vintages known on its own date, not the revised data known at training time. That rules out lookahead, and the model trains on the same kind of fresh, unrevised data it sees when predicting. Passing the harness's `PointInTimeData` view refuses dates after its cutoff.
- **"Lag 1" means the latest *published* period, not the previous calendar month.** At the end of March, CPI is known through February, so `cpi_yoy` is February's value and `cpi_yoy_lag1` is January's. The task's sketch (`series[-1]`) did the same thing implicitly. With harness cutoffs (month ends), each series is the same distance behind on every row.
- **Daily and weekly series are averaged by calendar month** before computing changes, using only months that have ended by the as-of date. This smooths daily noise. It also avoids April 2020's negative WTI print (−$37), which would make a percent change meaningless; April's monthly average was +$16.5. The spreads and claims also keep their latest value.
- **Jobless claims are computed on the weekly data.** The 4-week average needs all 4 weeks present. The YoY change compares it with the 4 weeks ending 52 weeks earlier.
- **Target lags for both targets.** The features don't depend on the target, so one feature snapshot (Task 10's `feature_snapshots`) serves every model.
- **Missing values:**
  - A monthly gap of up to 1 month is forward filled (`ffill_limit`). Example: October 2025 CPI, never published because of the shutdown.
  - A series more than 3 months behind (`max_stale_months`) gives `None` for its features. For a daily or weekly value the limit is 14 days (`max_stale_days`).
  - `None` instead of NaN, so a feature dict can be stored as JSONB as it is. `build_matrix` turns `None` into NaN.
- **Cache.** A row depends only on its date, so `build_features_as_of` builds it once per store. A monthly-retrain backtest no longer rebuilds about 300 training rows at every refit. In the timing check below, 108 rows took 3.4s to build and 1ms from the cache.
- **Added V1 series the task didn't list:** wages, rent and MICH. I left out the NSA rent series (CUUR0000SEHA) because it overlaps with the SA one.

### Limitation: ALFRED vintages start late
In the real database, the first vintage of many series is long after the series itself begins:

| Series | First vintage |
|---|---|
| UNRATE, CPIAUCSL | 1960, 1972 |
| PPIACO, MICH | 1996, 1999 |
| ICSA, GASREGW | 2009 |
| CES0500000003, DCOILWTICO, CUSR0000SEHA | 2011 |
| T10Y2Y, T10Y3M | 2014 |

A strict point-in-time row before those dates has nothing for those series. From 1990, 13 of the 41 columns are missing in more than half of the rows, and `drop_sparse` removes them. **From 2015 on, which is Task 19's window, every feature is present in every month.**

### Backfill option: `VintageStore(..., backfill=True)`
Training from 1990 instead of 2015 gives 360 rows at Task 19's first fit instead of 60, and covers three recessions. So I added an option, off by default, in `src/eco_prediction/backtest/data.py`:

```python
store = VintageStore.from_db(backfill=True)
store.release_lags          # days, once the series are loaded
WalkForwardBacktest(..., store=store)
```

- **What it does.** Each period in a series' first vintage is treated as published `release_lag` days after the period date, if that's earlier than the first vintage. Later vintages are real revisions and keep their dates. Because it's done in the store, the features, `PointInTimeData` and the harness all use it, and the lookahead checks still apply.
- **The lag is measured, not guessed.** `release_lag(history)` takes the 90th percentile of the delays (rounded up) for periods newer than anything in the first vintage. History that a later vintage adds further back (PPI back to 1913, CPI to 1947) isn't counted as a release. Pass `release_lags={...}` to override. A series with no newer periods is left as it is.
- **Measured lags (days from the period date):** CPI 53, UNRATE 37, PPI 48, wages 37, MICH 30, rent 48, gasoline 1, claims 5, spreads 1, oil 8.
- **How much lookahead it adds.** None for the spreads and oil (market prices, never revised), and almost none for gasoline. Claims, PPI and MICH leak only their later revisions, which are small. CPI and unemployment don't need it: their vintages go back to 1972 and 1960.
- **Coverage from 1990 with backfill:** the only gaps left are real ones.
  - wages: 47% missing, because the series starts in 2006
  - gasoline: 2–4% missing, because it starts in August 1990
  - claims: one row missing, October 2025, when claims weren't published during the shutdown
- **Nothing changes after the first vintages.** Every feature row from 2014-01 on is identical to the strict build. Random-walk backtests for 2020–2023 give identical results with and without backfill.

---

## Checks
- `uv run pytest`: 180 passed (18 new in `tests/test_features.py`, 4 new backfill tests in `tests/test_backtest_harness.py`). They cover:
  - PPI MoM, 3-month and YoY changes against hand calculation
  - oil changes on monthly averages, and months counted only once they've ended
  - CPI and unemployment features describing the latest published period, with lags
  - the staleness boundary (3 months is fine, 4 is `None`)
  - one missing month filled, two not, and a value published as missing
  - claims level, 4-week average, YoY, and a missing week
  - the spread's staleness boundary, calendar dummies and trend
  - periods after the as-of date ignored
  - matrix rows using the vintage known on each date, first print vs revision
  - a view refusing dates after its cutoff
  - rows cached per store, with callers getting a copy
  - missing series giving `None` with every name present, and short history giving `None`
  - `drop_sparse`
  - backfill: the release lag ignoring back-added history and outliers, first-vintage rows moved only when that's earlier, revisions untouched, identical values after the first vintage, lag override
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
- **Real data:**
  - March 2020: claims 3.28M, February unemployment 3.5%, CPI YoY 2.3%, oil −33% MoM.
  - June 2022: CPI YoY 8.5% (May), oil +62% YoY.
  - 2015–2023: no missing values.
