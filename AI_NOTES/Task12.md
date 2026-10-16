# AI Task 12: AI feature pipeline

Done on 2026-10-15. No migration and nothing written to the live database. Features are built on demand. Task 15 (LightGBM) and Task 21 (the job) store them as snapshots.

## Quick reference

```python
from eco_prediction.ai.features import AIFeatureEngineer, build_ai_features
from eco_prediction.db.forecasts import save_feature_snapshot

features = build_ai_features(conn, date(2026, 10, 15), "HF_NEW_MODELS")  # + target_* aliases
save_feature_snapshot(conn, date(2026, 10, 15), features)

fe = AIFeatureEngineer()
X = fe.build_matrix(VintageStore.from_db(), month_ends)        # or a PointInTimeData view
```

```sh
uv run pytest tests/test_ai_features.py
```

---

## Decisions (my defaults, not asked)

### The same design as the economy's `features/engineering.py`
- **An `AIFeatureEngineer` class:** `build_features`, `build_features_as_of` (cached per store), `build_matrix` and `series_as_of`. It reads a `VintageStore` or a `PointInTimeData` view.
  - So Tasks 13–15 can run AI features through the walk-forward harness, and a date past a view's cutoff raises `LookaheadError`.
- **It reuses the economy's helpers** (`to_monthly`, `pct_change`, `_months_between`, `_store`, `Features`) rather than copying them. The economy file is unchanged.
- **Lags count back from each series' latest available month,** not a fixed calendar month (the ragged edge).
- **A series more than 3 months stale,** or not started yet, gives None for its features. Every feature name is always present, so rows line up.
- **The task's `build_ai_features(conn, as_of_date, target_series)`** is a thin wrapper. It loads the AI series from `conn` into a store.

### What the features are

| Kind | Series (prefix) | Features |
|---|---|---|
| Counts | arXiv (`arxiv`), HF (`hf`), GitHub repos (`github`), GitHub stars (`stars`) | level, `_log` (natural log), `_lag1/3/6/12`, `_growth_1m/3m/12m` (%), `_months_behind` |
| Step series | GPQA (`gpqa`), SWE-bench (`swebench`), ARC-AGI-2 (`arc2`), compute (`flops`) | level, `_lag1/3/6/12`, `_chg_1m/3m/12m` (points or log10 units), `_months_since_change`, `_last_jump`, `_months_behind`, and `_headroom` (100 − level) for the three percentages |
| Cross-series | arXiv vs. HF and GitHub | `corr12_hf_arxiv` and `corr12_github_arxiv` (correlation of monthly log changes over the last 12 months), `log_ratio_hf_arxiv` and `log_ratio_github_arxiv`, `sota_increases_12m` (monthly increases across the three benchmarks in the last year) |
| Calendar | the as-of date | `month_1..12`, `quarter_1..4`, `trend_months` since 2010-01 |

**Notes on the choices:**
- **Step series get changes in points, not %.** A 5-point SOTA gain means the same at 40% and at 90%, and compute is already in log10.
- **`_months_since_change` and `_last_jump`** are the task's "time since last update, magnitude of last jump". A series that has never changed reports the months since its first value, and None for the jump.
- **Cross-series signals are deliberately small.** The task's example is the "HF upload trend for predicting arXiv". A correlation needs 12 full months of both series and some variation, otherwise it's None. The ratios use the latest month both series share.
- **`sota_increases_12m`** is a frontier-activity index. It counts only benchmarks with a value 12 months back, so a benchmark that started recently doesn't count as "no increases".
- **`GITHUB_AI_STARS` is included as a count.** It has no backfill (one snapshot a month from now on), so its growth features stay None for a year.

### `target_series` adds aliases
- With a target series, that series' features are repeated under a `target` prefix: `target`, `target_log`, `target_lag1`, `target_months_since_change`, and so on.
- **Why:** history is short (HF from 2022-04, SOTA series from 2023-11 to 2025-03), so Task 15 will probably want one model across the series of a kind. The aliases give every series the same column names.
- Cross-series features aren't aliased. An unknown series raises `ValueError`.

### Reconstructed data
- **Backfilled rows are read like any vintage** at their `as_of` (the first day after their month). Before live pulls started, they're the only history there is.
- **So backtests over the backfilled period inherit their hindsight.** For example, late Epoch evaluations are dated as if known at month end (AI_NOTES/Task7.md).
- **Rejected:** a per-feature "reconstructed" flag. Until live pulls have run for a while every row is reconstructed, so the flag would be constant in training. The backtests in Tasks 13–15 should report results for the reconstructed period separately instead.

### Snapshots
- **The economy's `save_feature_snapshot` is used as is.** It already turns NaN into JSON null, and `feature_snapshots` has no domain column to fill.
- An AI snapshot is told apart by its forecast's question, as in the economy.

---

## What I did
- **`src/eco_prediction/ai/features.py` (new):**
  - `AIFeatureEngineer` and `build_ai_features`.
  - `store_from_conn` (a store loading AI series from one connection).
  - The series constants (`COUNT_SERIES`, `STEP_SERIES`, `AI_FEATURE_SERIES`, `SERIES_PREFIX`).
- **`tests/test_ai_features.py`: 16 tests, not tagged `economy`.** They run on a hand-built store, so each value is computed by hand:
  - **Counts:**
    - Level, log and lags, including a live revision of the previous month.
    - Growth rates.
  - **Point in time:**
    - The day before a month and a revision are published, neither is seen.
    - A `PointInTimeData` view refuses dates after its cutoff.
  - **Missing data:**
    - Series with no data and a series 5 months stale give None.
    - Calendar features never go missing.
    - The feature names are identical with and without data.
  - **Step series:**
    - Lags, changes, months since change, last jump and headroom.
    - A series that never changed.
    - Only percentages get headroom.
  - **Cross-series:**
    - HF exactly 2× arXiv gives correlation 1 and log ratio ln 2.
    - Missing GitHub gives None.
    - Benchmark increases count only series with a full year.
  - **Calendar:** month, quarter and the trend.
  - **Target aliases:** for a count series and a step series. An unknown series raises.
  - **The matrix:** its index, and NaN for missing features.
  - **The database:** features built from the database, and a snapshot round trip, with JSON null for missing values and the same keys.

## Checks
- `uv run pytest`: **547 passed**, including the economy safety net.
- `ruff check`, `ruff format` and `mypy src` are clean.
- **Not checked on real data:** the AI backfills haven't been run on the live database yet (Tasks 4–7).

## Where I changed the task's plan
- **A class plus the task's function,** rather than only `build_ai_features`, so the harness can use it like the economy's `FeatureEngineer`.
- **`target_series` adds `target_*` aliases** rather than selecting features.
- **Reconstructed data isn't flagged per feature** (reasoning above).

## Tradeoffs
- **109 features (about 130 with target aliases) for short histories.** HF has about 54 months, SOTA 12 to 36. Task 15 should drop sparse columns (`features.engineering.drop_sparse` works on this matrix) and keep trees shallow.
- **The 12-month correlation is noisy.** Twelve monthly changes give a wide confidence band. It's a weak signal at best.
- **Stars and GitHub repos overlap** (the same topics). Trees cope, but SHAP credit will split between them.
- **Features describe the latest month,** so their meaning shifts with ingestion lag. A run on the 1st, before that day's live pulls, sees one month less. `_months_behind` lets a model see this, as the economy's harness cutoffs do with month ends.
