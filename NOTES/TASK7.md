# Task 7: loading FRED data into the database

Done on 2026-10-01.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate                        # apply 002 first
uv run python -m eco_prediction.data.ingest                       # all 12 V1 series
uv run python -m eco_prediction.data.ingest CPIAUCSL UNRATE       # just these
uv run python -m eco_prediction.data.ingest --full                # refetch all history
uv run python -m eco_prediction.data.ingest CPIAUCSL --observation-start 2024-01-01
uv run pytest tests/test_ingest.py
```

---

## What I did

### `src/eco_prediction/data/ingest.py`
- `ingest_series(conn, client, series_id)` loads one series:
  - It upserts `FRED` into `sources` and the series' metadata into `series`. Metadata is refreshed on every run.
  - It inserts the observations with `ON CONFLICT DO NOTHING`.
  - It doesn't commit; the caller does.
- **The first run backfills the full vintage history. Later runs are incremental.** They only fetch vintages published after the latest stored `as_of`, and skip rows whose value is the same as what's stored. With nothing new, it costs 2 small requests and no observation fetch.
- `--full` / `full=True` refetches everything. It's safe to run any time.
- `ingest_many()` commits each series on its own. A failing series is reported and the run moves on.
- The command prints `fetched` (rows after dropping repeats) and `inserted` (rows new to the database) for each series. It exits with 1 if any series failed.

### `migrations/002_widen_series_metadata.sql`
- `units`, `frequency` and `seasonal_adjustment` are now `TEXT`. ICSA's frequency, "Weekly, Ending Saturday", is 23 characters and didn't fit `VARCHAR(20)`.

### V1 series

| Series | What | Vintages from |
|---|---|---|
| CPIAUCSL | CPI, all items (SA) | 1972 |
| UNRATE | Unemployment rate | 1960 |
| T10Y2Y, T10Y3M | 10y−2y and 10y−3m Treasury spreads (daily) | 2014 |
| DCOILWTICO | WTI crude oil (daily) | 2011 |
| GASREGW | Regular gasoline (weekly) | 2009 |
| PPIACO | PPI, all commodities | 1996 |
| CES0500000003 | Average hourly earnings, total private | 2011 |
| ICSA | Initial jobless claims (weekly) | 2009 |
| MICH | Michigan 1-year inflation expectations | 1999 |
| CUUR0000SEHA / CUSR0000SEHA | CPI rent of primary residence (NSA / SA) | 2011 |

**Choice:** the task said "wage growth series" without naming one. I picked CES0500000003, which is monthly, has vintages, and comes out with the jobs report.

### Things to remember
- **ALFRED's vintage history starts late for many series** (see the table above). Before that date, point-in-time queries return nothing, even though FRED has the data. Backtests for those series can't start earlier without using the revised data (lookahead).
- **`as_of` is a date, not a time.** The Treasury spreads are posted late on the same day: 95% of T10Y2Y values since 2024 have `as_of == observed_at`. If a backtest decides something *during* day D, it should query as of D − 1. As of D means "known by the end of D". Oil (median 6 days) and jobless claims (median 5 days) arrive later.
- **Incremental runs keep to the periods already loaded.** If you first loaded a series with `--observation-start`, run `--full` to backfill the earlier periods.

---

## Checks
- `uv run pytest`: 48 passed (8 new ingest tests, including a live ingest of the last 2 years of CPIAUCSL that is repeated to confirm nothing new is inserted).
- `ruff check src tests` and `mypy src tests`: no problems.
- Real `eco_forecast` database: migration 002 applied, all 12 series loaded (58,310 observations, about 12 s). A second run inserted nothing.
- Spot check: CPI for Jan 2024 reads 309.685 as of 2024-02-14 (first release) and 309.698 today (after the 2025 and 2026 seasonal revisions).
