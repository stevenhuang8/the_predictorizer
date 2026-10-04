# Task 8: backfilling history from 1990

Done on 2026-10-04.

## Quick reference

```sh
uv run python -m eco_prediction.scripts.backfill_data                  # all 12 V1 series, from 1990
uv run python -m eco_prediction.scripts.backfill_data CPIAUCSL UNRATE  # just these
uv run python -m eco_prediction.scripts.backfill_data --start 1980-01-01
uv run python -m eco_prediction.scripts.backfill_data --restart        # ignore the checkpoint
uv run python -m eco_prediction.scripts.backfill_data --report         # coverage report only
uv run pytest tests/test_backfill.py
```

---

## What I did

### `src/eco_prediction/scripts/backfill_data.py`
- For each series, it calls `ingest_series(full=True, observation_start=--start)`, which fetches every ALFRED vintage for those periods. Each series is committed in its own transaction.
- **Checkpointing.** After each series commits, it is recorded in `logs/backfill_checkpoint.json`. The file is written atomically and `logs/` is gitignored. A rerun skips the series already done, so an interrupted run carries on from where it stopped.
  - A checkpoint only counts for the start/end range it was written for. A run with a different range starts over.
  - A series that fails isn't recorded, so the next run tries it again.
  - Ctrl-C exits with code 130 and keeps the checkpoint.
- **Progress and logging.** It logs a line like `[i/N] series: fetched= inserted= in Xs (run Ys)` to stderr and to `logs/backfill.log`.
- **Rate limits.** The FRED client from Task 6 already handles these (120 requests/min, with backoff on 429 and 5xx).
- **Monthly snapshots aren't needed.** Asking for the full real-time period already returns every revision.
- **Coverage report.** At the end it prints, for each series: rows, vintages, first and last period, and first and last vintage. It flags a series with no data, or one whose first period is more than a year after `--start`.

### The task estimated "several hours". The real run took 12 s
Task 7's ingest had already loaded the full vintage history, so this run mostly confirmed it. Only 6 rows were new: vintages published since 2026-10-01. On an empty database the run makes about 4 requests per series, which still finishes in well under a minute.

---

## Coverage (2026-10-04)

| Series | First period | First vintage (point-in-time starts) |
|---|---|---|
| CPIAUCSL | 1947 | 1972-07 |
| UNRATE | 1948 | 1960-03 |
| T10Y2Y / T10Y3M | 1976 / 1982 | 2014-01 |
| DCOILWTICO | 1986 | 2011-04 |
| GASREGW | 1990-08 (the series starts here) | 2009-09 |
| PPIACO | 1913 | 1996-12 |
| **CES0500000003** | **2006-03** (the series starts here) | 2011-03 |
| ICSA | 1967 | 2009-05 |
| MICH | 1978 | 1999-02 |
| CUUR0000SEHA / CUSR0000SEHA | 1914 / 1981 | 2011-04 |

There are 58,316 observations in total. Every series except CES0500000003 has periods back to about 1990.

## Things to decide later
- **The wage series starts in 2006.** CES0500000003 covers all private employees and only begins in 2006. AHETPI (production and nonsupervisory workers) goes back to 1964 and could replace it, or be added alongside it.
- **Point-in-time data starts late.** For most series, ALFRED's first vintage is in 2009–2014. Periods before that are stored only as they stood on that first vintage date, so a backtest before then sees nothing. For market series that are essentially never revised (the spreads, oil and gas), we could add a pseudo-vintage dated at the observation date plus the publication lag. That would be a modeling choice, so I didn't do it here.

## Checks
- `uv run pytest`: 53 passed (5 new: checkpoint/resume, retrying a failed series, ignoring a checkpoint for a different range or with `--restart`, an unreadable checkpoint, and the coverage report).
- `ruff check src tests` and `mypy src tests`: no problems.
- Real database: the full run took 12 s. A second run logged `Resuming: 12 of 12 series already done` and made no API calls.
- Revision spot checks:
  - CPI for June 1995 reads 152.5 (1995-07-14, first release), then 152.6, 152.5, and 152.4 (2002-02-20).
  - UNRATE for April 2020 reads 14.7 at first release, then 14.8, 14.7, 14.9, and 14.8 after the seasonal revisions.
