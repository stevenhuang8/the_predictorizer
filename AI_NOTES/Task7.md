# AI Task 7: Epoch AI client, benchmark SOTA and largest training run

Done on 2026-10-14. Nothing has been written to the live `eco_forecast` database yet. The backfill and live pulls are the user's to run (see Quick reference). No key is needed.

## Quick reference

```sh
uv run python -m eco_prediction.ai.ingest_epoch --backfill --dry-run   # print only, all four series
uv run python -m eco_prediction.ai.ingest_epoch --backfill             # all four series
uv run python -m eco_prediction.ai.ingest_epoch                        # live: last 12 ended months
uv run python -m eco_prediction.ai.ingest_epoch gpqa --backfill        # one series: gpqa, swe_bench, arc_agi_2, compute
uv run pytest tests/test_epoch_client.py
```

Every run downloads two files (~5 MB) once and takes a few seconds.

---

## What Epoch publishes (checked 2026-10-14)
- **No API, only static downloads** that Epoch refreshes as it adds models:
  - `https://epoch.ai/data/benchmark_data.zip` has one CSV per benchmark, with one row per model run.
  - `https://epoch.ai/data/notable_ai_models.csv` has 1,078 models, 540 of them with a date and a compute estimate.
- **Every row has the model's release date.** Epoch-run benchmarks also have `Started at`, the date of Epoch's own run.
- **`benchmark_metadata.csv` names each benchmark's headline column:**
  - GPQA Diamond and SWE-bench Verified use `Best score (across scorers)`.
  - ARC-AGI-2 uses `Score`. It's copied from the ARC Prize leaderboard and is in `arc_agi_2_external.csv`.
  - `arc_agi_external.csv` is ARC-AGI-1.
- **Scores are fractions (0 to 1).** Compute is in FLOP.
- **The data is CC BY 4.0,** so it can be used with credit. The source description says so.

## Decisions made before starting (2026-10-14)

### The user's choices
- **A monthly running max, not event-dated records.** A month's value is the best among models released by that month's last day.
  - This fits the monthly 1/3/6-month targets and reuses `ai/monthly_ingest.py` unchanged in behavior. Backfill rows are reconstructed, and live rows are only added when a month's value changes.
  - **Rejected:** one row per record, dated to the release date. It keeps exact dates, but needs a new ingest path, and every later step would have to resample it.
- **ARC-AGI-2, not ARC-AGI-1.** ARC-AGI-1 has been at 98.5% since 2026-06, so there's nothing left to forecast. ARC-AGI-2 is at 95.0%.
- **Compute estimates of every confidence level,** including "Speculative". Frontier labs rarely disclose compute, so excluding them would drop Grok 4, Gemini 1.0 Ultra and others from the record line.

### My defaults (stated, not asked)
- **Scores are stored in percent** (0 to 100), so thresholds like `swe_bench_verified>=80` read naturally. Compute is stored as log10(FLOP). Both are rounded to 4 decimals.
- **Each SOTA series starts in the month its benchmark was published** (from `benchmark_metadata.csv`): GPQA Diamond 2023-11, SWE-bench Verified 2024-08 and ARC-AGI-2 2025-03. That first month's value includes older models evaluated since. Compute starts 2010-01.
- **Months before the first dated result are left out** rather than stored as 0.
- **A live pull re-checks the last 12 ended months, not 3.** The whole history comes in one download anyway. A model released months ago but only evaluated now changes those months, and the change is stored as a live vintage.
- **Rows without a date or value are skipped.** A value or date that doesn't parse, a missing column, or a file that isn't a zip raises, because that means Epoch changed its format.

---

## What I did

### `ai/months.py` and `ai/monthly_ingest.py` (small, shared changes)
- **`MonthlyCount.count` is now a `float`.** It holds a count, or a month's running max.
- **`MonthlySeries` has two new fields, with defaults that keep arXiv, HF and GitHub as they were:**
  - `allow_zero` (default False): a 0% score is real. Negative values are always refused.
  - `value_format` (default `.0f`): how dry runs and progress print a value. Epoch uses `.2f`.

### `ai/sources/epoch_client.py`
- **`EpochClient(BaseAPIClient)`:**
  - `benchmark_results(benchmark)` returns each run as a percent.
  - `training_runs()` returns each model as log10(FLOP). A compute of 0 or less raises.
  - Each file is downloaded once per client and cached, so the four series share one zip download and one CSV download.
  - The timeout is 120 s.
- **`GPQA_DIAMOND`, `SWE_BENCH_VERIFIED` and `ARC_AGI_2`** each give a file and its headline column.
- **`running_max(results, months)`** gives the month-end best so far.

### `ai/ingest_epoch.py`
- **Four series under one source, `Epoch AI`, all in domain `ai`:**

  | Key | Series | Units | First month | Forecast target |
  |---|---|---|---|---|
  | `gpqa` | `EPOCH_GPQA_SOTA` | Percent | 2023-11 | `benchmark_sota` |
  | `swe_bench` | `EPOCH_SWEBENCH_SOTA` | Percent | 2024-08 | `benchmark_sota` |
  | `arc_agi_2` | `EPOCH_ARC2_SOTA` | Percent | 2025-03 | `benchmark_sota` |
  | `compute` | `EPOCH_LARGEST_FLOPS` | log10(FLOP) | 2010-01 | `largest_training_run` |

- **The default command runs all four.** An optional first argument picks one. The usual `--backfill`, `--start`, `--months` and `--dry-run` options apply to each.
- **A dry run of all four labels each series** with a `== SERIES_ID` line.

### `tests/test_epoch_client.py` (21 tests, not tagged `economy`)
- **Parsing:**
  - The headline column is used, as a percent.
  - Rows without a date or score are skipped.
  - The zip is downloaded once across benchmarks.
  - Compute is log10, and blank estimates are skipped.
- **Format changes raise:** a missing file in the zip, a missing column, a bad date, a bad value, no usable rows, a non-zip download, zero compute, and a 404.
- **Running max:**
  - It steps only at records, from unsorted input.
  - A release on a month's last day counts for that month.
  - Months before the first release are left out.
  - Older models count in the first month.
  - Values are rounded to 4 decimals.
- **Ingestion (throwaway database):**
  - A late evaluation of a July model adds live vintages for July to September, and leaves June alone.
  - A 0% ARC-AGI-2 score is stored, but zero compute is refused.
  - All four series share the `Epoch AI` source.
- **CLI:**
  - One series fetches only its file and closes the client.
  - The labeled dry run of all four.
  - A download failure exits with code 1.

### Checks
- `uv run pytest`: 424 passed, including the economy safety net.
- `ruff check`, `ruff format --check` and `mypy src` are clean.

## Real data (dry runs on 2026-10-14)

| Series | 2026-09 | Records so far |
|---|---|---|
| GPQA Diamond | 96.40% (Gemini 4 Argon, released 2026-09-30) | Dated runs start in 2024-09 |
| SWE-bench Verified | 83.47% (Claude Opus 4.7, 2026-04) | 9 records since 2024-11 |
| ARC-AGI-2 | 95.00% (GPT-6 Astra, 2026-09-03) | 0% (2024-07) to 95% |
| Largest training run | 27.97 log10 FLOP (GPT-6 Astra) | 26.70 (Grok 4, 2025-07) before that |

## Where I changed the task's plan
- **Monthly running max instead of one row per record** (the user's choice).
- **`EPOCH_ARC2_SOTA` instead of `EPOCH_ARC_SOTA`,** because it tracks ARC-AGI-2 (the user's choice).
- **Scores are stored in percent,** using each benchmark's headline column rather than `mean_score`.
- **No separate notification for Epoch updates.** The live pull re-downloads and re-checks 12 months, so rerunning it monthly (or daily) picks updates up.
- **The shared monthly ingest was generalized** (float values, `allow_zero`, `value_format`) instead of writing a second ingest path.

## Tradeoffs
- **All history is hindsight.** The backfill is today's Epoch data, dated as if each month's value were known on the first day of the next month. Many scores were only measured later: GPT-4o's SWE-bench Verified score (released 2024-11) comes from a February 2026 run, and Gemini 4 Argon (released 2026-09-30) was evaluated on 2026-10-13. The rows are flagged `reconstructed`. Live pulls record genuine vintages from now on.
- **Epoch re-runs and prunes.** SWE-bench Verified only has 35 runs, all from Epoch's 2026 re-run, so its history starts in 2024-11 even though the series starts in 2024-08. GPQA has no dated runs before 2024-09.
- **⚠️ The latest compute record looks suspect.** Epoch's row for GPT-6 Astra says 9.4e27 FLOP (27.97), but its own note estimates "around 9e26 FLOP" with a CI of about [5e26, 2e27]. If the note is right, the value is 10× too high, and the jump from 26.70 to 27.97 in 2026-09 is mostly a data-entry error. I stored Epoch's value as published. If Epoch corrects it, the live pull will record the correction as a new vintage. Worth keeping in mind before a model trains on this series.
- **Speculative compute estimates are in the record line.** Some records (GPT-3.5, Gemini 1.0 Ultra, Grok 4) rest on Epoch's guesses.
- **Step series change rarely.** GPQA is flat for months at a time, which favors a random-walk baseline, and saturating benchmarks (GPQA at 96%, ARC-AGI-2 at 95%) leave little room to forecast.
- **Rows with no release date are dropped** (two GPQA rows today), even though their scores are real.
