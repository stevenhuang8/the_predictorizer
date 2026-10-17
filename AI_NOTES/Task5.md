# AI Task 5: Hugging Face Hub client and HF_NEW_MODELS ingestion

Done on 2026-10-12. Nothing has been written to the live `eco_forecast` database yet. The backfill and the first live pull are the user's to run (see Quick reference).

## Quick reference

```sh
uv run python -m eco_prediction.ai.ingest_huggingface --backfill --dry-run   # print only
uv run python -m eco_prediction.ai.ingest_huggingface --backfill             # 2022-04 to last month
uv run python -m eco_prediction.ai.ingest_huggingface                        # live: last 3 ended months
uv run pytest tests/test_huggingface_client.py
```

`HF_TOKEN` in `.env` is optional; it's sent when set.

---

## What the API allows (probed 2026-10-12)
- **No count endpoint.** `/api/models` returns no total, so counting a month means listing every model created in it. Pages hold at most 1,000 models (`limit=5000` still returns 1,000). The next page is a cursor URL in the `Link` header.
- **`createdAt` only goes back to March 2022.** Every model older than that has `createdAt = 2022-03-02T23:29:04Z`, the date of HF's platform migration. So history starts in **April 2022** (`FIRST_MONTH`), about 4.5 years. The task said "backfill from model createdAt" without knowing that.
- **The pace now:** about 4,400 new models a day in a 5-page sample on 2026-10-12. September 2026 as a whole came to 92,008 (about 3,100 a day).
- **Rate limit:** `ratelimit-policy: "fixed window";"api";q=500;w=300`. That is 500 requests per 5 minutes for an IP without a token. The `ratelimit` header gives the requests left (`r=`) and the seconds until the window resets (`t=`).

## Decisions (my defaults, not asked)
- **One newest-first scan per run.**
  - `sort=createdAt&direction=-1`, following the cursor links.
  - `createdAt` is tallied by UTC month until the listing has passed the earliest month asked for.
  - Each month is reported as soon as the scan passes it, so the backfill commits month by month.
  - Listing oldest-first would reach the old months first, but a live pull would then have to page through all of history to get to recent months.
- **The scan waits 2 days before closing a month** (`ORDER_SLACK`). The listing is sorted by `createdAt` but paged by a cursor on the internal id. If the two orders ever disagree slightly near a month boundary, a late model still counts. The cost is a few extra pages.
- **The rate limit is 450 per 5 minutes,** under HF's 500. A sliding window of 450 can't put more than 450 into any fixed window. On a 429, the client waits the `t=` seconds from the `ratelimit` header (a `_retry_after` override).
- **The count is what exists now:** public models that haven't been deleted. Deleted models (largely spam) and models still private don't appear. So a backfilled month undercounts what was uploaded at the time. The series name says "(public and not deleted as of the count)", and the rows are flagged `reconstructed`. Live re-counts record any drift as new vintages, for example when spam is deleted.
- **Refactor: shared monthly-count ingestion** (`ai/monthly_ingest.py`). arXiv's backfill, live-row and CLI code would have been copied nearly line for line. Now each source module only defines a `MonthlySeries` and a client with `count_months`. See the addendum in `Task4.md`.

---

## What I did

### `ai/sources/huggingface_client.py`
- **`HuggingFaceClient(BaseAPIClient)`:**
  - `iter_created()` yields every model's `createdAt`, newest first.
  - `count_months(months)` runs one scan, and yields each month newest first as the scan passes it.
- **Errors:**
  - Asking for a month before April 2022 raises `ValueError` before any request is made.
  - A model without a valid `createdAt` raises `SourceAPIError`, as does invalid JSON.
- **`from_env` reads `HF_TOKEN`,** treating an empty value as unset. The token is sent as `Authorization: Bearer` only when set.

### `ai/monthly_ingest.py` (new, shared with arXiv)
- **`MonthlySeries`** holds the source, the series metadata, `first_month` and `live_months`.
- **`MonthlyCounter`** is a protocol: `count_months` plus `close`.
- **`ingest_backfill`** commits after each month. It only counts months that have no rows yet, so a rerun after an interruption picks up where it stopped.
- **`ingest_live`, `live_months` and `run_cli`** keep their behavior from Task 4.
- **`run_cli` checks its flags:** it rejects a `--start` before the series' `first_month`. It also closes the client when it finishes.

### `ai/months.py`
- `MonthlyCount` and the month helpers, moved out of the arXiv client.

### `ai/ingest_huggingface.py`
- The source is `Hugging Face Hub`. The series is `HF_NEW_MODELS` (Models, Monthly, domain `ai`), starting 2022-04.

### `tests/test_huggingface_client.py` (14 tests, not tagged `economy`)
- **Listing:**
  - Follows the cursor links. The first request carries the parameters; later ones use the link exactly as given.
  - Counts by UTC month, including a model at 23:59:59 on the last day.
  - Stops once it's past the earliest month: the 4th page is never fetched.
  - The 2-day slack catches a model listed out of order.
  - A listing that runs out still reports every month asked for, with zeros.
  - Pre-2022-04 months are refused with no request made.
  - A missing `createdAt` raises, as does invalid JSON.
- **Rate limit and token:**
  - The limiter is 450 per 300s.
  - A 429 waits the `t=187` from the header.
  - The token is sent only when set.
  - `from_env` reads `HF_TOKEN`, and treats an empty value as unset.
- **Ingestion (throwaway database):**
  - Backfill rows are reconstructed. Live rows record a month that dropped (models deleted since) and a new month.
  - The series is registered as `ai`, Models, under `Hugging Face Hub`.
  - **Interrupted backfill:** after a failure partway through the scan, the month already counted stays committed. The rerun counts only the missing months.
- **CLI:**
  - A `--start` before 2022-04 is rejected.
  - The dry run prints newest first, and closes the client.

## Tradeoffs
- **Survivorship bias.** Old months count only the models that still exist. Growth since 2022 is therefore overstated, by however much has been deleted from early months. Nothing in the API can fix that. Live vintages from now on are genuine.
- **Only about 54 months of history** (2022-04 to 2026-09). That is fine for 1-, 3- and 6-month horizons, but thin for long backtests or seasonality.
- **Every backfill lists everything since April 2022, and so does a resumed one.** Months already stored are skipped when writing, but the scan still has to page past newer months to reach them. There is no documented way to start the listing at a date.
- **A live pull lists about 4 months of models** (the current partial month plus 3 ended months), roughly 400–500 requests. That uses most of one 5-minute window, so allow about 5 minutes when scheduled.
- **No sharing of HF's window between processes.** Two runs close together, or a run right after the probes, can hit a 429. The client then waits out the reset, which happened during the backfill dry run below.

## Checks
- `uv run pytest`: 386 passed, exit 0. That is 317 economy, 16 schema, 25 base client, 14 arXiv and 14 Hugging Face. The task 1 safety net reported nothing missing or skipped.
- `ruff check`, `ruff format --check` and `mypy` pass on `src/eco_prediction/ai` and the AI tests.
- **Live dry run (real API, no database), 2026-10-12, `--months 1`:** 2026-09 had 92,008 models.
- **Full backfill dry run (real API, no database), 2026-10-12:** all 54 months, 2026-09 back to 2022-04, exit 0, in **34 minutes**.
  - About 3.07M models were counted, so about 3,100 pages.
  - **One 429** came early. Requests from the live dry run just before were still in HF's fixed window. The client waited the 180s reset and carried on.
  - Every month is nonzero. The smallest is 5,509 (2022-04).
  - Selected months:

    | Month | Models |
    |---|---|
    | 2022-04 | 5,509 |
    | 2023-01 | 14,659 |
    | 2024-01 | 37,633 |
    | 2025-01 | 99,924 |
    | 2025-07 | 72,804 |
    | 2026-01 | 122,550 |
    | 2026-09 | 92,008 |

  - **The series is not smooth.** It grew about 20× from 2022 to 2025, then went sideways at around 70–120k a month since 2025, with jumps such as 2026-01. Some of that is probably bulk uploads and spam that hasn't been deleted yet. That is worth keeping in mind when judging forecast errors on this series.

## Update 2026-10-16: pacing with a token
- `HF_TOKEN` is now in `.env` (a free account's fine-grained read token).
- **The client paces at 900 requests per 5 minutes with a token** (`TOKEN_MAX_CALLS`), and still 450 without one. It used to stay at 450 either way.
- **Hugging Face's API limits** (huggingface.co/docs/hub/rate-limits): 500 anonymous per IP, 1,000 for a free account's token.
- **The token's quota is separate from the IP's anonymous one.** That was checked from the `ratelimit` headers: the tokened request showed 990 left while the anonymous quota was at 420. So a tokened run doesn't slow an untokened one on the same machine.
- **README downloads count against the separate "resolvers" bucket** (3,000 / 5,000), not the API one.
