# AI Task 4: arXiv client and ARXIV_AI_SUBMISSIONS ingestion

Done on 2026-10-12. Nothing has been written to the live `eco_forecast` database yet. The backfill and the first live pull are the user's to run (see Quick reference).

## Quick reference

```sh
uv run python -m eco_prediction.ai.ingest_arxiv --backfill --dry-run   # print only, ~17 min
uv run python -m eco_prediction.ai.ingest_arxiv --backfill             # 2010-01 to last month
uv run python -m eco_prediction.ai.ingest_arxiv                        # live: last 3 ended months
uv run pytest tests/test_arxiv_client.py
```

---

## Decisions made before starting (2026-10-12)

### The arXiv search API, not OAI-PMH (user's choice)
- **The task said OAI-PMH, filtered by submission date, primary categories only.** OAI-PMH can't do that. Its `from`/`until` select records by **last modification**, not submission. In a probe, a 2011 paper (1102.1087) came back under 2024-01-02 because it had just been revised.
- Counting submissions through OAI-PMH would mean harvesting the whole `cs` set and counting locally: about 1M records, at ~3KB each roughly 3GB, taking hours at arXiv's rate limit.
- **Chosen:** one search-API query per month: `(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND submittedDate:[month]`, reading `opensearch:totalResults`.
  - About 200 requests for 2010 to now.
  - A paper in several of the categories is counted once.
  - `submittedDate` is the first version's date, so revisions don't move a paper's month.
- **This changes the series' meaning:** `cat:` matches cross-lists as well as the primary category. The series is "papers listed in cs.AI, cs.LG or cs.CL", not "papers whose primary category is one of them". The series name says so.
- **Sanity check against arXiv's own monthly listing pages:**

  | Category, month | Search API | Listing page |
  |---|---|---|
  | cs.CL, 2024-01 | 1,204 | 1,167 |
  | cs.LG, 2024-01 | 2,450 | 2,488 |
  | cs.AI, 2010-01 | 37 | 35 |
  | cs.LG, 2010-01 | 21 | 21 |
  | cs.CL, 2010-01 | 5 | 5 |

  Close, but not identical. The two count slightly different things (the listing goes by announcement/ID, the API by submission date). Repeated queries returned the same number.

### My defaults (stated, not asked)
- **Backfilled rows:** `as_of` = the first day of the next month, when the month's count was complete, with `reconstructed = true`.
  - Storing them with `as_of` = backfill date would hide all of history from backtests.
  - The value is **today's** count, not what was countable then: later cross-lists and withdrawals can move it a little. The `reconstructed` flag marks that.
- **Live rows:** each run re-counts the last 3 ended months. A row with `as_of` = today and `reconstructed = false` is stored for a month that is new, or whose count changed since its latest row. An unchanged count stores nothing.
- **The current month is never stored.** A half-month count would look like a collapse.
- **Backfill skips months that already have rows,** so an interrupted backfill resumes, and it never adds a reconstructed row under a month that has live rows. It commits one year at a time.

---

## What I did

### `ai/sources/arxiv_client.py`
- **`ArxivClient(BaseAPIClient)`:**
  - `count_submissions(month, categories)` sends one query.
  - `monthly_counts(start, end)` makes one call per month.
- **Errors:**
  - arXiv answers errors as an Atom feed with an entry whose id ends in `/api/errors`, sometimes with status 200. `parse_total_results` raises on those, on invalid XML, and on a feed with no `totalResults`.
  - `_error_message` pulls the error summary out of the feed.
- **`max_results=1`, not 0:** `max_results=0` gets a 500 from arXiv every time.
- **Rate limit: 1 request per 5s, not arXiv's documented 1 per 3s.**
  - At 3s, arXiv started answering 429 (no `Retry-After`) after about six requests. A probe found the block lasted about 4 minutes (235s).
  - Once it cleared, 30 requests at 10s and then 5s gaps all succeeded.
  - **Backoff:** starts at 15s and is capped at 300s, with 6 retries. The waits are 15, 30, 60, 120, 240 and 300s, about 13 minutes in all, which outlasts a 4-minute block. The 5xx errors use the same backoff.
- **Month helpers:** `month_start`, `next_month`, `iter_months` and `last_complete_month`.

### `ai/store.py` (shared by the coming AI sources)
- **`AISeriesInfo` and `AIObservation`** (the latter has a `reconstructed` field).
- **`upsert_ai_series`** inserts with `domain = 'ai'`. It raises if a series of that id already exists in another domain.
- **`insert_ai_observations`** writes `reconstructed`, with `ON CONFLICT DO NOTHING`.
- **Reuses the economy's `ensure_source` and `latest_stored`** by importing them. No economy code changed.

### `ai/ingest_arxiv.py`
- **Registers** the source `arXiv` and the series `ARXIV_AI_SUBMISSIONS` (Papers, Monthly, domain `ai`).
- **`ingest_backfill`** and **`ingest_live`** don't commit; `main` commits.
- **Refuses a count of 0.** These categories have had papers every month since 2010, so a 0 means the search index failed. The run errors out instead of storing it.
- **CLI flags:** `--backfill`, `--start` (default 2010-01-01), `--months` (default 3), and `--dry-run`. A dry run prints each row as it would be stored, and never opens a database connection.
- Errors exit with code 1.

### `tests/test_arxiv_client.py` (14 tests, not tagged `economy`)
- **Client:**
  - Query text and the month's boundaries, including a leap February.
  - Reads `totalResults`, and sends `max_results=1`.
  - One query per month.
  - An error feed with status 200 raises. One with status 500 is retried, and reported if it keeps failing.
  - Malformed feeds raise.
  - The 429 backoff schedule, and the 5s rate.
  - The month and window helpers.
- **Ingestion (throwaway database):**
  - Backfill rows are reconstructed, and as of the next month. `get_series_value_as_of` can't see January on Jan 31, and can on Feb 1.
  - The series is registered as `ai`, Papers, Monthly, under source `arXiv`.
  - Backfill skips stored months, so it resumes.
  - The live pull stores only new and changed months, and a rerun with the same counts stores nothing.
  - A zero count is refused.
- **CLI:** the dry run's output and that it never touches the database; an API failure exits 1.

## Where I changed the task's plan
- **The search API instead of OAI-PMH, and cross-lists included** (see Decisions).
- **No resumption-token pagination:** a count is one request. The paging the task asked for belongs to OAI-PMH.
- **`ai/store.py` added,** so the Hugging Face, GitHub and Epoch tasks don't each write their own `domain`/`reconstructed` SQL.

## Tradeoffs
- **Today's counts stand in for history.** Reconstructed values can differ slightly from what was countable at the time. Live rows from now on are genuine vintages.
- **The series depends on arXiv's search index.** If arXiv changes how `submittedDate` or `cat:` behave, the series shifts. The listing pages are an independent cross-check if it ever looks off.
- **A backfill is slow on purpose:** about 17 minutes at one request per 5s. Up to 13 more minutes go to each 429 block.
- **The live pull re-counts 3 months** (3 requests). Changes to older months after that aren't recorded as new vintages.

## Checks
- `uv run pytest`: 372 passed, exit 0. That is 317 economy, 16 schema, 25 base client and 14 arXiv. The task 1 safety net reported nothing missing or skipped.
- `ruff check`, `ruff format --check` and `mypy` pass on `src/eco_prediction/ai` and the new tests.
- **Live dry run (real arXiv, no database), 2026-10-12:** 2026-07 had 8,104 papers, 2026-08 had 9,109 and 2026-09 had 11,194.
- **First full backfill dry run, at the original 3s pace:** failed after about a year of months, on repeated 429s. That led to the 5s rate and the long backoff.
