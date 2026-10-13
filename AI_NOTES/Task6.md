# AI Task 6: GitHub client, GITHUB_AI_ACTIVITY and GITHUB_AI_STARS

Done on 2026-10-12. Nothing has been written to the live `eco_forecast` database yet. The backfill, live pulls and the first stars snapshot are the user's to run (see Quick reference). `GITHUB_TOKEN` isn't in `.env` yet; everything below was checked without one.

## Quick reference

```sh
# GITHUB_TOKEN=... in .env: a fine-grained PAT with no extra permissions is enough
uv run python -m eco_prediction.ai.ingest_github --backfill --dry-run   # print only
uv run python -m eco_prediction.ai.ingest_github --backfill             # 2017-01 to last month
uv run python -m eco_prediction.ai.ingest_github                        # live: last 3 ended months
uv run python -m eco_prediction.ai.ingest_github stars                  # this month's stars snapshot
uv run pytest tests/test_github_client.py
```

---

## What the API allows (probed 2026-10-12)
- **`total_count` answers "how many repos match" in one request.** For Jan 2024: `topic:ai` gave 1,044 and `topic:deep-learning` gave 786.
- **OR across `topic:` qualifiers is rejected** (422 "Validation Failed"). Several `topic:` qualifiers in one query mean AND: all three topics together gave 19 for Jan 2024, far fewer than any one topic.
- **Rate limit:**
  - Search allows 10 requests a minute without a token and 30 with one.
  - When it runs out, GitHub answers **403** (not 429) with `X-RateLimit-Remaining: 0`, and `X-RateLimit-Reset` gives the reset time in epoch seconds.
  - This IP kept running out during the probes. Unauthenticated searches are tight.
- **Search returns at most 1,000 results per query,** and at most 100 per page.
- **Sorting by stars works.** For `topic:machine-learning` the top three were tensorflow (200,905), prompts.chat (172,770) and transformers (167,615).

## Decisions made before starting (2026-10-12)

### The user's choices
- **Two series** (the user chose "New repos + stars snapshot"):
  - `GITHUB_AI_ACTIVITY` is the forecast target `github_ai_activity`: new AI repos per month.
  - `GITHUB_AI_STARS` is a separate series with a live monthly snapshot only, never backfilled.
    - GitHub only has today's star counts. A backfill of "stars per month" would really be today's stars on repos created that month, which mostly measures how old the repos are.
- **Topics: `ai`, `machine-learning` and `deep-learning`, deduplicated.** Each repo counts once, by inclusion-exclusion over every non-empty subset of the topics: 7 count queries per month.
  - **Rejected:** adding `llm` and `generative-ai`, either summed (double counting) or deduplicated (31 queries per month).
- **`GITHUB_TOKEN`** is read from `.env`. It's optional: without it, searches drop to 10 a minute.
  - The task said `.env.local`. Everything in the project loads `.env` through `load_dotenv()`, so I used that.

### My defaults (stated, not asked)
- **History starts 2017-01** (`FIRST_MONTH`), when GitHub added topics. Earlier months would mostly count old repos that had topics added later.
- **Stars snapshot:**
  - One per calendar month, stored under the month it was taken in (`observed_at` = first of the month, `as_of` = the run date).
  - A second run in the same month does nothing, without searching. Running it daily is cheap.
- **The top 1,000 by stars, merged across topics:** each topic's top 1,000 (10 pages of 100) is fetched, the repos are merged by id, and the 1,000 with the most stars are kept and summed.
  - The overall top 1,000 is always inside the three per-topic lists. A repo in the overall top 1,000 has some topic *t*, and every repo above it with topic *t* is also above it overall, so it's in *t*'s top 1,000.
  - A snapshot is about 30 searches.
- **Rate-limited 403s wait for the reset.** A 403 with `X-RateLimit-Remaining: 0`, or a 429, waits until `X-RateLimit-Reset` plus a second. `Retry-After`, for secondary limits, wins when present. Any other 403 (for example, no access) raises at once.
- **Incomplete results are never stored.** When GitHub's search times out, it returns `incomplete_results: true` with a partial count. The client retries 3 times with backoff, then raises.

---

## What I did

### `ai/sources/base_client.py` (small change)
- **A new `_rate_limit()` hook** sets the default limiter's (max calls, period). The default is still `MAX_CALLS`/`PERIOD`. GitHub uses it to pick 27 or 9 depending on the token (originally 30 or 10; see "With the token").
- **The rate-limit log now reports the limiter's own numbers.** The 25 base-client tests still pass.

### `ai/sources/github_client.py`
- **`GitHubClient(BaseAPIClient)`:**
  - `search_repositories` does one page, retried while results are incomplete.
  - `count_repos` returns `total_count`.
  - `count_new_repos(month)` does the inclusion-exclusion.
  - `count_months` is the `MonthlyCounter` interface.
  - `top_starred` and `top_starred_total` handle the stars.
- **Headers:** `Accept: application/vnd.github+json` and `X-GitHub-Api-Version: 2022-11-28`, plus `Authorization: Bearer` when there's a token.
- **Forks are excluded,** because search leaves them out by default. Only public repos are counted.
- **A wall clock is injectable** (`now`), so the reset wait can be tested.

### `ai/ingest_github.py`
- **The default command** runs the shared monthly CLI for `GITHUB_AI_ACTIVITY` (Repositories, from 2017-01).
- **`stars [--dry-run]`** takes the `GITHUB_AI_STARS` snapshot (Stars, Monthly).
- **Both series are under one source,** `GitHub`, and both are domain `ai`.
- **A zero is refused** for either series.

### `tests/test_github_client.py` (17 tests, not tagged `economy`)
- **New repos:**
  - The query helpers.
  - The 7-query inclusion-exclusion, with the arithmetic checked against a table.
  - The token and API-version headers.
  - `count_months`.
  - Pre-2017 months are refused with no request made.
- **Errors:**
  - Incomplete results are retried, then refused after 4 attempts.
  - A 422 raises with GitHub's message.
  - A rate-limited 403 waits until the reset: 42s plus 1.
  - `Retry-After` wins on a 429.
  - Any other 403 isn't retried.
  - The client's rate is 27 with a token and 9 without.
  - `GITHUB_TOKEN` is read from the environment, and an empty value is treated as unset.
- **Stars:**
  - The merge doesn't double count.
  - A short page ends a topic.
  - Full pages are followed, 100 per page.
  - Asking for more than 1,000 is refused.
- **Ingestion (throwaway database):**
  - Activity backfill rows are reconstructed, in domain `ai`.
  - One stars snapshot per month: a second October run doesn't search, and November adds a row.
  - Both series share one `GitHub` source.
  - Zero stars are refused.
- **CLI:** `stars --dry-run` and the activity dry run route correctly, without touching the database.

## Where I changed the task's plan
- **The topic query:** the task's `topic:ai OR topic:machine-learning OR ...` is rejected by GitHub. It was replaced by inclusion-exclusion.
- **Two series instead of one,** and stars aren't backfilled (the user's choice).
- **The token comes from `.env`, not `.env.local`.**
- **No result-paging for counts:** `total_count` is exact, so the 1,000-result cap only matters for the stars snapshot.

## Tradeoffs
- **Reconstructed counts drift.** A repo counts if it carries a topic *now* and still exists. Topics added later raise past months, and deleted repos lower them. Live re-counts record the drift as new vintages.
- **Topic tagging is voluntary.** Most AI repos have no topics at all, so the series tracks tagged repos, a stable-ish fraction. Newer vocabulary (`llm`, `generative-ai`) is missed unless a repo also uses one of the three topics.
- **The stars series is dominated by a few huge repos** (tensorflow alone is about 200k). Monthly changes mostly reflect the top repos' growth and repos entering or leaving the top 1,000. It has no history until snapshots build up, about one point a month.
- **Costs:**

  | Run | Searches | With a token | Without |
  |---|---|---|---|
  | Backfill, 2017-01 to 2026-09 | 117 months × 7 = 819 | ~31 min | ~91 min |
  | Live pull | 21 | under a minute | ~2 min |
  | Stars snapshot | ~30 | ~1 min | ~3 min |

## Checks
- `uv run pytest`: 403 passed, exit 0. That is 317 economy, 16 schema, 25 base client, 14 arXiv, 14 Hugging Face and 17 GitHub. The task 1 safety net reported nothing missing or skipped.
- `ruff check`, `ruff format --check` and `mypy` pass on `src/eco_prediction/ai` and the AI tests.
- **Real-API dry runs, no token, no database (2026-10-12):**
  - `--dry-run --months 1`: **2026-09 had 20,365** new AI repos. That took 7 searches, with exit 0.
  - `stars --dry-run`: **21,510,135** stars on the top 1,000. It hit one rate-limited 403, waited out the reset (about 53s), and finished with exit 0.
  - **Cross-check through the client:** 2024-01 had 3,292 and 2025-01 had 4,525.
- **The recent jump is worth watching:** about 4.5k a month in early 2025, 20k in September 2026. Part of it may be real (the agent-tooling boom). Part may be that new repos haven't yet been deleted or cleaned up the way older ones have. That cleanup is the drift the live re-counts will record. The full backfill wasn't dry-run here: without a token it takes about 80 minutes. Run `--backfill --dry-run` once the token is in place to see the whole curve.

## With the token (2026-10-12)
- **The token loads and works.** `GITHUB_TOKEN` is in `.env`, and GitHub's `/rate_limit` reports a search limit of 30 a minute.
- **Full backfill dry run (real API, no database):** all 117 months, 2017-01 to 2026-09, in **28 minutes**, exit 0. Every month is nonzero; the smallest is 716 (2017-01).

  | Month | New AI repos |
  |---|---|
  | 2017-01 | 716 |
  | 2018-01 | 1,560 |
  | 2019-01 | 1,623 |
  | 2020-07 | 2,478 |
  | 2022-01 | 2,522 |
  | 2023-01 | 3,079 |
  | 2024-01 | 3,292 |
  | 2025-01 | 4,525 |
  | 2025-07 | 7,791 |
  | 2026-01 | 10,950 |
  | 2026-07 | 25,539 |
  | 2026-09 | 20,365 |

  - **Steady growth to 2024, then a sharp acceleration from 2025.** The caveat above about recent months not yet being cleaned up applies most to 2026. The series will be hard to forecast across that break.
- **The run hit the rate limit 27 times** out of 819 searches, even though the client paced at exactly 30 a minute. GitHub's window and the client's sliding window don't line up exactly. Each 403 waited for the reset, and nothing was lost.
  - **Changed:** the client now keeps to **27 a minute with a token and 9 without** (90% of the limits), so this doesn't happen. The rate test was updated. The backfill wasn't rerun at the new pace: it would take about 31 minutes instead of 28.
