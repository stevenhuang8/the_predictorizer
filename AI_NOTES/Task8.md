# AI Task 8: Polymarket client and platform tables (Metaculus deferred)

Done on 2026-10-14, for Polymarket. **Metaculus is deferred** (see below). Nothing has been written to the live `eco_forecast` database yet: migration 011, the first run and the daily runs are the user's to do (see Quick reference). No key is needed.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate                          # applies 011_platform_questions
uv run python -m eco_prediction.ai.ingest_polymarket --dry-run      # list markets, ~2 s
uv run python -m eco_prediction.ai.ingest_polymarket                # first run: ~2,500 histories, ~7 min
uv run python -m eco_prediction.ai.ingest_polymarket                # daily after that: ~1 min
uv run python -m eco_prediction.ai.ingest_polymarket --no-history   # questions + today's prices only
uv run pytest tests/test_polymarket_client.py
```

---

## What the APIs allow (probed 2026-10-14)

### Metaculus
- **Every request now needs a token** (`Authorization: Token ...`). Without one the API answers 403, and that includes its docs and OpenAPI schema. The spec is public in their GitHub repo (`Metaculus/metaculus`, `docs/openapi.yml`).
- **Its terms forbid this project's use without permission:** "data may not be used to train, **evaluate**, or otherwise create or develop AI/ML models or algorithms without Metaculus's prior written permission." Comparing our Claude forecaster with the community prediction is evaluating an AI model.
- **The community prediction is limited.** A normal token gets the *current* community prediction on only about 50 questions, with no history.
- **A "Bot Benchmarking" tier** covers about 250 open and 250 resolved questions, "intended for training and evaluation purposes". It's requested through their Data Needs Form, which is linked from the spec. Non-commercial research and hobby use is "generally supported".

### Polymarket
- **No key is needed.** `requests`' default User-Agent works, but Python's urllib one gets a 403.
- **Gamma `/events?tag_slug=ai`** is paged 100 at a time by `offset`, with open and closed events listed separately. The tag has 964 events with 10,083 markets. 10,062 of those are binary, and **2,528 have traded at least $10k**: 2,107 resolved (556 yes, 1,551 no), 1 resolved 50/50, and the rest open or awaiting resolution. The earliest opened in 2023-02.
- **A market's `outcomes`, `outcomePrices` and `clobTokenIds` are JSON strings.** Some inactive placeholder markets have no token ids at all (e.g. 570816, "Will GPT-5 be released by August 20?", volume 0).
- **CLOB `/prices-history?market=<token>&interval=max&fidelity=1440`** gives about one point a day since the market opened, closed markets included. Points land near 00:00 UTC.
- **The `ai` tag is broad.** Besides AI progress ("Gemini Argon HLE ≥ 55%?", "top AI model at the end of April?"), it covers company events (IPOs, acquisitions, CEO departures, outages). Task 10 does the curation.

## Decisions made before starting (2026-10-14)

### The user's choices
- **Metaculus is deferred, and Polymarket is built now.**
  - The user first chose "build it, add the token later". After the terms came to light, the user chose to defer it until permission is granted through the Data Needs Form (the Bot Benchmarking tier fits).
  - The tables already allow `platform = 'metaculus'`, so a Metaculus client only needs to produce `PlatformQuestion`s and `PlatformProbability`s.
- **New tables (migration 011), not `series`/`observations`.** Platform forecasts are benchmarks, not targets, and each needs its question text and resolution criteria.
- **Markets:** every binary Yes/No market in an AI-tagged event, open or closed, with **at least $10k traded** (`--min-volume`).
- **History:** one price a day for each market's full history, then live daily prices from there on.

### My defaults (stated, not asked)
- **Probability = the Yes token's price.**
  - Live prices come from Gamma's `outcomePrices[0]`, the price Polymarket shows.
  - History comes from the CLOB, using the last point of each UTC day.
- **`reconstructed`:**
  - **True** for history: backfilled from Polymarket's timestamped record, as with the other AI sources.
  - **False** for a price captured live on the day.
  - A day's live price is never overwritten by history.
- **History stops yesterday** (today is incomplete) **or on the close day.** Prices after the close are just the payout.
- **A market is resolved only when** it's closed, `umaResolutionStatus = resolved`, and its prices are exactly 1/0 (yes) or 0/1 (no).
  - Proposed or disputed resolutions and 50/50 outcomes stay unresolved (NULL). The raw status and prices are kept in `metadata`.
  - A resolution is refreshed on every run, so it follows any later change on Polymarket.
- **`prices_through` says how far each market's prices are stored without gaps,** from history or live. It drives which histories are fetched:
  - The first run fetches all ~2,500.
  - A daily run fetches none for open markets, because yesterday's live price already covers them. It only fetches for new markets, newly closed ones, and markets left with a gap by a missed day.
- **One command does everything** (upsert markets, then history, then live prices), so it can run daily. Each market is committed on its own, so an interrupted first run resumes where it stopped.
- **No `sources` row.** The `platform` column is the source, and `sources` only describes `series`.
- **The client keeps to 60 requests per 10 s** across both hosts. Polymarket publishes no limit for these endpoints.
- **Markets without token ids are skipped** rather than failing the run. They were never tradable and have no prices. A missing id or question, or malformed JSON, still raises.

---

## What I did

### `migrations/011_platform_questions.sql`
- **`platform_questions`:**
  - Columns: platform (`polymarket` | `metaculus`), external_id, title, resolution_criteria, url, opened_at, closes_at, resolved_at, resolved_outcome (`yes` | `no` | NULL), prices_through, metadata (JSONB), first_seen_at and updated_at.
  - `UNIQUE (platform, external_id)`, and an outcome needs a `resolved_at`.
- **`platform_probabilities`:**
  - Columns: question_id, as_of, probability (0 to 1) and reconstructed.
  - The primary key is `(question_id, as_of)`, one price a day.
- **Purely additive:** no existing table is touched.

### `ai/platform_store.py`
- **`PlatformQuestion` and `PlatformProbability`** can be shared by every platform.
- **`upsert_platform_question`** returns (id, prices_through, whether it's new).
- **`insert_platform_probabilities`** keeps existing days.
- **`set_prices_through`** records how far prices are stored.

### `ai/sources/polymarket_client.py`
- **`PolymarketClient(BaseAPIClient)`:**
  - `iter_events` pages both open and closed events.
  - `ai_markets(min_volume)` gives binary markets, deduplicated, in listing order.
  - `price_history(token)` reads the CLOB.
- **`parse_market`** turns raw JSON into a `Market`, with the outcome rules above.
- **`daily_prices(history, before)`** gives the last price of each UTC day.

### `ai/ingest_polymarket.py`
- **`ingest(conn, client, today)`** runs the three steps.
- **The CLI** takes `--min-volume`, `--no-history` and `--dry-run`. A dry run lists markets only, with no history and no database.

### Tests
- **`tests/test_polymarket_client.py`: 27 tests, not tagged `economy`.**
  - **Parsing:**
    - Open and resolved (yes, no, 50/50) markets.
    - A proposed resolution isn't an outcome yet.
    - Non-binary and tokenless markets are skipped.
    - Malformed JSON and missing ids raise.
  - **Listing:**
    - Paging by offset, open then closed.
    - Deduplication.
    - The volume filter.
  - **History:**
    - Parsed and sorted, the CLOB URL and params.
    - A bad body raises.
    - The last price of each UTC day, today excluded.
    - History ends yesterday or on the close day.
  - **Ingestion (throwaway database):**
    - The first run stores questions, history and live prices.
    - The next day fetches no history.
    - A missed day fetches only the gap.
    - A new resolution updates the question.
    - Table constraints.
  - **CLI:**
    - The dry run fetches no history.
    - A negative volume is rejected.
    - API errors exit with code 1.
- **`tests/test_ai_schema.py` (one-line change):** migrating a database at 008 used to assert that exactly `[9, 10]` were applied. It now checks that they come first, so 011 and later migrations don't break it.

## Where I changed the task's plan
- **No Metaculus client** (deferred: terms and access). There's no `metaculus_client.py` or `test_metaculus_client.py` yet.
- **`ai/ingest_polymarket.py` instead of `ai/ingest_forecasting_platforms.py`.** Each platform gets its own command, as the other sources do.
- **Storage is two new tables,** as the task suggested ("may need new table").
- **Question text and resolution criteria live in `platform_questions`,** not in `questions`. Task 10 decides which of them become our own binary questions.
- **No `sources` rows** for the platforms.

## Tradeoffs
- **Selection bias:** markets are chosen by today's volume. A market that was thin when a forecast would have been made is still included if it got busy later. Task 10 or the benchmark comparison can use the volume as of the forecast date if this matters.
- **Daily granularity.** A daily price is the last trade-based point of the UTC day. Intraday moves (e.g. around an announcement) are lost.
- **Live prices and history differ slightly in kind.** The live price is Gamma's displayed price, and history is the CLOB's series. On a day with both, the live price wins.
- **Question text is refreshed on every run.** Polymarket sometimes clarifies descriptions, and only the latest text is kept. Point-in-time retrieval (task 16/17) shouldn't treat it as known at the market's opening.
- **The `ai` tag is noisy.** About a third of the markets are corporate events, not AI progress.
- **Without Metaculus there's one benchmark, not two,** until access is granted.
