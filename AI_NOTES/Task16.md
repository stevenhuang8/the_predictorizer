# AI Task 16: LLM call log and retrieval corpus

Done on 2026-10-16. Migration 014 was applied to the live database by the user the same day. The 2,538 Polymarket markets were indexed into the live corpus that day (published 2023-02-16 to 2026-10-16). `llm_calls` is empty until Task 18.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate --status   # live: at 014 since 2026-10-16
uv run pytest tests/test_llm_tables.py
```

```python
from eco_prediction.ai.llm.calls import LLMCall, attach_forecast, cached_call, context_hash, log_call
from eco_prediction.ai.llm.retrieval import index_platform_questions, retrieve_documents

index_platform_questions(conn); conn.commit()                  # 2,538 markets, < 1 s
retrieve_documents(conn, date(2026, 10, 14), "polymarket_question",
                   query="best AI model end of year")          # published by end of the 14th
```

---

## Decisions (my defaults, not asked)

### Migration `014_llm_tables.sql`, not `010`
- 010–013 were already taken, so this is 014.
- **Purely additive:** two new tables, and no existing table or constraint is touched.

### `llm_calls`: the task's columns plus what Tasks 18–19 need
| Added | Why |
|---|---|
| `question_id` (FK, cascade) | A call is made before its forecast exists, and a backtest may store no forecast. `forecast_id` is filled in later with `attach_forecast`. |
| `prompt_version`, `as_of_date` | Which prompt template, and which data cutoff, produced the call. Task 19's leakage audit needs both. |
| `parsed_response` (JSONB), `error` | Failed calls are logged too, so failure rates and wasted spend can be counted. |
| `cache_read_tokens`, `cache_write_tokens` | Prompt-cached tokens are billed differently. Without them, `cost_usd` can't be recomputed from the row. |

**Constraints:**
- `raw_response` is nullable, but a row must have a response or an error.
- `context_hash` must be 64 lowercase hex characters.
- Token counts and cost can't be negative.

### `context_hash` deduplicates; it isn't unique
- **What it hashes:** SHA-256 of the model, the prompt and the request parameters, as canonical JSON with key order ignored.
- **`cached_call(conn, model, hash)`** returns the latest successful call with that hash. A backtest re-run can reuse it instead of paying again.
- **Why not unique:** sampling the same prompt twice on purpose is legitimate, so the hash is indexed, not constrained.

### `retrieval_documents`: the task's columns plus `title`, `updated_at` and a search column
- **`search`** is a generated `tsvector` (English) over the title and content, with a GIN index. Postgres maintains it, so there's no external index to keep in sync.
- **`metadata`** is `NOT NULL DEFAULT '{}'` and must be an object, like `model_versions.parameters`.
- **Upserts only touch rows whose text, date or metadata changed.** A re-index reports inserted / updated / unchanged, and `updated_at` means something.

### Point in time: "published by the end of `as_of_date`, UTC"
- **The rule:** `published_at < as_of_date + 1 day` at 00:00 UTC, the harness's cutoff convention. Pass the day before the forecast.
- **Hindsight caveat:** retrieval trusts `published_at`. A document indexed today with an old date shows its *current* text as of that date. A revised arXiv abstract (v2) would leak a little, the same caveat as the reconstructed observations. Task 17's indexers should date documents by first version where the source allows it.

### Retrieval matches any word by default
- **`match="any"`** ORs the query's stemmed words and ranks by `ts_rank_cd`, so documents sharing the most words come first.
- **`match="all"`** is web-search syntax (every word required, plus `"phrases"`, `or` and `-word`).
- **Why "any" is the default:** I first used web-search syntax only. On the real markets, "best AI model end of year" found nothing, because no title says "year". Task 18 will build queries from question text, which needs the forgiving mode.
- **Rejected:** embeddings / pgvector. That means a new extension and an embedding API cost. Full text is enough for titles and abstracts at this size. Revisit if Task 18's context looks poor.

### Only one indexer here: Polymarket markets
- **Why only one:** it's the only source whose text is already stored. arXiv abstracts, HF model cards and GitHub release notes need new API fetches, which are Task 17's job. I didn't add stub indexers for them.
- **`index_platform_questions`** works like this:
  - A document is a market's title plus its resolution criteria.
  - It's dated `opened_at`, or `first_seen_at` for the 4 markets with no open date. That's later than the truth, so it's safe.
  - The type is `polymarket_question`; the task said `metaculus_question`, which is deferred.
  - **Resolution and prices are left out on purpose.** They're outcomes and signals with their own dates. Putting them in document text would leak the answer to any question about that market. Metadata carries only the URL, the scheduled close and the category.

---

## What I did
- **`migrations/014_llm_tables.sql` (new):** `llm_calls` and `retrieval_documents`, with indexes.
- **`src/eco_prediction/ai/llm/` (new package):**
  - `calls.py`: `LLMCall`, `LoggedCall`, `context_hash`, `log_call`, `cached_call` and `attach_forecast`.
  - `retrieval.py`: `Document`, `UpsertResult`, `upsert_documents`, `retrieve_documents`, `end_of_day` and `index_platform_questions`.
- **`tests/test_llm_tables.py`: 8 tests, not tagged `economy`, each on a fresh migrated database.**
  - **Context hash:** stable under key order, and changes with the model, the prompt or the parameters.
  - **Calls:**
    - A call is logged with all its fields, then linked to its forecast.
    - Deleting the forecast cascades to its calls.
    - A missing forecast, a bad hash, and a row with neither response nor error are rejected. A failed call is logged.
  - **Cache:** it returns the latest *successful* call for the same model and hash. It skips errors, other models and other prompts.
  - **Upserts:** they count inserts, updates and unchanged rows. A duplicate key in one batch is fine, and a naive datetime is rejected.
  - **Point in time:** a document published at 00:00 UTC on 03-01 is hidden as of 02-29 and shown as of 03-01. Results come newest first.
  - **Querying:**
    - Type filters (one or several).
    - Stemming.
    - Ranking in both modes.
    - Exclusion (`-word`).
    - Any vs. all.
    - Stop-word-only queries.
    - `limit` and `since`.
  - **Polymarket indexing:**
    - A resolved market's document has its title and criteria, but no outcome, resolution date or volume.
    - It's dated at its opening.
    - Re-indexing is a no-op.
    - A market with no open date gets `first_seen_at`.

## Checks
- `uv run pytest`: **595 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **On real data, in a throwaway database** (migrated to 014, with the live database's 2,538 Polymarket markets copied in, then dropped):
  - Indexing takes 0.9 s; a re-index takes 0.6 s and reports 2,538 unchanged.
  - Queries take 1–4 ms, and the GIN index executes in 0.3 ms.
  - "OpenAI GPT-5 release" as of 2025-06-01 returns only markets opened by then: the GPT-5 release-date markets, plus two GPT-4.5 ones.
  - "SWE-bench" finds nothing, which is correct: no market in the set mentions it.

## Where I changed the task's plan
- **Numbered 014, with extra columns** (above).
- **`retrieve_documents` adds `query`, `match` and `since`.** `query_type` is the document-type filter (one or several).
- **Only the Polymarket indexer.** arXiv, HF and GitHub are Task 17.
- **`llm_calls` is in its own module (`calls.py`),** not `retrieval.py`.

## Tradeoffs
- **Lexical, not semantic.** "frontier model launch" won't find "GPT-5 released". Task 18's query builder should use the question's own words, which carry the entity names.
- **Ranking ignores recency except as a tie-break.** It also treats "GPT-4.5" and "GPT-5" as near-equal for a GPT-5 query, because both share the "gpt" token. `since`, plus asking for more than you need and filtering, are the levers for now.
- **`published_at` is trusted** (above). The point-in-time guarantee is only as good as each indexer's dates.
- **Prompts are stored in full.** That's fine at this scale, but long prompts with ten documents each make `llm_calls` the biggest table quickly (about 20 KB a call). Storing document ids instead of text was rejected, because a logged prompt should be reproducible on its own.
- **Long texts outrank close titles.** On the live corpus, "best AI model end of year" ranked "US Government removes public access to … AI model in 2026?" above the "best AI model at the end of 2025" markets. Their long resolution criteria repeat the query words. The fix is to weight titles above content (`setweight`) and normalize `ts_rank_cd` by length. That changes the generated column, so it needs a new migration. Worth doing in Task 17 or 18 once the arXiv text is in.
