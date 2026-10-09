# AI Domain + LLM Forecaster: Spec

Status: planning, no code yet
Last updated: 2026-10-09

## Goal

Extend the forecasting system (already live for the US economy) to a second domain: **the direction of AI**. Add an **LLM forecaster** (Claude) as a new model type. It competes with the existing statistical and ML models under the same scoring rules.

The existing principles all still apply:
- every forecast is resolvable
- inputs are frozen at prediction time
- forecasts are append-only
- data is stored point-in-time
- models must beat baselines
- the system is judged on calibration, not single misses

## Principles specific to this phase

- **The economy pipeline must keep working unchanged.** This domain is purely additive. Existing tables, jobs, CLI commands and dashboard pages keep their current behavior by default. The existing pytest suite must stay green after every task.
- **Reconstructed vintages are labeled.** Most AI sources do not keep revision history the way ALFRED does. Live pulls are snapshotted with `as_of` = fetch date. Historical backfills are built only from timestamped records, such as arXiv submission dates or Epoch publication dates, and are flagged `reconstructed = true`.
- **The LLM must not see the future.** The LLM forecaster sees only a point-in-time context bundle. Backtests exclude any question that resolves before the LLM's training cutoff, because the model may have memorized the answer.
- **The LLM's work is reproducible.** The exact prompt, model ID, parameters and raw response are stored with every LLM forecast.

## Scope

### Numeric targets (point + 80% interval, monthly, horizons 1/3/6 months)

| Target | Source | Notes |
|---|---|---|
| arXiv submissions (cs.AI + cs.LG + cs.CL, monthly count) | arXiv API / OAI-PMH | Dense history back to ~2010 |
| Hugging Face new public models per month | Hugging Face Hub API | Backfill from model `createdAt` |
| GitHub AI activity (new repos / stars on AI topics per month) | GitHub Search API | Rate-limited; snapshot monthly |
| Benchmark SOTA score (SWE-bench Verified, GPQA Diamond, ARC-AGI) | Epoch AI Benchmarking Hub | Step series; only moves when a new SOTA appears |
| Largest training run to date (log10 FLOPs) | Epoch AI notable models dataset | Step series |

### Binary targets (probability, event questions with a resolution date)

- Examples:
  - "Will SOTA on benchmark X exceed Y% by date D?"
  - "Will a model above 1e27 FLOPs be reported by date D?"
  - "Will lab X release model Y by date D?"
- Questions come from two places:
  - auto-generated thresholds on the numeric series
  - curated questions mirrored from Metaculus or Polymarket
- Market/community benchmark: the Metaculus community prediction and the Polymarket price, both stored point-in-time.
- Horizons are given in days until the resolution date.

### Non-goals for this phase

- Forecasting stock prices of AI companies
- User-submitted questions
- Autonomous agents that browse the live web during backtests. Retrieval comes only from the stored, timestamped corpus.

## Data Sources

arXiv API, Hugging Face Hub API, GitHub API, Epoch AI datasets (benchmarks, notable models), Metaculus API, Polymarket API. Optional: Semantic Scholar (citation counts).

Each source gets a client with rate limiting and retry, following the pattern of `data/fred_client.py`. Each source and series is registered in the existing `sources` / `series` tables. All values go into `observations` with `as_of`.

## Schema Changes (migration `009_ai_domain.sql` and later)

Today's schema only fits the economy:
- `forecast_target` is a hard-coded enum (`cpi_yoy`, `unemployment`, `fomc_decision`)
- `questions` has a CHECK that ties `probability` to `fomc_decision`
- `resolutions.actual_outcome` only allows cut/hold/hike

The changes are additive only:
- Add a `domain` column (`economy` | `ai`, `DEFAULT 'economy'`) to `questions` and `series`.
- Add the new targets to the `forecast_target` enum (`ALTER TYPE ... ADD VALUE`).
- Add a `binary` question type. Its resolution is a `yes`/`no` outcome.
- Relax the FOMC-only CHECK constraints so the economy rows stay valid.
- Add a nullable `horizon_days` and a `question_text` column for event questions.
- Add an `observations.reconstructed BOOLEAN DEFAULT FALSE` flag.
- Add an `llm_calls` table: forecast_id, model_id, prompt, context_hash, raw_response, input/output tokens, created_at.
- Add a `retrieval_documents` table holding the timestamped text corpus (arXiv abstracts, release notes, Metaculus question text) with `published_at`, so the LLM can do point-in-time retrieval.

## Modeling Approach

1. **Baselines:**
   - numeric: random walk, historical mean, linear trend
   - binary: base rate, community/market forecast
2. **Statistical:** reuse ARIMA/ETS from `models/statistical.py`, trying them on a log scale for the count series.
3. **ML:** LightGBM on AI features (lags, growth rates, cross-series signals such as the HF upload trend for arXiv).
4. **LLM forecaster (Claude, via the `anthropic` SDK):**
   - Input is a context bundle containing:
     - the question
     - the resolution rule
     - recent series values as of the forecast date
     - the stats-model forecasts
     - the top-k retrieved documents with `published_at <= forecast_date`
   - Output is structured JSON: point + 80% interval, or a probability, plus a rationale.
   - The default model is Opus. A cheaper Haiku variant is used for bulk backtests.
   - Variants:
     - an LLM-only forecaster
     - an LLM ensemble that combines the LLM with the stats models, with weights learned walk-forward
5. **Calibration:** reuse `models/calibration.py`: conformal intervals for numeric targets, isotonic/Platt scaling for binary.
6. **Explanations:**
   - SHAP for LightGBM
   - the stored rationale for the LLM

Validation is walk-forward only. The LLM's backtest window is limited to dates after its training cutoff. Report this as a separate, smaller evaluation set.

## Post-mortems

Reuse the 4 existing categories: bad data, bad model, regime change, variance. Add an LLM-drafted miss explanation, stored as a draft note that a human confirms.

## Success Criteria

- Count series: the best model beats the random walk on RMSE at the 3-month horizon.
- Binary questions: Brier score close to the Metaculus/Polymarket benchmark.
- The LLM forecaster beats the base-rate baseline after its training cutoff. If it does not, it is reported as such and kept out of live use.
- 80% intervals cover about 80% of outcomes.
- The economy test suite and the live economy job are unaffected.

## Build Phases

### Phase 1: Foundation
- Regression safety net: snapshot the current economy test results, and run them in every task
- Additive schema migration 009 (domain, new targets, binary type, reconstructed flag)
- `src/eco_prediction/ai/` package skeleton and shared HTTP client utilities

### Phase 2: Data
- arXiv client + monthly-count ingestion + backfill
- Hugging Face Hub client + ingestion + backfill
- GitHub client + monthly snapshot ingestion
- Epoch AI benchmark + notable-models ingestion (SOTA and compute step series)
- Metaculus / Polymarket clients: question mirroring and point-in-time community probabilities

### Phase 3: Questions + models
- AI question generator (numeric questions per horizon, threshold-based binary questions, curated binary questions)
- Binary resolution support in `db/resolutions.py`, and Brier scoring of binary questions in metrics
- AI feature engineering
- Run baselines, stats and LightGBM through the walk-forward harness on the AI series

### Phase 4: LLM forecaster
- `llm_calls` and `retrieval_documents` tables, and the point-in-time retrieval corpus
- Claude forecaster with structured output, prompt versioning and cost tracking
- Leakage-safe LLM backtest (only after the training cutoff), plus the LLM-ensemble model
- LLM model comparison report against the baselines and the market

### Phase 5: Live loop + UI
- AI forecast job + AI resolution job (separate entry points; `forecast_job.py` is not changed)
- LLM-drafted post-mortem notes
- `eco-forecast` CLI: `--domain ai` subcommands
- Next.js dashboard: AI section (open questions, LLM rationale, track record vs market)

## Implementation Notes (for task generation)

- Python project managed with uv (pyproject.toml, src/ layout, pytest, ruff, mypy). Postgres via docker-compose. Raw SQL migrations in `migrations/`, applied by `db/migrate.py`.
- Reuse the existing modules:
  - `backtest/harness.py`, `backtest/metrics.py`, `backtest/data.py`
  - `models/baselines.py`, `models/statistical.py`, `models/lightgbm_model.py`, `models/calibration.py`, `models/explainability.py`
  - `db/forecasts.py`, `db/resolutions.py`, `db/postmortems.py`, `postmortem/classifier.py`
- New code goes under `src/eco_prediction/ai/`: `sources/`, `features.py`, `questions.py`, `llm/`, `jobs/`.
- Shared files that need small, guarded edits:
  - Python: `db/forecasts.py`, `db/resolutions.py`, `scheduler/resolution_job.py`, `backtest/metrics.py`, `cli.py`
  - Web: `web/lib/dashboard-data.ts`, `web/lib/track-record-data.ts`
  - Every change must default to the current economy behavior, and must be covered by the existing tests plus new ones.
- The Anthropic API key goes in `.env.local` as `ANTHROPIC_API_KEY`. Tests mock the LLM; no network calls in pytest.
- The first task should be the regression safety net, and every later task depends on it directly or through other tasks.
