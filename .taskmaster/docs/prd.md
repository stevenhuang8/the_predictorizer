# Trend Forecasting App: Spec

Status: planning, no code yet
Last updated: 2026-09-28

## Goal

Build a forecasting system that makes reliable, calibrated predictions from real data, and can explain why a forecast was wrong when it misses.

"Reliable" means calibrated: when the system says 70%, those events happen about 70% of the time. Every model must beat simple baselines to earn its place.

## Principles

- Every prediction is a resolvable forecast: a specific question, a probability or interval, a resolution date, and a rationale.
- Freeze everything at prediction time: input data as it existed that day, model version, feature contributions.
- Forecasts are append-only. Never update a row; revisions are new rows linked to the previous one.
- Store data vintages. Economic data gets revised, and backtests must only use what was known at the time.
- Not every miss is a mistake. Judge the system on calibration over many forecasts, not single outcomes.

## V1 Scope: US Economy

### Targets

| Target | Type | Frequency | Market/expert benchmark |
|---|---|---|---|
| CPI inflation (YoY) | Numeric + 80% interval | Monthly | Cleveland Fed nowcast |
| Unemployment rate | Numeric + 80% interval | Monthly | Survey of Professional Forecasters |
| FOMC rate decision | Probability: cut / hold / hike | 8x per year | CME FedWatch, Kalshi |

### Horizons

1, 3, and 6 months ahead.

### Features (FRED / ALFRED)

- Yield curve spreads (10y-2y, 10y-3m)
- Oil and gasoline prices
- PPI
- Wage growth (Atlanta Fed tracker)
- Initial jobless claims
- Michigan consumer inflation expectations
- Rent indices (lead the CPI shelter component)
- CPI components (shelter, energy, food, core goods) for post-mortem attribution

### Non-goals for v1

- Stock or market predictions
- Non-US economies
- Recession prediction (resolution too slow, too few historical examples)
- LLM agent reasoning layer
- AI-direction domain
- User-submitted forecasts

## Success Criteria

- CPI and unemployment: beat the random walk baseline on RMSE at the 3-month horizon in walk-forward backtest.
- FOMC: Brier score close to market-implied odds. Consistently beating the market is not expected.
- 80% intervals contain the actual value roughly 80% of the time.

## Tech Stack

- **Data + modeling:** Python (pandas, statsforecast, LightGBM, SHAP, scikit-learn for calibration)
- **Database:** self-hosted PostgreSQL in Docker, hand-written schema, raw SQL migrations (learning goal: manage connections, indexes, backups directly instead of using Supabase)
- **Frontend:** Next.js + TypeScript (later phase)
- **Scheduling:** TBD (cron vs Trigger.dev vs Inngest)

## Data Sources

**V1 (economy):** FRED, ALFRED (vintages), BLS, BEA, Treasury, Cleveland Fed nowcast, CME FedWatch, Kalshi

**Future (AI domain):** arXiv API, Semantic Scholar, GitHub API, Hugging Face Hub API, Epoch AI datasets, Stanford AI Index, Metaculus, Polymarket

## Database Schema (draft)

- `sources`: data provider metadata
- `series`: each tracked time series (id, source, units, frequency)
- `observations`: series_id, observed_at, as_of, value (as_of preserves revisions)
- `questions`: target, horizon, resolution date, resolution rule
- `model_versions`: model type, params, training window, code hash
- `feature_snapshots`: exact feature values used for a forecast
- `forecasts`: question_id, model_version_id, feature_snapshot_id, prediction, interval/probabilities, rationale, created_at, previous_forecast_id (append-only)
- `resolutions`: question_id, actual value, resolved_at, score (error or Brier)
- `postmortems`: forecast_id, miss category, notes

## Modeling Approach

1. Baselines: last value (random walk), historical mean, market/expert benchmark
2. Statistical: ARIMA, exponential smoothing
3. ML: LightGBM on engineered features
4. Calibration: isotonic or Platt scaling for probabilities, conformal prediction for intervals
5. Explanations: SHAP values stored per forecast

Validation is walk-forward only. No random splits.

## Post-mortem Categories

1. **Bad data:** input was revised or wrong
2. **Bad model:** signal was present, model misread it
3. **Regime change:** the world shifted in a way history couldn't show
4. **Variance:** expected miss given the stated probability

## Build Phases

### Phase 1: Data pipeline
- [ ] Postgres in Docker, migration setup
- [ ] Core schema (sources, series, observations)
- [ ] FRED/ALFRED ingestion with vintage storage
- [ ] Backfill history to ~1990

### Phase 2: Backtest harness
- [ ] Point-in-time data loader (as_of queries)
- [ ] Walk-forward evaluation loop
- [ ] Baseline models
- [ ] Scoring: RMSE, Brier, interval coverage, calibration curves

### Phase 3: Models
- [ ] Statistical models
- [ ] LightGBM with feature engineering
- [ ] Calibration layer
- [ ] SHAP snapshots
- [ ] Compare all models against baselines

### Phase 4: Live loop
- [ ] Scheduled forecasting on data releases
- [ ] Resolution job
- [ ] Post-mortem workflow
- [ ] Next.js dashboard: open forecasts, probability history, track record

## Open Questions

- Scheduler choice
- How FOMC odds are sourced reliably over time (API access, historical data)
- Whether the dashboard reads Postgres directly or goes through an API layer
- When to add the AI domain and the agent layer

## Implementation Notes (for task generation)

- Python project managed with uv (pyproject.toml, src/ layout, pytest, ruff).
- Postgres runs via docker-compose; Docker is not yet installed on the dev machine, so the first task should include installing Docker/OrbStack.
- Next.js frontend is deferred to Phase 4; only a placeholder `web/` directory until then.
