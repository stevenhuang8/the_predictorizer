# Eco Prediction

A forecasting system that makes **specific, scoreable predictions**, keeps an honest record of how they turned out, and explains why it was wrong when it misses. It covers two domains:

- **US economy** (live): CPI inflation, unemployment and Fed rate decisions.
- **Direction of AI** (live since 2026-10-16): research output, open-source activity, benchmark records, training compute and AI questions from Polymarket, with an **LLM forecaster (Claude)** competing against the statistical and ML models under the same rules.

### Ground rules (both domains)

1. **Every forecast can be scored**: a clear question, a resolution date and a resolution rule.
2. **No hindsight**: models only see data as it was published on the forecast date. Every published version of a value is stored with its `as_of` date.
3. **Forecasts are never edited**: they're saved before the outcome is known, with the model version and exact inputs.
4. **Models must beat simple baselines** like "next month will look like this month".
5. **Judge calibration, not single misses**: a "70%" call should come true about 70% of the time.

---

## How it works

```
 economy: FRED / ALFRED ─┐                                      ┌─ baselines, ARIMA/ETS, LightGBM
 AI: arXiv, HF, GitHub,  ├▶ PostgreSQL ──▶ point-in-time ──▶ features ─┤  (AI: + market price, base rates,
     Epoch, Polymarket ──┘  (every vintage,   view of data             │   threshold classifier)
 AI text corpus ────────────▶ retrieval ──▶ context bundle ──────────▶ └─ Claude (opt-in, budgeted)
 (papers, model cards,         (published                                        │
  releases, markets)            before the forecast)                              ▼
                                                                         stored forecasts
        dashboard + CLI ◀── post-mortem: why did it miss? ◀── resolve + score     (inputs, SHAP,
                            (+ LLM-drafted note to confirm)                        rationales)
```

Two daily jobs, one per domain (`scheduler/forecast_job.py`, `ai/jobs/ai_forecast_job.py`), each under its own lock: refresh data, resolve and score due questions, classify misses (bad data / bad model / regime change / bad luck), and make the new forecasts.

---

## Economy domain

| Target | Prediction | Horizons |
|---|---|---|
| **CPI inflation** (YoY %) | Value + 80% range | 1, 3, 6 months |
| **Unemployment rate** | Value + 80% range | 1, 3, 6 months |
| **Fed decisions (FOMC)** | P(cut / hold / hike) per meeting | 7 days before, and 1, 3, 6 months |

**Data:** 16 series from FRED and its vintage archive ALFRED: the targets, yield curve, oil and gas prices, producer prices, wages, jobless claims, inflation expectations, rent, the Fed funds target and short-term Treasury yields. These become 41 features per date (`features/engineering.py`).

**Models:** random walk and historical mean (baselines), ARIMA and ETS (statistical), and LightGBM predicting the *change* from today. For the Fed: always-hold, climatology, persistence (Markov chain) and a 3-class LightGBM classifier. Live: random walk, ARIMA, LightGBM, persistence and the FOMC classifier.

**Testing:** walk-forward backtests (`backtest/harness.py`) train only on data published by each date. LightGBM is tuned on time-ordered folds with a gap. Ranges come from each model's past errors (conformal calibration in `models/calibration.py`), and every LightGBM forecast stores SHAP values. Numeric forecasts are scored with RMSE, MAE, bias and interval coverage; Fed forecasts with the 3-class Brier score (0 perfect, 2 confidently wrong).

### Results

Full report: `results/model_comparison.html` (regenerate with `uv run eco-forecast backtest`).

**Inflation and unemployment** (2020–2023, 48 monthly forecasts, RMSE in pp, lower is better):

| Model | CPI 1m | CPI 3m | CPI 6m | Unemp. 1m | Unemp. 3m | Unemp. 6m |
|---|---|---|---|---|---|---|
| **ARIMA** | **0.83** | **1.24** | **1.87** | 3.35 | 4.40 | 4.74 |
| ETS | 1.16 | 1.68 | 2.37 | 3.05 | 3.86 | 3.78 |
| Random walk | 1.27 | 1.84 | 2.60 | 2.67 | 3.17 | 2.52 |
| LightGBM | 1.04 | 1.98 | 2.82 | 2.84 | 3.28 | 4.18 |
| Historical mean | 2.86 | 2.85 | 2.74 | **2.62** | **2.62** | **1.90** |

- ✅ ARIMA is the best inflation model at every horizon, 28–35% better than the random walk.
- ❌ LightGBM missed the main goal of beating the random walk on 3-month inflation; its top feature was a time trend.
- ⚠️ The COVID spike dominates unemployment; the flat mean "wins" only because it never moves (by MAE the random walk is best).
- ⚠️ 80% ranges were too narrow in this turbulent period (LightGBM's covered only 35–42%).

**Fed decisions** (2005–2026, 173 meetings, 7 days before each, Brier score):

| Model | All meetings | 50 meetings where rates changed |
|---|---|---|
| Always hold | 0.578 | 2.000 |
| Climatology | 0.466 | 1.131 |
| Persistence | 0.331 | 0.743 |
| **LightGBM** | **0.239** | **0.508** |

✅ ML clearly pays off here: short-term yields minus the Fed rate reveal what markets expect.

---

## AI domain forecasting

The second domain asks how fast AI is moving, under the same rules as the economy, and adds an **LLM forecaster (Claude)** that competes with the statistical and ML models. It is purely additive: the economy's tables, jobs, CLI and dashboard behave as before, and a regression safety net (`tests/test_regression_safety.py`) fails if any economy test goes missing. Spec: `.taskmaster/docs/prd_ai.md`; design and decisions: `NOTES/AI_DOMAIN.md`; per-task notes: `AI_NOTES/`.

**What is forecast**

| Questions | Source | Asked |
|---|---|---|
| **Monthly counts**: arXiv AI papers (cs.AI/LG/CL), new Hugging Face models, new GitHub AI repos | arXiv, Hugging Face Hub and GitHub search APIs | value + 80% range, 1/3/6 months ahead |
| **Records**: best GPQA Diamond, SWE-bench Verified and ARC-AGI-2 scores; largest training run (log10 FLOP) | Epoch AI | value + 80% range, 1/3/6 months ahead |
| **Thresholds**: "Will arXiv AI papers be ≥ 12,400 in Nov 2026?" (+10/25/50%, or 25/50/75% of the gap to 100%, or 2/3/10x compute) | the same series | P(yes), 1/3/6 months ahead |
| **Polymarket questions**: AI releases, benchmark records and model rankings (≥ $10k traded) | Polymarket | P(yes), 90, 30 and 7 days before the market closes |

**Models:** random walk, historical mean and linear trend (baselines); ARIMA and ETS on a log or "headroom to 100%" scale; LightGBM on 109 AI features (growth rates, records, cross-series signals), pooled across benchmarks; for yes/no questions, base rates, the market's own price and a LightGBM threshold classifier; and the **Claude forecaster**.

**The LLM forecaster** (`ai/llm/`): for each question it builds a point-in-time context bundle (the series' last 12 months, the baselines' forecasts, the market price, and documents retrieved from a dated corpus of arXiv abstracts, Hugging Face model cards, GitHub release notes and Polymarket questions, all published before the forecast) and asks `claude-opus-5-5` for structured JSON with a rationale first. Every call is logged to `llm_calls` with its prompt, response, tokens and cost; identical calls are reused, never paid for twice. Its backtest only uses forecasts made after the model's training data cutoff (June 2026), so it can't recall outcomes. An ensemble blends the LLM with the other models, with weights learned walk-forward.

**Scoring:** numeric forecasts by error and relative RMSE; yes/no by the Brier score (0 perfect, 1 confidently wrong), calibration, and comparison with the Polymarket price on the same day. Misses get the economy's four post-mortem categories, and optionally an LLM-drafted explanation for a person to confirm. Metaculus is on hold: its API terms forbid evaluating AI models without written permission.

**Success criteria and current performance:** the PRD's bar is that the LLM beats the base rate after its training cutoff (and the statistical models beat the random walk). Live AI forecasts began on 2026-10-16; the first numeric ones resolve in early December 2026, so there's no live track record yet. On synthetic data the models behave as designed (LightGBM 1,181 vs ARIMA 1,812 vs random walk 15,087 RMSE on a series with a cross-series lead; the threshold classifier's Brier 0.054 vs 0.254 for the base rate). The LLM hasn't been run: the post-cutoff backtest (588 forecasts) is built and estimated at about $44 on Opus 5.5 or $1.10 on Haiku 5.5, waiting on a budget decision.

### Running it

```sh
uv run eco-forecast forecast --domain ai --target arxiv --horizon 3          # every model, now (not stored)
uv run eco-forecast forecast --domain ai --target binary --question-id 2585  # one yes/no question
uv run eco-forecast inspect --domain ai --kind curated --open                 # AI questions
uv run eco-forecast scores --domain ai                                        # AI track record
uv run eco-forecast ai-forecast --dry-run                                     # what the daily job would store
uv run eco-forecast review-postmortems --domain ai                            # LLM-drafted notes to confirm
```

The dashboard's **AI progress** section (`/ai`) shows each series with every model's forecast, the open questions, each forecast with its rationale, and the track record against the market.

### LLM cost management

- **The LLM never runs unless asked:** the daily job forecasts with it only with `--llm MODEL --llm-budget USD`; post-mortem drafts only with `--draft --budget USD`; the backtest refuses to run without `--budget`. Each stops before the next call once new spending reaches its budget.
- **Estimate before spending:** `uv run python -m eco_prediction.ai.llm.backtest --dry-run` counts the forecasts and prices them from the real prompts.
- **Nothing is paid for twice:** a call whose model, prompt and parameters match an earlier successful one reuses it (`llm_calls.context_hash`), so re-runs are free.
- **See what was spent:** `llm_calls` holds every call's tokens and cost (failed calls included); the dashboard and `eco-forecast scores --domain ai` show totals.
- **Cheaper models:** `--model claude-haiku-5-5` costs about 1/40th of Opus 5.5 per forecast, with the same training cutoff.

---

## Tech stack

| Layer | Tools |
|---|---|
| **Data and models** | Python 3.13, pandas, statsforecast (ARIMA/ETS), LightGBM, SHAP, scikit-learn, SciPy, matplotlib; Claude (`claude-opus-5-5`) via the `anthropic` SDK |
| **Database** | PostgreSQL 16 in Docker, psycopg2, hand-written SQL migrations; Postgres full-text search for the retrieval corpus |
| **Data sources** | FRED / ALFRED; arXiv, Hugging Face Hub, GitHub, Epoch AI, Polymarket |
| **Frontend** | Next.js 16, TypeScript, Tailwind CSS 4 |
| **Tooling** | uv, pytest (680+ tests; 95% coverage of the AI modules), ruff, mypy; launchd + Postgres advisory locks for the daily jobs |

## Project layout

```
eco_prediction/
├── src/eco_prediction/
│   ├── data/          # FRED client, ingestion, FOMC calendar + decision labels
│   ├── ai/            # AI domain: sources/ (API clients), ingestion, questions, features, models,
│   │   ├── llm/       #   Claude forecaster, retrieval corpus, call log, LLM backtest, ensemble, report
│   │   └── jobs/      #   daily AI forecast job + resolution job
│   ├── db/            # connection pool, migration runner, per-table helpers
│   ├── backtest/      # point-in-time views, walk-forward harness, metrics
│   ├── features/      # economy feature pipeline
│   ├── models/        # baselines, ARIMA/ETS, LightGBM, FOMC model, SHAP, calibration
│   ├── scheduler/     # daily forecast job + resolution/scoring job
│   ├── postmortem/    # classifies each miss
│   └── cli.py         # `eco-forecast` command
├── migrations/        # numbered SQL migrations (001–016)
├── deploy/launchd/    # the two daily jobs' launch agents
├── tests/  results/  web/         # web/app/ai: the dashboard's AI section
├── NOTES/             # economy task notes
└── AI_NOTES/          # AI domain task notes
```

Key tables: `observations` (every vintage: `series, period, as_of, value, reconstructed`), `questions` (with `domain`, binary questions and Polymarket links), `forecasts`, `model_versions`, `feature_snapshots`, `resolutions`, `postmortems` (with LLM drafts); for the AI domain also `platform_questions` / `platform_probabilities` (Polymarket), `retrieval_documents` (the dated corpus) and `llm_calls` (every LLM call, its tokens and cost).

---

## Getting started

**Prerequisites:** Docker (or OrbStack), [uv](https://docs.astral.sh/uv/), Node.js, a free [FRED API key](https://fredaccount.stlouisfed.org/apikeys), and on macOS `brew install libomp`.

**AI keys (all optional):** the AI ingestion runs without keys; `GITHUB_TOKEN` and `HF_TOKEN` in `.env` raise rate limits (a free Hugging Face token doubles them, and makes the first backfill much faster). `ANTHROPIC_API_KEY` in `.env.local` is only needed to run the LLM, which never happens unless you ask for it with a budget (see *LLM cost management*).

```sh
# 1. Configure and start the database
cp .env.example .env.local          # fill in FRED_API_KEY and a Postgres password
set -a; . ./.env.local; set +a
docker compose up -d                # Postgres on port 5433

# 2. Install, migrate, load data
uv sync
uv run python -m eco_prediction.db.migrate
uv run python -m eco_prediction.scripts.backfill_data    # economy history from 1990

# 3. Economy forecasts
uv run eco-forecast forecast --target cpi --horizon 3    # every model, now
uv run eco-forecast forecast --target fomc               # next Fed meeting
uv run eco-forecast scores                               # live track record

# 4. AI data (--backfill for history, --dry-run to print only; no flag = last 3 months)
uv run python -m eco_prediction.ai.ingest_arxiv --backfill
uv run python -m eco_prediction.ai.ingest_huggingface --backfill
uv run python -m eco_prediction.ai.ingest_github --backfill
uv run python -m eco_prediction.ai.ingest_github stars   # monthly stars snapshot
uv run python -m eco_prediction.ai.ingest_epoch --backfill   # benchmark SOTA + compute
uv run python -m eco_prediction.ai.ingest_polymarket         # markets + daily prices (run daily)
uv run python -m eco_prediction.ai.llm.build_corpus          # text for the LLM (~1.5 h the first time)

# 5. Daily pipelines (by hand, or via launchd at 09:00 and 09:30) and dashboard
uv run python -m eco_prediction.scheduler.forecast_job --once-per-month
uv run python -m eco_prediction.ai.jobs.ai_forecast_job --dry-run           # check first: writes nothing
uv run python -m eco_prediction.ai.jobs.ai_forecast_job --once-per-month    # questions + forecasts, no LLM
for job in forecast-job ai-forecast-job; do
  cp deploy/launchd/com.eco-prediction.$job.plist ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.eco-prediction.$job.plist
done
cd web && cp .env.local.example .env.local && npm install && npm run dev   # localhost:3000, /ai

# 6. The LLM (optional, costs money): estimate, then run with a budget
uv run python -m eco_prediction.ai.llm.backtest --dry-run
uv run python -m eco_prediction.ai.llm.backtest --model claude-haiku-5-5 --budget 3
uv run python -m eco_prediction.ai.llm.comparison_report      # results/llm_comparison.html
```

Checks: `uv run pytest`, `uv run ruff check src`, `uv run mypy src`.

---

## Key design decisions

| Decision | Why | Cost |
|---|---|---|
| Store every data vintage | Backtests can't secretly use revised data | More data; every query picks a version |
| Score against the first release | That's what was known at resolution | Revisions can make a forecast look wrong (tagged `bad_data`; revised-value metrics reported as diagnostics) |
| Models build their own point-in-time training rows | Lookahead is impossible; backtest and live share one code path | Slower than one big `fit(X, y)` |
| LightGBM predicts the change, not the level | Trees can't extrapolate past seen levels | Still hasn't beaten the random walk at 3 months |
| Raw SQL migrations on self-hosted Postgres | Full control, learning goal | More code, no hosted backups |
| Only the scheduled job writes forecasts | One source of truth for the track record | Manual CLI forecasts aren't saved |
| AI domain is additive, guarded by a safety net | The live economy pipeline can't regress | Some shared code needs guarded, economy-default branches |
| LLM backtests only after its training cutoff | The model may have memorized earlier outcomes | A smaller LLM evaluation set |
| The LLM is opt-in, capped by a budget, and reuses identical calls | Spending is a decision, never a side effect | Its forecasts wait until someone runs it |
| AI questions resolve to the first *live* value | Backfilled history is reconstructed with hindsight | The first scores arrive a month after live pulls start |

## Known limitations

- LightGBM doesn't yet beat the random walk on 3-month inflation; dropping the time-trend feature looked promising in a quick test.
- Ranges are too narrow in turbulent periods; conformal recalibration isn't in the live job yet.
- ALFRED's archive starts 2009–2011 for several series, so early backtests have fewer features.
- Live economy forecasts began in October 2026 and none have resolved yet. Live AI forecasts began on 2026-10-16; the first resolve in early December 2026.
- The LLM forecaster hasn't been run on real questions: its backtest and live use wait on a budget decision.
- AI history is reconstructed (backfilled with hindsight): GitHub counts repos by their topics today, Hugging Face counts only models not since deleted, and late benchmark evaluations are dated as if known at the time.
