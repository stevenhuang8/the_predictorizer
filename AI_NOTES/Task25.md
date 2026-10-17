# AI Task 25: the end-to-end test and the documentation

Done on 2026-10-16. That completes the AI domain's 25 tasks. Nothing is written to the live database, and no API is called.

## Quick reference

```sh
uv run pytest tests/test_ai_integration.py -v                     # the whole AI lifecycle, ~30 s
uv run pytest --cov=eco_prediction.ai --cov-report=term           # AI coverage (95%)
```

- **Docs:** `README.md` (the "AI domain forecasting" section, getting started, cost management) and `NOTES/AI_DOMAIN.md` (design, prompts, results, costs, what's next).

---

## Decisions (my defaults, not asked)

### One test through the whole lifecycle, on the real code paths
- **`tests/test_ai_integration.py` runs the pipeline once, in order,** on a throwaway database:
  1. **Ingest** through the real `monthly_ingest.ingest_backfill`, from fake counters for arXiv, HF, GitHub, GPQA and compute. **Polymarket** goes through the real `ingest_polymarket.ingest`, from a fake client returning two markets with daily prices.
  2. **Questions and forecasts** come from the AI job (`run_ai_forecast_job`), with every model: random walk, trend, LightGBM with SHAP, base rates, the threshold classifier and the market price. **The LLM is mocked** (a fake Anthropic client) under a budget, and its calls are logged and linked to their forecasts.
  3. **Two months later:** `ingest_live` brings November's values (arXiv jumps to about 2.6× trend), and a market closes "yes".
  4. **Resolve and score:** the arXiv November question and the market resolve, and their forecasts are scored.
  5. **Post-mortems:** every scored forecast is classified, and the arXiv jump is a regime change for every model. LLM drafts follow, and one is confirmed through the review CLI.
  6. **The track record:**
     - The scorecard has numeric rows for the main models, and binary rows for thresholds and Polymarket questions, Brier in [0, 1].
     - `eco-forecast scores --domain ai` prints it.
     - A later run adds the new month, duplicates nothing, and makes no LLM call unless asked.
- **The economy runs alongside on the same database,** using the economy job test's own synthetic store:
  - Before and after the AI pipeline, its questions, forecasts and scores are identical.
  - A re-run stores nothing new.
  - Its scored forecasts and question listing contain no AI data.
- **No external API is called:** arXiv, HF, GitHub, Epoch, Polymarket and Anthropic are all faked.
- **The test found nothing wrong in the code.** It took three fixes to the test itself:
  - The economy store has no CPI.
  - The fake market lacked the `closed_at` metadata key the real client always sets.
  - The "surprise" November I first chose was below the 2%/month trend.

### Coverage
- **Added `pytest-cov` as a dev dependency.**
- **The AI modules are at 95%** (4,334 statements, 198 missed). Every module is above the task's 80%; the lowest are `ingest_github` (84%) and `llm/backtest` (85%).

### README
- **The "AI domain (in progress)" roadmap is now "AI domain forecasting".** It covers:
  - What's forecast.
  - The models and the LLM approach.
  - Scoring.
  - Success criteria and current performance: no live track record yet, the synthetic results, and the LLM not yet run, with its estimated cost.
  - Commands, the dashboard section, and **LLM cost management** (opt-in budgets, the dry-run estimate, call reuse, the call log, Haiku).
- **The architecture diagram** now shows both domains' sources, the text corpus and retrieval, the opt-in Claude step, and LLM-drafted post-mortems.
- **Getting started** adds:
  - Which keys are optional and what each does: `GITHUB_TOKEN` and `HF_TOKEN` in `.env`, and `ANTHROPIC_API_KEY` in `.env.local`, used only by budgeted LLM runs.
  - The corpus build, the AI job (dry run first), both launchd agents, and the dashboard's `/ai`.
  - An optional LLM step: estimate, then a budgeted run, then the report.
- **Tech stack, layout, key tables, design decisions and known limitations** are updated:
  - Migrations run 001–016.
  - New sections cover `ai/llm/`, `ai/jobs/` and `deploy/launchd/`.
  - New tables: the Polymarket tables, `retrieval_documents` and `llm_calls`.
  - Two new design decisions: the opt-in, budgeted LLM, and resolution to the first live value.
  - New limitations: the LLM not yet run, and reconstructed AI history.

### `NOTES/AI_DOMAIN.md`
- **One document over the 25 task notes:**
  - What it does, and the decisions by area: data, models, the LLM, prompts, operations.
  - Results so far (live and synthetic).
  - A cost analysis, from the dry-run estimate and a live-use projection.
  - Seven future improvements, starting with running the LLM backtest on Haiku.

## Checks
- `uv run pytest`: **689 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **Every command the README names** starts and accepts the flags shown (`--help` on each script and `eco-forecast` subcommand), and both launchd plists exist.

## Not done
- **A from-scratch walk through the README on a fresh machine** (Docker, migrations, backfills). It was verified piece by piece on this machine over the tasks, not end to end on a new one.
- **Any real LLM numbers:** the budget decision comes next.
