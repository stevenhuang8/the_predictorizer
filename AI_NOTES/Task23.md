# AI Task 23: `--domain ai` in the CLI

Done on 2026-10-16. No migration. Every command here reads; none writes or calls the API.

## Quick reference

```sh
uv run eco-forecast forecast --domain ai --target arxiv --horizon 3       # every AI model, with LightGBM's drivers
uv run eco-forecast forecast --domain ai --target binary --question-id 2585
uv run eco-forecast inspect --domain ai [--kind numeric|threshold|curated] [--open] [--limit 50]
uv run eco-forecast inspect --question-id 2585                           # any domain
uv run eco-forecast scores --domain ai [--include-backtest]
uv run eco-forecast backtest --domain ai                                  # the LLM comparison report (Task 20)
uv run eco-forecast ai-forecast | ai-resolve | review-postmortems         # the jobs (Tasks 21-22)
uv run pytest tests/test_cli_ai.py
```

**AI targets:**
- `arxiv`, `hf`, `github`: monthly counts.
- `gpqa`, `swebench`, `arc2`: benchmark records, in %.
- `flops`: the largest training run, in log10 FLOP.
- `binary`: one yes/no question, with `--question-id`.

---

## Decisions (my defaults, not asked)

### `--domain economy` is the default
- **Economy commands keep their output,** with one deliberate exception below. The economy CLI tests (part of the safety net) pass unchanged.
- **One of them caught a change I'd introduced.** I'd dropped `choices` from `--target` so AI targets could pass, which moved the rejection of `--target gdp` from the parser to `main`.
  - Restored: the parser accepts any target of either domain, and `main` checks it matches `--domain`.
  - `--target arxiv` without `--domain ai` gets an error that says to add it.

### The one change: plain `inspect` lists only economy questions
- **Before:** it listed every question in the database. That was fine until the AI job's first run, after which it printed all 2,525 AI questions too.
- **Now:** it shows the economy's 7, which is what it showed before the AI domain existed.

### AI commands
- **The code is in `ai/cli_ai.py`;** `cli.py` only routes. Economy commands don't import the AI stack.
- **`forecast --domain ai`** works like the economy version: it refits every model on the database's data and prints, never stores.
  - **Numeric targets** get each AI job model, with LightGBM's top 5 SHAP drivers. A model that can't fit says why, for example ARIMA's 24-month minimum.
  - **`--target binary --question-id N`** runs that question's binary models (market price, base rate, and the threshold classifier for threshold questions) as if asked today. `--date` changes "today".
  - **A numeric question id** gets an error showing the numeric command to use instead.
- **`inspect --domain ai`** lists AI questions, soonest first.
  - There are 2,525, so `--limit` defaults to 50, with a footer of counts by kind.
  - `--kind` and `--open` narrow it.
  - `inspect --question-id` now also shows an AI question's key and text (economy questions have neither, so their output is unchanged).
- **`scores --domain ai`** is Task 21's AI scorecard.
- **`backtest --domain ai`** runs Task 20's LLM comparison report.
  - `backtest` passes its options through, so `--domain` is taken out of them first; `--domain=ai` works too.
  - **Note:** the task asked for "AI backtest report". The statistical models' long AI backtests have no report of their own; the LLM report covers them on the post-cutoff window.
- **No LLM from the CLI.** `forecast --domain ai` never calls Claude; the LLM runs only in the jobs, opt-in and capped.

---

## What I did
- **`src/eco_prediction/ai/cli_ai.py` (new):** `forecast_ai_numeric`, `forecast_ai_binary`, `list_ai_questions` and `ai_scores`.
- **`src/eco_prediction/cli.py`:**
  - `--domain` on `forecast`, `inspect` and `scores`, and stripped from `backtest`'s pass-through options.
  - `forecast` gets `--question-id`; `inspect` gets `--kind`, `--open` and `--limit`.
  - Help text covers the AI domain.
  - `list_questions(conn, domain="economy")`.
  - `inspect_question` shows the key and text.
- **`tests/test_cli_ai.py`: 15 tests, not tagged `economy`.** They run on a migrated database with arXiv history, the job's AI questions, a market and an economy question.
  - **Help:** it lists the AI commands, and `forecast --help` shows the domain and targets.
  - **The target lists in `cli.py` and `cli_ai.py` agree.**
  - **`backtest` routes by domain** (both spellings), and rejects an unknown domain.
  - **Numeric forecasts:**
    - One model, and all five with drivers.
    - A series with no data explains each model.
  - **Binary forecasts:** a curated question with the market's price; a threshold question with one chosen model.
  - **Error cases:**
    - `binary` without an id.
    - An economy target under `--domain ai`.
    - An unknown model.
    - A numeric question id.
  - **`inspect`:** the AI listing with its limit footer and `--kind`; `--open` drops a market once it resolves; one AI question shows its key and text.
  - **`scores`:** none at first, then the numeric scorecard once a value resolves.
  - **The economy without `--domain`:**
    - `inspect` excludes AI questions.
    - `scores` is the economy's.
    - An AI target is refused with a hint.
    - Defaults are unchanged, and the AI flags don't change the economy listing.

## Checks
- `uv run pytest`: **687 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **Run by hand on the live database** (the task's manual checks):
  - `inspect --domain ai --limit 5` gives a footer of "5 of 2525 shown (curated 2441, numeric 21, threshold 63)", and `--kind curated --open` gives 165.
  - **`forecast --domain ai --target arxiv --horizon 3`** gives January 2027, with data through 2026-10-15 and the latest value 11,245 (September):

    | Model | Forecast |
    |---|---|
    | random walk | 11,245 |
    | linear trend | 11,931 |
    | ARIMA | 10,925 |
    | ETS | 9,422 |
    | LightGBM | 11,489 |

    LightGBM's top driver is `month_10`, the month of the year. It ran in 8 s.
  - **`forecast --domain ai --target binary --question-id 2585`** ("OpenAI announces it has achieved AGI before 2027?") gives the market at 0.125 and the base rate at 0.399.
  - **A threshold question (arXiv ≥ 12,400 for 2026-11)** gives the base rate at 0.463 and the classifier at 0.217, in 16 s.
  - `scores --domain ai` reports nothing scored yet, which is correct.
  - **Plain `inspect`** shows the economy's 7 questions again.

## Where I changed the task's plan
- **argparse, not click.** The CLI already uses argparse; the task's example was click.
- **`backtest --domain ai` is the LLM comparison report.**
- **`inspect --domain ai` adds `--kind`, `--open` and `--limit`,** because of the volume.
- **Plain `inspect` now filters to the economy** (above).

## Tradeoffs
- **`forecast --domain ai` refits every model on each call:** about 8 s for a series, and 15 s for a threshold question, most of it the classifier's fit. That's fine by hand; a dashboard should read stored forecasts instead (Task 24).
- **The AI target list exists twice,** in `cli.py` for parsing without importing the AI stack, and in `cli_ai.py`. A test keeps them in step.
