# AI Task 22: post-mortems for AI forecasts, with LLM drafts

Done on 2026-10-16. **Migration 016 is not applied to the live database; that's yours to run.** No LLM draft has been made (budget deferred). Everything was tested with a fake client, and nothing was spent.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate                                   # applies 016 (adds draft columns)
uv run python -m eco_prediction.ai.postmortems                               # classify AI misses, free
uv run python -m eco_prediction.ai.postmortems --draft --budget 1            # + Haiku drafts, capped at $1
uv run eco-forecast review-postmortems --domain ai                           # the drafts to review
uv run eco-forecast review-postmortems --domain ai --confirm 812 --note "..."   # confirm (optionally edited)
uv run eco-forecast review-postmortems --domain ai --reject 812              # discard; the category stays
uv run pytest tests/test_llm_postmortem.py
```

- **The daily AI forecast job now classifies post-mortems** right after resolving: rule-based, no API calls. It works before migration 016 too, because only drafts need the new columns.

---

## Decisions (my defaults, not asked)

### AI forecasts needed categories first
- **The economy classifier deliberately skips AI forecasts.** Its query is economy-only, with a comment pointing at this task.
- **The task says a draft supplements the category,** so AI forecasts needed a category first. `ai/postmortems.py` gives them one, with the same four categories and the same rules:
  - **Numeric:** the economy's `classify_numeric`, unchanged.
    - Within 1.5× the interval's half-width: expected variance.
    - A first live value later revised toward the forecast: bad data.
    - A move beyond the 99th percentile of past same-length moves: regime change.
    - Anything else: bad model.
    - **Count series (arXiv, HF, GitHub) are judged on the log scale,** where their growth is steady. In levels, every recent move dwarfs the early history, so almost everything would look like a regime change. Their notes start "On the log scale:".
  - **Binary:** the FOMC rule, with a reference forecast in place of the bill spread.
    - The outcome got at least 20%: expected variance.
    - Otherwise, the reference had favoured it (at least 50%): bad model, since the model missed an available signal.
    - The reference missed it too: regime change, a surprise.
    - The reference is the Polymarket price for curated questions, and the historical base rate for threshold questions.
    - Binary outcomes aren't revised, so bad data never applies.

### A floating-point edge in the 20% rule
- **For a forecast of 80% yes when the outcome was no,** the outcome's probability is `1 − 0.8 = 0.19999999999999996`. That's just under 20%, so it was categorised as bad model instead of expected variance. A test caught it.
- **The FOMC rule never hit this,** because there the probability is read directly rather than computed as 1 − p.
- **Fix:** comparisons allow a 1e-9 tolerance (`TOLERANCE`).

### Drafts: opt-in, Haiku, misses only, live only by default
- **What a draft is:** a 2–3 sentence explanation of why a forecast missed (`draft_llm_postmortem`). The prompt includes:
  - The question and its resolution rule.
  - The forecast: point and interval, or the probability.
  - The model's own rationale (LLM forecasts) or its top 5 SHAP drivers (LightGBM).
  - What happened.
  - The classifier's category and reason.
  - The series' values from 3 months before the forecast until it resolved.
  - **Documents published between the forecast and the resolution.** Hindsight is the point of a post-mortem. A test checks that a document from before the forecast and one from after the resolution are both left out.
- **Model: Claude Haiku 5.5,** as the task asked, at low effort.
  - Haiku 5.5 has no server-side refusal fallback, so drafts don't send one. Drafting with Opus or Sonnet would (`make_drafter`).
  - The system prompt asks it to weigh data issues, model limitations, regime changes and variance in that order, name the specific event, and say in one clause if the evidence points to a different category.
- **Which forecasts:** only misses (not expected variance), and only live forecasts unless `--include-backtest`. The first real run made about 500 catch-up backtests, which would multiply the cost.
- **`--draft` needs `--budget`.** The run stops at the budget, and each draft commits as it's made.
- **Calls go through the forecaster's machinery:** logged to `llm_calls`, reused by context hash, failures logged and skipped. An empty or overlong (over 1,500 characters) explanation counts as a malformed reply.

### Reusing the forecaster
- **`ClaudeForecaster._forecast`'s call-log-validate body is now a public `complete(prompt, schema, validate, system=..., prompt_version=...)`,** and the drafts use it with their own prompt.
- **The forecast request and its context hash are unchanged.** Earlier logged calls stay reusable, and Task 18's tests, which pin the exact request body and the wire request through the real SDK, still pass.

### Storage: migration 016, not the task's 011
- **011–015 were taken.** 016 adds to `postmortems`:
  - `llm_drafted` and `human_confirmed` (both `NOT NULL DEFAULT FALSE`).
  - `draft_note`.
  - `draft_call_id`, a foreign key to `llm_calls`, set to null if the call is deleted.
  - `confirmed_at`.
- **Constraints:** a draft must have a note, and a confirmed draft must have been drafted and dated. An index covers the review queue.
- **The draft stays separate from `notes`,** which remains the classifier's reason. Economy post-mortems get the defaults, and a test shows a manual economy post-mortem is untouched by everything here.
- **An automatic reclassification keeps an existing draft.**

### Review workflow
- **`eco-forecast review-postmortems --domain ai`** lists unconfirmed drafts, each with its forecast, category, the classifier's reason and the draft.
- **`--confirm ID`** marks a draft confirmed and dated. `--note TEXT` replaces its text with your edit.
- **`--reject ID`** discards the draft and keeps the category.
- **`--all`** lists confirmed drafts too.
- **It's non-interactive** (flags, not prompts), so it scripts and tests easily.
- **`--domain` only accepts `ai`:** only AI forecasts have drafts.

---

## What I did
- **`migrations/016_llm_postmortems.sql` (new).**
- **`src/eco_prediction/ai/postmortems.py` (new):**
  - Classification: `ai_forecasts_to_review`, `classify_binary`, `classify_ai` and `run_ai_postmortems`.
  - Drafting: `make_drafter`, `draft_prompt`, `draft_llm_postmortem`, `undrafted_misses` and `draft_postmortems`.
  - Review: `list_drafts`, `confirm_draft` and `reject_draft`.
  - Two CLIs: classify/draft, and review.
- **`src/eco_prediction/ai/llm/claude_forecaster.py`:** `complete()` extracted from `_forecast`; `request()` takes a system prompt.
- **`src/eco_prediction/ai/jobs/ai_forecast_job.py`:** classifies post-mortems after resolving.
- **`src/eco_prediction/cli.py`:** `review-postmortems`.
- **`tests/test_llm_postmortem.py`: 11 tests, not tagged `economy`.**
  - **Seed:** a migrated database where the job forecasts arXiv and two markets, November arrives far above trend, and both markets resolve yes.
  - **Binary rules:** including the 1 − 0.8 boundary.
  - **Categories from the database:**
    - The market's own 10% miss is a surprise.
    - Its 60% forecast is expected variance.
    - The arXiv jump is a regime change, judged on logs.
  - **The log scale for counts.**
  - **Drafts:**
    - Every miss and nothing else is drafted.
    - The request uses Haiku at low effort, with no fallback and the draft system prompt.
    - The prompt includes the category, the values since the forecast, and only documents from the right window.
    - Drafts are stored unconfirmed with their call, nothing is redrafted, and a reclassification keeps them.
  - **Live only by default, and the budget stops after one draft.**
  - **A malformed draft** is logged and skipped.
  - **A draft needs a category first.**
  - **Review:**
    - The listing.
    - Confirm with an edited note (it leaves the queue).
    - Reject (the category stays).
    - Confirming with no draft fails.
    - `--note` without `--confirm` is refused.
  - **Schema constraints:** a draft without a note, and a confirmation without a draft, are both refused.
  - **Economy post-mortems untouched, and the CLI passthrough.**

## Checks
- `uv run pytest`: **672 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **A dry run of the AI job on the live database,** with the new step: it ran cleanly, and the counts were unchanged.
  - "Post-mortems: 0 classified" is correct. Every live AI forecast so far is for a question that hasn't resolved yet.
  - The 541 forecasts from the first run were all recognised as stored.

## Not done
- **Migration 016 on the live database.** It's needed for drafts, not for the daily job's classification.
- **Real drafts, and the task's "manually review sample LLM drafts".** They wait for the budget decision, and for live AI forecasts to resolve: the first 1-month numeric questions resolve in early December.

## Where I changed the task's plan
- **A new module, `ai/postmortems.py`,** not an extension of `postmortem/classifier.py`, which stays economy-only. It reuses that file's `classify_numeric` and `Classification`.
- **`draft_llm_postmortem(conn, forecast_id, drafter)`** loads the question, prediction and outcome itself, rather than taking them as arguments.
- **Migration 016,** with `draft_call_id` and `confirmed_at` added.
- **AI forecasts get rule-based categories,** which they didn't have. That's needed before a draft can supplement one.

## Tradeoffs
- **The binary rule's reference is the market or the base rate, not an independent signal.** When the market price model misses, its reference is itself, so its misses are always a regime change. That's consistent ("nobody saw it coming"), but the market can't be "bad model" by construction.
- **Drafts see hindsight,** which is the point, but they can also rationalise. A plausible story isn't evidence. That's why drafts stay unconfirmed until a person reads them.
- **The draft doesn't set the category.** If the LLM disagrees, it says so in the text, and a person can change the category with the economy's `postmortem.classifier --set ID CATEGORY` (manual wins over automatic).
