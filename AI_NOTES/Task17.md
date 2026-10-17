# AI Task 17: the retrieval corpus from the AI sources

Done on 2026-10-16. **Migration 015 is not applied to the live database, and the backfill hasn't been run there.** Both are yours to do (see Quick reference). The live corpus still holds only the 2,538 Polymarket markets indexed in Task 16.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate                     # applies 015 (rebuilds the search column)
uv run python -m eco_prediction.ai.llm.build_corpus            # first run: the backfill, ~1.5 h (mostly Hugging Face)
uv run python -m eco_prediction.ai.llm.build_corpus            # monthly after that: minutes
uv run python -m eco_prediction.ai.llm.build_corpus --sources arxiv github   # some sources only
uv run python -m eco_prediction.ai.llm.build_corpus --start 2024-06-01       # re-read from a date
uv run pytest tests/test_retrieval_corpus.py
```

- **Sources:** `arxiv`, `huggingface`, `github`, `polymarket`.
- **Commits:** after each source, and for Hugging Face after each lab, so an interrupted backfill resumes where it stopped.
- **Tokens:** `GITHUB_TOKEN` and `HF_TOKEN` are used when set. Neither is required.

---

## Decisions (my defaults, not asked)

### arXiv: topic searches, not every AI paper
- **Scope:** `TOPIC_QUERIES` in cs.AI/LG/CL covers SWE-bench, GPQA, ARC-AGI, FrontierMath, Humanity's Last Exam, `ti:"technical report"`, `ti:"system card"`, scaling laws, frontier models and training compute.
  - Each topic is one paged search (500 a page, oldest first). A paper found by several topics is stored once, with all of them in `metadata.topics`.
  - The corpus is about 3,200 papers since 2025-07, likely about 8,000 since 2023. The categories as a whole get 4,000–11,000 papers a month (2024–2026, from the arXiv backfill), mostly irrelevant to our questions. An earlier draft of these notes said 30,000, which was wrong. Lexical retrieval gets worse as noise grows.
- **Dating:** a paper is dated 3 days after its first submission (`ARXIV_ANNOUNCE_LAG`).
  - arXiv announces papers the evening after submission, or after a weekend. So the submission time is early by up to a few days, and the 3 days make it late instead.
- **The abstract is the latest version's,** because the API serves no other. A revised paper is flagged `metadata.revised` (that's hindsight, as with reconstructed observations).
- **arXiv's transient empty pages** are retried twice, then raise.

### Hugging Face: lab cards, read and dated 30 days after creation
- **Which models:** every model of `HF_AUTHORS`, about 20 frontier and open-weight labs.
  - Chosen up front. Picking by likes or downloads would select with hindsight.
  - allenai and google-deepmind are left out: about 800 research checkpoints, and no models.
  - Quantized and converted copies (`DERIVED`: GGUF, AWQ, FP8, MLX, …) are skipped.
  - That's about 2,800 models since 2023.
- **Dating, which needs your attention:** Hugging Face has no "made public" date, only `createdAt`. Labs often create repos privately before a launch.
  - A card dated at creation would let a backtest "see" a model before its launch. That would leak the answer to Polymarket's release questions.
  - So a card is dated 30 days after creation (`HF_PUBLIC_LAG`), and the README is read at the latest commit on or before that moment. Text and date agree.
  - The cost: a release enters the corpus a month late. Polymarket markets and GitHub SDK notes (below) carry release timing without the leak.
  - A model created privately more than 30 days before launch still leaks. I know of no way to detect that from the API.
- **Cards that can't be read:**
  - A model's first commit almost never has a README, which is why it's read later rather than at the first commit.
  - Gated models (Llama, Gemma, some Cohere and FLUX models) refuse both the commit list and the README without approved access. Their document is just the name, task and creation date.
- **Monthly runs:** the run lists each lab (cheap) and fetches cards only for new models, or models still inside their 30-day window. Settled cards aren't fetched again.

### GitHub: stable releases of 10 repos
- **The repos:**
  - PyTorch, Transformers, vLLM, SGLang, Ollama, JAX, TRL and TensorRT-LLM.
  - The OpenAI and Anthropic Python SDKs. Their notes say when a new model became callable ("add gpt-5"). That's release timing with an exact, honest date.
- **What's stored:**
  - Drafts and prereleases are skipped. Ollama alone had 28 releases in 10 weeks, mostly RCs.
  - Dated `published_at`. Each repo resumes from its newest stored release.
- **This uses the core API, not search:** 5,000 requests an hour with a token, 60 without. A backfill needs about 40.

### Text cleanup
- Cards and release notes lose YAML front matter, HTML tags and images. They're cut at a word boundary to 3,000 and 4,000 characters (`CARD_CHARS`, `RELEASE_CHARS`). vLLM's notes run to 44 KB.
- A cut that lands exactly at a word's end keeps that word. A test caught the first version dropping it.

### Ranking (migration 015, and a change to `retrieval.py`)
- **Titles are weight A and bodies weight B,** so a title match counts 2.5× a body match.
- **The rank is now `ts_rank(search, q, 1)`,** not `ts_rank_cd`. Cover density scores every matching stretch of text, so a long body repeating the query words still won (7.9 vs 1.8) even divided by log length. Frequency ranking with the same normalization puts the title match first (0.15 vs 0.05).
- **On the real markets,** "best AI model end of year" now returns the "best AI model … at the end of 2025?" markets first. In Task 16 it returned a US-ban market.

### Metaculus stays deferred
- The task listed `metaculus_question`. Polymarket's markets (`polymarket_question`, Task 16) stand in for it.

---

## What I did
- **`ai/sources/arxiv_client.py`:**
  - `ArxivPaper`, `TOPIC_QUERIES` and `parse_papers`.
  - `search_page`, `iter_search` and `fetch_abstracts`.
- **`ai/sources/huggingface_client.py`:**
  - `HFModel` and `Card`.
  - `iter_author_models`, `commits`, `readme` and `card_as_of`.
- **`ai/sources/github_client.py`:** `Release` and `iter_releases`.
- **`ai/llm/build_corpus.py` (new):**
  - The document builders, `clean_markdown`, and the per-source indexers.
  - `build_corpus` and the CLI.
- **`ai/llm/retrieval.py`:** the rank function (above).
- **`migrations/015_retrieval_title_weight.sql` (new):** rebuilds `search` with weights. No data changes.
- **`tests/test_retrieval_corpus.py`: 14 tests, not tagged `economy`.**
  - **arXiv:**
    - The feed is parsed (id and version, whitespace, dates, categories, authors).
    - Paging and the empty-page retry. An empty page that persists raises.
    - Topics are deduplicated and merged. The exact query string is checked.
  - **Hugging Face:**
    - Author listing across pages.
    - A card is the README at the latest commit before the date, across commit pages.
    - Gated models, a missing README and no commit by the date all give no card.
  - **GitHub:** drafts are skipped, and the listing stops before `since` mid-page.
  - **Documents:**
    - Markdown cleanup and cutting.
    - Dating: arXiv +3 days, cards +30 days.
    - The header of a gated model's document, and a release with no notes.
  - **Building, on a database with fake clients:**
    - Every source is indexed. Derived, too-old and gated models are handled.
    - A model still in its window is read as of now and hidden from today's retrieval.
    - Rebuilding is incremental: arXiv resumes a week back, settled cards aren't fetched again, and releases resume from the newest. `--start` overrides that.
    - Unknown sources are rejected.
  - **Ranking:** a title match outranks a long body that repeats the words.
  - **The CLI** reports each source.

## Checks
- `uv run pytest`: **609 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **A real sample, in a throwaway database migrated to 015, then dropped:**
  - Sources: the live database's markets; vLLM and openai-python releases since 2025; moonshotai and deepseek-ai cards since 2025; arXiv since 2025-07.
  - Built in 84 s: 3,190 abstracts, 66 cards, 268 releases and 2,538 markets.
  - "SWE-bench" as of 2025-10-01 returns 2025 SWE-bench papers, all dated before the cutoff.
  - Model cards come out clean (e.g. DeepSeek-R1-Zero: the header, then the card's introduction).

## Where I changed the task's plan
- **Method names:** `fetch_abstracts(start_date, end_date)` is as the task said. HF has `iter_author_models` plus `card_as_of`, not a bare `fetch_model_cards()`, because model cards need a model list and a date. GitHub has `iter_releases(repo, since)`.
- **No Metaculus client.**
- **Topic and lab lists** rather than whole categories or popularity rankings (above).
- **Added:** migration 015 and the ranking change, the follow-up flagged in Task 16.

## Tradeoffs
- **Polymarket crowds the results.** Dozens of near-identical markets (one per date or per company) fill a top 5. On the sample, "DeepSeek release" and "gpt-5 model" returned only markets. Task 18's context builder should take a few documents per type (`query_type`) rather than one global top 10.
- **Release news reaches the corpus late:** a month late from model cards, and a day or so from SDK notes. Polymarket markets are on time, but they're questions, not facts.
- **The topic list decides what the LLM can read.** A new benchmark (a successor to SWE-bench) needs adding to `TOPIC_QUERIES`. Papers already indexed are kept; a `--start` re-read picks up the new topic's history.
- **Lab and repo lists need upkeep** as labs appear or rename. zai-org was THUDM, and CohereLabs was CohereForAI.
- **The backfill is about 1.5 hours,** almost all of it Hugging Face's 450 requests per 5 minutes. With `HF_TOKEN` the limit is higher.
