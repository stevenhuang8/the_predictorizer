"""Build the retrieval corpus from the AI sources (AI task 17).

Usage:
    uv run python -m eco_prediction.ai.llm.build_corpus              # all sources, incremental
    uv run python -m eco_prediction.ai.llm.build_corpus --sources arxiv github
    uv run python -m eco_prediction.ai.llm.build_corpus --start 2023-01-01   # re-backfill

    from eco_prediction.ai.llm.build_corpus import build_corpus
    build_corpus(conn, ["arxiv"], now=datetime.now(UTC))   # commits per source

Four sources, each its own `document_type` in `retrieval_documents`:

- **`arxiv_abstract`**: papers in cs.AI/LG/CL on the forecasts' topics
  (`arxiv_client.TOPIC_QUERIES`: the benchmarks, technical reports, scaling,
  frontier models, training compute). Dated `ARXIV_ANNOUNCE_LAG` (3 days)
  after first submission: arXiv announces a paper the evening after it's
  submitted, or after a weekend, so the submission time is a day or so early.
  The abstract is the latest version's (the API serves no other); a revised
  paper is flagged `revised` in metadata.
- **`hf_model_card`**: models from `HF_AUTHORS` (frontier and open-weight
  labs, chosen up front rather than by popularity, which would pick with
  hindsight), minus quantized and converted copies (`DERIVED`). Hugging Face
  gives no date a repo went public, only when it was created, and labs often
  create repos privately before a launch. So a card is dated `HF_PUBLIC_LAG`
  (30 days) after creation, with the README as it stood then, so the text and
  its date agree. A gated model (Llama, Gemma) has no readable card; its
  document is the model's name, task and creation date.
- **`github_release`**: stable releases of `RELEASE_REPOS` (core frameworks,
  inference engines, and the OpenAI and Anthropic SDKs, whose notes announce
  new models), dated when published. Prereleases are skipped.
- **`polymarket_question`**: the markets already mirrored by
  `ingest_polymarket` (`retrieval.index_platform_questions`).
Metaculus stays deferred (AI_NOTES/Task8.md).

Runs are incremental. arXiv resumes a week before its newest document, and
each repo's releases from its newest. Hugging Face lists every lab's models
(a few requests each) and fetches cards only for models not yet indexed, or
still inside their 30-day window. `--start` sets the earliest date to index
(`CORPUS_START`, 2023-01-01, when nothing is indexed yet). Card and release
texts are cleaned of front matter, HTML and images, and cut to
`CARD_CHARS` / `RELEASE_CHARS`. Each source is committed as it finishes, and
Hugging Face after each lab, so an interrupted backfill resumes.

A first backfill takes about 2 minutes for arXiv (5 s a request), about 80
for Hugging Face (450 requests per 5 minutes), and 1 for GitHub.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta

from psycopg2.extensions import connection as Connection

from eco_prediction.ai.llm.retrieval import (
    Document,
    UpsertResult,
    index_platform_questions,
    upsert_documents,
)
from eco_prediction.ai.sources.arxiv_client import ArxivClient, ArxivPaper
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.sources.github_client import GitHubClient, Release
from eco_prediction.ai.sources.huggingface_client import (
    Card,
    HFModel,
    HuggingFaceClient,
)
from eco_prediction.db.connection import ConfigError, connection

ARXIV_ABSTRACT = "arxiv_abstract"
HF_MODEL_CARD = "hf_model_card"
GITHUB_RELEASE = "github_release"
SOURCES = ("arxiv", "huggingface", "github", "polymarket")

CORPUS_START = date(2023, 1, 1)
ARXIV_ANNOUNCE_LAG = timedelta(days=3)
ARXIV_OVERLAP = timedelta(days=7)  # re-read before the newest paper, for late entries
HF_PUBLIC_LAG = timedelta(days=30)
CARD_CHARS = 3000
RELEASE_CHARS = 4000

HF_AUTHORS = (
    "openai", "meta-llama", "google", "mistralai", "Qwen", "deepseek-ai",
    "microsoft", "nvidia", "apple", "ibm-granite", "CohereLabs", "xai-org",
    "moonshotai", "zai-org", "MiniMaxAI", "stabilityai", "black-forest-labs",
    "tencent", "baidu", "ByteDance-Seed",
)  # fmt: skip
RELEASE_REPOS = (
    "pytorch/pytorch", "huggingface/transformers", "vllm-project/vllm",
    "sgl-project/sglang", "ollama/ollama", "jax-ml/jax", "huggingface/trl",
    "NVIDIA/TensorRT-LLM", "openai/openai-python", "anthropics/anthropic-sdk-python",
)  # fmt: skip
DERIVED = re.compile(
    r"gguf|awq|gptq|fp8|int4|int8|mlx|bnb|onnx|quantized|-4bit|-8bit|nvfp4|w4a16|exl2",
    re.IGNORECASE,
)


# --- documents ----------------------------------------------------------------------

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML = re.compile(r"<[^>]+>")
_BLANK = re.compile(r"\n\s*\n\s*\n+")


def clean_markdown(text: str, limit: int) -> str:
    """Markdown without front matter, images or HTML tags, cut to `limit`
    characters at a word boundary."""
    text = _FRONT_MATTER.sub("", text.replace("\r\n", "\n"))
    text = _HTML.sub(" ", _IMAGE.sub("", text))
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANK.sub("\n\n", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if not text[limit].isspace():  # mid-word: drop the partial word
        cut = cut.rsplit(None, 1)[0]
    return cut.rstrip() + " …"


def arxiv_document(paper: ArxivPaper) -> Document:
    return Document(
        ARXIV_ABSTRACT,
        paper.arxiv_id,
        paper.abstract,
        paper.published + ARXIV_ANNOUNCE_LAG,
        title=paper.title,
        metadata={
            "url": f"https://arxiv.org/abs/{paper.arxiv_id}",
            "submitted_at": paper.published.isoformat(),
            "primary_category": paper.primary_category,
            "categories": list(paper.categories),
            "authors": list(paper.authors[:10]),
            "topics": list(paper.topics),
            "revised": paper.version > 1,
        },
    )


def model_document(model: HFModel, card: Card | None) -> Document:
    author = model.id.split("/", 1)[0]
    task = f" ({model.pipeline_tag})" if model.pipeline_tag else ""
    header = (
        f"{model.id}{task}: a model repository by {author} on Hugging Face, "
        f"created {model.created_at:%Y-%m-%d}."
    )
    body = clean_markdown(card.text, CARD_CHARS) if card else ""
    return Document(
        HF_MODEL_CARD,
        model.id,
        f"{header}\n\n{body}" if body else header,
        model.created_at + HF_PUBLIC_LAG,
        title=model.id,
        metadata={
            "url": f"https://huggingface.co/{model.id}",
            "author": author,
            "pipeline_tag": model.pipeline_tag,
            "created_at": model.created_at.isoformat(),
            "gated": model.gated,
            "card_revision": card.revision if card else None,
            "card_committed_at": card.committed_at.isoformat() if card else None,
        },
    )


def release_document(release: Release) -> Document:
    title = f"{release.repo} {release.name or release.tag}"
    body = clean_markdown(release.body, RELEASE_CHARS)
    return Document(
        GITHUB_RELEASE,
        f"{release.repo}@{release.tag}",
        body or title,
        release.published_at,
        title=title,
        metadata={
            "url": release.url,
            "repo": release.repo,
            "tag": release.tag,
            "prerelease": release.prerelease,
        },
    )


# --- indexing -----------------------------------------------------------------------


def _midnight(d: date) -> datetime:
    return datetime.combine(d, time(), tzinfo=UTC)


def _newest(
    conn: Connection, document_type: str, repo: str | None = None
) -> datetime | None:
    sql = "SELECT max(published_at) FROM retrieval_documents WHERE document_type = %s"
    params: list[object] = [document_type]
    if repo is not None:
        sql += " AND metadata->>'repo' = %s"
        params.append(repo)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]  # type: ignore[index, no-any-return]


def index_arxiv(
    conn: Connection, client: ArxivClient, start: date, end: date
) -> UpsertResult:
    """Papers first submitted from `start` through `end`."""
    return upsert_documents(
        conn, map(arxiv_document, client.fetch_abstracts(start, end))
    )


def _settled_models(conn: Connection, now: datetime) -> set[str]:
    """Indexed models whose card date has passed: their card can't change."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_id FROM retrieval_documents"
            " WHERE document_type = %s AND published_at <= %s",
            (HF_MODEL_CARD, now),
        )
        return {row[0] for row in cur.fetchall()}


def index_hf_author(
    conn: Connection,
    client: HuggingFaceClient,
    author: str,
    start: date,
    now: datetime,
) -> UpsertResult:
    """One lab's models created since `start`, skipping derived copies and
    models already settled."""
    settled = _settled_models(conn, now)
    documents = []
    for model in client.iter_author_models(author):
        if model.created_at < _midnight(start) or DERIVED.search(model.id):
            continue
        if model.id in settled:
            continue
        card = None
        if not model.gated:
            card = client.card_as_of(
                model.id, min(model.created_at + HF_PUBLIC_LAG, now)
            )
        documents.append(model_document(model, card))
    return upsert_documents(conn, documents)


def index_releases(
    conn: Connection, client: GitHubClient, repo: str, since: datetime
) -> UpsertResult:
    """Stable releases of `repo` published since `since`."""
    releases = (r for r in client.iter_releases(repo, since) if not r.prerelease)
    return upsert_documents(conn, map(release_document, releases))


@dataclass
class CorpusResult:
    by_source: dict[str, UpsertResult] = field(default_factory=dict)

    def add(self, source: str, result: UpsertResult) -> None:
        old = self.by_source.get(source, UpsertResult(0, 0, 0))
        self.by_source[source] = UpsertResult(*(a + b for a, b in zip(old, result)))


def build_corpus(
    conn: Connection,
    sources: Iterable[str] = SOURCES,
    *,
    now: datetime,
    start: date | None = None,
    arxiv: Callable[[], ArxivClient] = ArxivClient.from_env,
    huggingface: Callable[[], HuggingFaceClient] = HuggingFaceClient.from_env,
    github: Callable[[], GitHubClient] = GitHubClient.from_env,
    authors: Sequence[str] = HF_AUTHORS,
    repos: Sequence[str] = RELEASE_REPOS,
    progress: Callable[[str, UpsertResult], None] | None = None,
) -> CorpusResult:
    """Index `sources` up to `now`, committing as each finishes (see module doc).

    `start` forces the earliest date indexed; by default each source resumes
    from what's already there, or from CORPUS_START.
    """
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    result = CorpusResult()

    def done(label: str, source: str, r: UpsertResult) -> None:
        conn.commit()
        result.add(source, r)
        if progress:
            progress(label, r)

    if "arxiv" in sources:
        newest = _newest(conn, ARXIV_ABSTRACT)
        since = start or (
            (newest - ARXIV_ANNOUNCE_LAG - ARXIV_OVERLAP).date()
            if newest
            else CORPUS_START
        )
        with arxiv() as ax:
            done("arxiv", "arxiv", index_arxiv(conn, ax, since, now.date()))
    if "huggingface" in sources:
        with huggingface() as hf:
            for author in authors:
                r = index_hf_author(conn, hf, author, start or CORPUS_START, now)
                done(f"huggingface {author}", "huggingface", r)
    if "github" in sources:
        with github() as gh:
            for repo in repos:
                newest = _newest(conn, GITHUB_RELEASE, repo)
                since = _midnight(start) if start else newest or _midnight(CORPUS_START)
                done(
                    f"github {repo}",
                    "github",
                    index_releases(conn, gh, repo, since),
                )
    if "polymarket" in sources:
        done("polymarket", "polymarket", index_platform_questions(conn))
    return result


def _print(label: str, r: UpsertResult) -> None:
    print(
        f"{label}: inserted={r.inserted} updated={r.updated} unchanged={r.unchanged}",
        flush=True,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    parser = argparse.ArgumentParser(
        description="Index arXiv, Hugging Face, GitHub and Polymarket text for retrieval."
    )
    parser.add_argument(
        "--sources", nargs="+", choices=SOURCES, default=list(SOURCES),
        help="sources to index (default: all)",
    )  # fmt: skip
    parser.add_argument(
        "--start", type=date.fromisoformat,
        help="earliest date to index (default: resume each source)",
    )  # fmt: skip
    args = parser.parse_args(argv)
    try:
        with connection() as conn:
            result = build_corpus(
                conn, args.sources, now=now(), start=args.start, progress=_print
            )
    except (ConfigError, SourceAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for source, r in result.by_source.items():
        _print(f"total {source}", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
