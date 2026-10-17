"""Tests for the retrieval corpus builder and the source methods behind it
(AI task 17).

Client tests replay fake HTTP responses offline. Corpus tests run fake
clients against a throwaway database (skipped if Postgres isn't reachable).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any, Self

import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.llm import build_corpus as bc
from eco_prediction.ai.llm.build_corpus import (
    HF_MODEL_CARD,
    arxiv_document,
    build_corpus,
    clean_markdown,
    model_document,
    release_document,
)
from eco_prediction.ai.llm.retrieval import (
    Document,
    retrieve_documents,
    upsert_documents,
)
from eco_prediction.ai.sources.arxiv_client import ArxivClient, ArxivPaper, parse_papers
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.sources.github_client import GitHubClient, Release
from eco_prediction.ai.sources.huggingface_client import (
    Card,
    HFModel,
    HuggingFaceClient,
)
from eco_prediction.data.fred_client import RateLimiter
from eco_prediction.db.migrate import migrate


def utc(y: int, m: int, d: int, h: int = 0) -> datetime:
    return datetime(y, m, d, h, tzinfo=UTC)


class FakeResponse:
    def __init__(
        self, body: Any, status_code: int = 200, next_url: str | None = None
    ) -> None:
        self._body = body
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.links = {"next": {"url": next_url}} if next_url else {}
        self.text = body if isinstance(body, str) else str(body)

    def json(self) -> Any:
        return self._body


class FakeSession:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs.get("params") or {}))
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def client(cls: type, *responses: FakeResponse, **kwargs: Any) -> Any:
    session = FakeSession(*responses)
    c = cls(
        session=session,
        rate_limiter=RateLimiter(max_calls=1000, period=60),
        sleep=lambda s: None,
        max_retries=0,
        **kwargs,
    )
    return c, session


# --- arXiv --------------------------------------------------------------------------


def entry(arxiv_id: str, version: int, title: str, published: str) -> str:
    return f"""<entry>
    <id>http://arxiv.org/abs/{arxiv_id}v{version}</id>
    <title>{title}
      (wrapped)</title>
    <updated>2025-01-01T00:00:00Z</updated>
    <published>{published}</published>
    <summary>  An abstract about
      {title}.  </summary>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="cs.CL"/>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
  </entry>"""


def feed(total: int, *entries: str) -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>{total}</opensearch:totalResults>
  {"".join(entries)}
</feed>"""


A = entry("2401.00001", 1, "SWE-bench agents", "2024-01-02T10:00:00Z")
B = entry("2401.00002", 3, "GPQA results", "2024-01-05T10:00:00Z")
C = entry("2402.00003", 1, "SWE-bench and GPQA", "2024-02-01T10:00:00Z")


def test_papers_are_parsed_from_the_feed() -> None:
    a, b = parse_papers(feed(2, A, B))
    assert (a.arxiv_id, a.version, b.version) == ("2401.00001", 1, 3)
    assert a.title == "SWE-bench agents (wrapped)"
    assert a.abstract == "An abstract about SWE-bench agents."
    assert a.published == utc(2024, 1, 2, 10)
    assert a.primary_category == "cs.CL" and a.categories == ("cs.CL", "cs.AI")
    assert a.authors == ("Ada Lovelace", "Alan Turing")


def test_searches_page_oldest_first_and_retry_an_empty_page() -> None:
    arxiv, session = client(
        ArxivClient,
        FakeResponse(feed(3, A, B)),
        FakeResponse(feed(3)),  # arXiv's transient empty page
        FakeResponse(feed(3, C)),
    )
    papers = list(arxiv.iter_search("q", page_size=2))
    assert [p.arxiv_id for p in papers] == ["2401.00001", "2401.00002", "2402.00003"]
    starts = [params["start"] for _, params in session.calls]
    assert starts == ["0", "2", "2"]
    assert session.calls[0][1]["sortBy"] == "submittedDate"


def test_an_empty_page_that_persists_raises() -> None:
    arxiv, _ = client(ArxivClient, *[FakeResponse(feed(3)) for _ in range(3)])
    with pytest.raises(SourceAPIError, match="no results at 0 of 3"):
        list(arxiv.iter_search("q"))


def test_abstracts_are_deduplicated_across_topics() -> None:
    arxiv, session = client(
        ArxivClient, FakeResponse(feed(2, A, C)), FakeResponse(feed(2, B, C))
    )
    topics = {"swe_bench": 'abs:"SWE-bench"', "gpqa": "abs:GPQA"}
    papers = list(arxiv.fetch_abstracts(date(2024, 1, 1), date(2024, 2, 29), topics))
    assert [(p.arxiv_id, p.topics) for p in papers] == [
        ("2401.00001", ("swe_bench",)),
        ("2401.00002", ("gpqa",)),
        ("2402.00003", ("swe_bench", "gpqa")),
    ]
    query = session.calls[0][1]["search_query"]
    assert query == (
        '(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND (abs:"SWE-bench") AND '
        "submittedDate:[202401010000 TO 202402292359]"
    )


# --- Hugging Face ---------------------------------------------------------------------

LISTING = "https://huggingface.co/api/models?author=lab&cursor=2"


def test_author_models_are_listed_across_pages() -> None:
    hf, session = client(
        HuggingFaceClient,
        FakeResponse(
            [{"id": "lab/a", "createdAt": "2025-01-01T00:00:00.000Z",
              "pipeline_tag": "text-generation", "tags": ["x"], "gated": False}],
            next_url=LISTING,
        ),
        FakeResponse([{"id": "lab/b", "createdAt": "2025-02-01T00:00:00.000Z",
                       "gated": "manual"}]),
    )  # fmt: skip
    a, b = hf.iter_author_models("lab")
    assert a == HFModel("lab/a", utc(2025, 1, 1), "text-generation", ("x",), False)
    assert b.gated and b.pipeline_tag is None
    assert session.calls[0][1]["author"] == "lab" and session.calls[1][0] == LISTING


def commit(sha: str, when: str) -> dict[str, str]:
    return {"id": sha, "date": when}


def test_a_card_is_the_readme_at_the_latest_commit_before_the_date() -> None:
    hf, session = client(
        HuggingFaceClient,
        FakeResponse(
            [commit("c3", "2025-03-01T00:00:00.000Z"), commit("c2", "2025-01-20T00:00:00.000Z")],
            next_url="https://huggingface.co/api/models/lab/a/commits/main?p=1",
        ),
        FakeResponse([commit("c1", "2025-01-01T00:00:00.000Z")]),
        FakeResponse("# Model A\nthe card as of c2"),
    )  # fmt: skip
    card = hf.card_as_of("lab/a", utc(2025, 1, 31))
    assert card == Card("# Model A\nthe card as of c2", "c2", utc(2025, 1, 20))
    assert session.calls[-1][0] == "https://huggingface.co/lab/a/raw/c2/README.md"


def test_gated_models_and_missing_readmes_have_no_card() -> None:
    gated = FakeResponse({"error": "Access to model lab/g is restricted."}, 401)
    hf, _ = client(HuggingFaceClient, gated)
    assert hf.card_as_of("lab/g", utc(2025, 1, 31)) is None
    no_readme = FakeResponse("Entry not found", 404)
    hf, _ = client(
        HuggingFaceClient,
        FakeResponse([commit("c1", "2025-01-01T00:00:00.000Z")]),
        no_readme,
    )
    assert hf.card_as_of("lab/a", utc(2025, 1, 31)) is None
    # No commit by the date: no card, and no README request.
    hf, session = client(
        HuggingFaceClient, FakeResponse([commit("c1", "2025-02-01T00:00:00.000Z")])
    )
    assert hf.card_as_of("lab/a", utc(2025, 1, 31)) is None
    assert len(session.calls) == 1


# --- GitHub ---------------------------------------------------------------------------


def release(tag: str, published: str | None, **kwargs: Any) -> dict[str, Any]:
    return {
        "tag_name": tag, "name": kwargs.get("name", tag), "body": kwargs.get("body"),
        "published_at": published, "html_url": f"https://github.com/o/r/releases/tag/{tag}",
        "draft": kwargs.get("draft", False), "prerelease": kwargs.get("prerelease", False),
    }  # fmt: skip


def test_releases_skip_drafts_and_stop_before_since() -> None:
    gh, session = client(
        GitHubClient,
        FakeResponse(
            [release("v3-draft", None, draft=True),
             release("v3", "2025-03-01T00:00:00Z", body="notes")],
            next_url="https://api.github.com/repositories/1/releases?page=2",
        ),
        FakeResponse([release("v2", "2025-02-01T00:00:00Z"),
                      release("v1", "2025-01-01T00:00:00Z")]),
        api_key="t",
    )  # fmt: skip
    got = list(gh.iter_releases("o/r", since=utc(2025, 1, 15)))
    assert [(r.tag, r.body) for r in got] == [("v3", "notes"), ("v2", "")]
    assert len(session.calls) == 2  # stopped inside page 2


# --- documents --------------------------------------------------------------------------


def test_markdown_is_cleaned_and_cut_at_a_word() -> None:
    card = (
        "---\nlicense: mit\ntags: [x]\n---\n<div align='center'><img src='a.png'></div>\n"
        "# Title\n\n\n\n![logo](l.png) Some **text** here.\r\nMore words follow"
    )
    assert (
        clean_markdown(card, 1000)
        == "# Title\n\n Some **text** here.\nMore words follow"
    )
    assert clean_markdown(card, 23) == "# Title\n\n Some **text** …"  # at a word end
    assert clean_markdown(card, 22) == "# Title\n\n Some …"  # mid-word
    assert clean_markdown("--- not front matter", 100) == "--- not front matter"


PAPER = ArxivPaper(
    "2401.00002", 3, "GPQA results", "An abstract.", utc(2024, 1, 5, 10),
    utc(2025, 1, 1), "cs.CL", ("cs.CL",), ("Ada",), ("gpqa",),
)  # fmt: skip
MODEL = HFModel("lab/m", utc(2025, 1, 1), "text-generation")


def test_documents_are_dated_conservatively() -> None:
    d = arxiv_document(PAPER)
    assert d.published_at == utc(2024, 1, 8, 10)  # announced, not submitted
    assert (
        d.metadata["revised"]
        and d.metadata["submitted_at"] == "2024-01-05T10:00:00+00:00"
    )
    card = Card("---\na: b\n---\n# M\nfacts", "c2", utc(2025, 1, 10))
    m = model_document(MODEL, card)
    assert m.published_at == utc(2025, 1, 31)  # 30 days after the repo was created
    assert m.content.startswith("lab/m (text-generation): a model repository by lab")
    assert m.content.endswith("# M\nfacts") and m.metadata["card_revision"] == "c2"
    gated = model_document(HFModel("lab/g", utc(2025, 1, 1), gated=True), None)
    assert (
        gated.content
        == "lab/g: a model repository by lab on Hugging Face, created 2025-01-01."
    )
    r = release_document(
        Release("o/r", "v1", None, "", utc(2025, 1, 1), "https://x", False)
    )
    assert (r.source_id, r.title, r.content) == ("o/r@v1", "o/r v1", "o/r v1")


# --- building the corpus ------------------------------------------------------------------


class FakeArxiv:
    def __init__(self, papers: list[ArxivPaper]) -> None:
        self.papers = papers
        self.calls: list[tuple[date, date]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def fetch_abstracts(self, start: date, end: date) -> Iterator[ArxivPaper]:
        self.calls.append((start, end))
        return iter(p for p in self.papers if start <= p.published.date() <= end)


class FakeHF(FakeArxiv):
    def __init__(self, models: dict[str, list[HFModel]]) -> None:
        self.models = models
        self.cards: list[tuple[str, datetime]] = []

    def iter_author_models(self, author: str) -> Iterator[HFModel]:
        return iter(self.models.get(author, []))

    def card_as_of(self, model_id: str, when: datetime) -> Card | None:
        self.cards.append((model_id, when))
        return Card(f"card of {model_id} as of {when:%Y-%m-%d}", "sha", when)


class FakeGitHub(FakeArxiv):
    def __init__(self, releases: dict[str, list[Release]]) -> None:
        self.releases = releases
        self.since: list[tuple[str, datetime]] = []

    def iter_releases(
        self, repo: str, since: datetime | None = None
    ) -> Iterator[Release]:
        assert since is not None
        self.since.append((repo, since))
        return iter(r for r in self.releases.get(repo, []) if r.published_at >= since)


def rel(tag: str, when: datetime, prerelease: bool = False) -> Release:
    return Release("o/r", tag, None, f"notes {tag}", when, "https://x", prerelease)


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def fake_clients(**kwargs: Any) -> dict[str, Any]:
    arxiv = FakeArxiv([PAPER, ArxivPaper(
        "2402.00009", 1, "SWE-bench later", "Later.", utc(2024, 2, 9), utc(2024, 2, 9),
        "cs.LG", ("cs.LG",), (), ("swe_bench",),
    )])  # fmt: skip
    hf = FakeHF({
        "lab": [
            HFModel("lab/old", utc(2022, 6, 1)),                  # before the start
            HFModel("lab/m-GGUF", utc(2025, 1, 1)),               # a derived copy
            HFModel("lab/m", utc(2025, 1, 1), "text-generation"),
            HFModel("lab/gated", utc(2025, 2, 1), gated=True),
            HFModel("lab/new", utc(2026, 10, 1)),                 # still in its window
        ]
    })  # fmt: skip
    gh = FakeGitHub(
        {"o/r": [rel("v2-rc1", utc(2025, 3, 1), True), rel("v1", utc(2025, 1, 1))]}
    )
    return {
        "arxiv": lambda: arxiv, "huggingface": lambda: hf, "github": lambda: gh,
        "authors": ["lab"], "repos": ["o/r"], **kwargs,
    }  # fmt: skip


NOW = utc(2026, 10, 16, 12)


def test_the_corpus_is_built_from_every_source(migrated: Connection) -> None:
    clients = fake_clients()
    result = build_corpus(migrated, now=NOW, **clients)
    assert {s: r.inserted for s, r in result.by_source.items()} == {
        "arxiv": 2, "huggingface": 3, "github": 1, "polymarket": 0,
    }  # fmt: skip
    hf = clients["huggingface"]()
    # Gated models fetch no card; a model in its window is read as of now.
    assert hf.cards == [("lab/m", utc(2025, 1, 31)), ("lab/new", NOW)]
    assert clients["arxiv"]().calls == [(date(2023, 1, 1), NOW.date())]

    docs = {
        d.source_id: d
        for d in retrieve_documents(migrated, date(2026, 12, 31), limit=50)
    }
    assert set(docs) == {
        "2401.00002",
        "2402.00009",
        "lab/m",
        "lab/gated",
        "lab/new",
        "o/r@v1",
    }
    assert docs["lab/new"].published_at == utc(2026, 10, 31)
    # Point in time: on 2026-10-16 the new model's card isn't public yet.
    today = retrieve_documents(migrated, date(2026, 10, 16), HF_MODEL_CARD, limit=50)
    assert {d.source_id for d in today} == {"lab/m", "lab/gated"}


def test_rebuilding_is_incremental_and_skips_settled_cards(
    migrated: Connection,
) -> None:
    build_corpus(migrated, now=NOW, **fake_clients())
    clients = fake_clients()
    result = build_corpus(migrated, now=NOW, **clients)
    assert result.by_source["arxiv"].unchanged == 1  # re-read a week before the newest
    assert clients["arxiv"]().calls == [
        (date(2024, 2, 9) - timedelta(days=7), NOW.date())
    ]
    assert clients["huggingface"]().cards == [
        ("lab/new", NOW)
    ]  # only the unsettled one
    assert clients["github"]().since == [("o/r", utc(2025, 1, 1))]
    assert result.by_source["github"] == (0, 0, 1)
    # --start re-reads from that date.
    clients = fake_clients()
    build_corpus(migrated, ["arxiv"], now=NOW, start=date(2024, 1, 1), **clients)
    assert clients["arxiv"]().calls == [(date(2024, 1, 1), NOW.date())]
    with pytest.raises(ValueError, match="unknown sources"):
        build_corpus(migrated, ["metaculus"], now=NOW)


def test_titles_outrank_long_bodies(migrated: Connection) -> None:
    long_body = "Which company has the best AI model? " * 30
    upsert_documents(
        migrated,
        [
            Document("t", "long", f"Rules. {long_body}", utc(2025, 1, 1),
                     title="Will the US ban an AI model?"),
            Document("t", "title", "Resolves by the leaderboard.", utc(2025, 1, 1),
                     title="Which company has the best AI model end of 2025?"),
        ],
    )  # fmt: skip
    hits = retrieve_documents(
        migrated, date(2025, 6, 1), query="best AI model end of year"
    )
    assert [d.source_id for d in hits] == ["title", "long"]


def test_main_reports_each_source(
    migrated: Connection,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_build(
        conn: Connection, sources: list[str], **kwargs: Any
    ) -> bc.CorpusResult:
        assert sources == ["github"] and kwargs["start"] == date(2025, 1, 1)
        result = bc.CorpusResult()
        result.add("github", bc.UpsertResult(2, 1, 0))
        return result

    monkeypatch.setattr(bc, "build_corpus", fake_build)
    monkeypatch.setattr(bc, "connection", lambda: _Ctx(migrated))
    assert bc.main(["--sources", "github", "--start", "2025-01-01"]) == 0
    assert "total github: inserted=2 updated=1 unchanged=0" in capsys.readouterr().out


class _Ctx:
    def __init__(self, conn: Connection) -> None:
        self.conn = conn

    def __enter__(self) -> Connection:
        return self.conn

    def __exit__(self, *exc: object) -> None:
        pass
