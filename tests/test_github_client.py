"""Tests for the GitHub client and the GITHUB_AI_ACTIVITY / GITHUB_AI_STARS series.

A fake session answers searches from tables, offline; ingestion tests use a
throwaway database (skipped if Postgres isn't reachable).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import date
from typing import Any

import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import ingest_github, monthly_ingest
from eco_prediction.ai.ingest_github import STARS_SERIES, ingest_stars_snapshot
from eco_prediction.ai.monthly_ingest import ingest_backfill
from eco_prediction.ai.months import MonthlyCount
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.sources.github_client import (
    GitHubClient,
    created_range,
    topic_query,
)
from eco_prediction.data.fred_client import RateLimiter
from eco_prediction.db.migrate import migrate

JAN = date(2024, 1, 1)
JAN_RANGE = "2024-01-01..2024-01-31"
TOKEN = "ghp_secret"


class FakeResponse:
    def __init__(
        self, body: Any, status_code: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.text = str(body)
        self.links: dict[str, Any] = {}

    def json(self) -> Any:
        return self._body


class SearchSession:
    """Answers each search with `answer(params)`, recording every call."""

    def __init__(self, answer: Callable[[dict[str, str]], FakeResponse]) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.answer(kwargs["params"])

    def close(self) -> None:
        pass


def counts(table: dict[str, int]) -> Callable[[dict[str, str]], FakeResponse]:
    return lambda p: FakeResponse(
        {"total_count": table[p["q"]], "incomplete_results": False, "items": []}
    )


def client_with(
    answer: Callable[[dict[str, str]], FakeResponse],
    *,
    api_key: str | None = TOKEN,
    sleeps: list[float] | None = None,
    now: float = 1_000.0,
) -> tuple[GitHubClient, SearchSession]:
    session = SearchSession(answer)
    client = GitHubClient(
        api_key=api_key,
        session=session,  # type: ignore[arg-type]
        rate_limiter=RateLimiter(max_calls=1000, period=60),
        sleep=(sleeps if sleeps is not None else []).append,
        now=lambda: now,
    )
    return client, session


# Jan 2024: |ai|=1044, |ml|=900, |dl|=786, pairwise and triple overlaps.
JAN_TABLE = {
    f"topic:ai created:{JAN_RANGE}": 1044,
    f"topic:machine-learning created:{JAN_RANGE}": 900,
    f"topic:deep-learning created:{JAN_RANGE}": 786,
    f"topic:ai topic:machine-learning created:{JAN_RANGE}": 120,
    f"topic:ai topic:deep-learning created:{JAN_RANGE}": 80,
    f"topic:machine-learning topic:deep-learning created:{JAN_RANGE}": 300,
    f"topic:ai topic:machine-learning topic:deep-learning created:{JAN_RANGE}": 19,
}


# --- new repos --------------------------------------------------------------


def test_query_helpers() -> None:
    assert created_range(date(2024, 2, 10)) == "2024-02-01..2024-02-29"
    assert topic_query(["ai", "deep-learning"], JAN_RANGE) == (
        f"topic:ai topic:deep-learning created:{JAN_RANGE}"
    )


def test_new_repos_are_the_union_of_the_topics() -> None:
    client, session = client_with(counts(JAN_TABLE))

    # 1044 + 900 + 786 - 120 - 80 - 300 + 19
    assert client.count_new_repos(JAN) == 2249
    assert len(session.calls) == 7
    call = session.calls[0]
    assert call["url"] == "https://api.github.com/search/repositories"
    assert call["params"]["per_page"] == "1"
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert call["headers"]["X-GitHub-Api-Version"] == "2022-11-28"


def test_count_months_yields_one_count_per_month() -> None:
    client, _ = client_with(counts(JAN_TABLE))

    assert list(client.count_months([JAN])) == [MonthlyCount(JAN, 2249)]


def test_months_before_topics_existed_are_refused() -> None:
    client, session = client_with(counts({}))

    with pytest.raises(ValueError, match="before 2017-01-01"):
        client.count_new_repos(date(2016, 12, 1))
    assert session.calls == []


def test_incomplete_results_are_retried_then_refused() -> None:
    answers = iter(
        [
            {"total_count": 3, "incomplete_results": True},
            {"total_count": 5, "incomplete_results": False},
        ]
    )
    sleeps: list[float] = []
    client, _ = client_with(lambda p: FakeResponse(next(answers)), sleeps=sleeps)
    assert client.count_repos("topic:ai") == 5
    assert sleeps == [1.0]

    client, session = client_with(
        lambda p: FakeResponse({"total_count": 3, "incomplete_results": True})
    )
    with pytest.raises(SourceAPIError, match="stayed incomplete after 4 attempts"):
        client.count_repos("topic:ai")
    assert len(session.calls) == 4


def test_validation_error_raises_with_githubs_message() -> None:
    client, _ = client_with(
        lambda p: FakeResponse({"message": "Validation Failed"}, 422)
    )

    with pytest.raises(SourceAPIError, match=r"\(422\): Validation Failed"):
        client.count_repos("topic:ai OR topic:ml")


# --- rate limits and auth ---------------------------------------------------


def test_exhausted_rate_limit_403_waits_until_reset() -> None:
    answers = iter(
        [
            FakeResponse(
                {"message": "API rate limit exceeded"},
                403,
                {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1042"},
            ),
            FakeResponse({"total_count": 7, "incomplete_results": False}),
        ]
    )
    sleeps: list[float] = []
    client, _ = client_with(lambda p: next(answers), sleeps=sleeps, now=1_000.0)

    assert client.count_repos("topic:ai") == 7
    assert sleeps == [43.0]  # reset in 42s, plus a second


def test_retry_after_wins_for_secondary_limits() -> None:
    answers = iter(
        [
            FakeResponse(
                {"message": "secondary rate limit"},
                429,
                {"Retry-After": "60", "X-RateLimit-Remaining": "0"},
            ),
            FakeResponse({"total_count": 7, "incomplete_results": False}),
        ]
    )
    sleeps: list[float] = []
    client, _ = client_with(lambda p: next(answers), sleeps=sleeps)

    assert client.count_repos("topic:ai") == 7
    assert sleeps == [60.0]


def test_other_403s_are_not_retried() -> None:
    client, session = client_with(
        lambda p: FakeResponse(
            {"message": "Resource not accessible"}, 403, {"X-RateLimit-Remaining": "12"}
        )
    )

    with pytest.raises(SourceAPIError, match=r"\(403\)"):
        client.count_repos("topic:ai")
    assert len(session.calls) == 1


def test_rate_limit_depends_on_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    assert GitHubClient(api_key=TOKEN).rate_limiter.max_calls == 27
    assert GitHubClient().rate_limiter.max_calls == 9

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_env")
    assert GitHubClient.from_env().api_key == "ghp_env"
    monkeypatch.setenv("GITHUB_TOKEN", "")
    client = GitHubClient.from_env()
    assert client.api_key is None
    assert "Authorization" not in client._auth_headers()


# --- stars ------------------------------------------------------------------


def star_pages(
    by_topic: dict[str, list[tuple[int, int]]], page_size: int = 100
) -> Callable[[dict[str, str]], FakeResponse]:
    def answer(p: dict[str, str]) -> FakeResponse:
        assert p["sort"] == "stars" and p["order"] == "desc"
        repos = by_topic[p["q"].removeprefix("topic:")]
        page = int(p["page"])
        chunk = repos[(page - 1) * page_size : page * page_size]
        return FakeResponse(
            {
                "total_count": len(repos),
                "incomplete_results": False,
                "items": [{"id": i, "stargazers_count": s} for i, s in chunk],
            }
        )

    return answer


def test_top_starred_merges_topics_without_double_counting() -> None:
    by_topic = {
        "ai": [(1, 500), (2, 300), (3, 10)],
        "machine-learning": [(4, 900), (1, 500), (5, 20)],
        "deep-learning": [(4, 900), (6, 400)],
    }
    client, session = client_with(star_pages(by_topic))

    assert client.top_starred(n=3) == [(4, 900), (1, 500), (6, 400)]
    # One short page per topic ends that topic.
    assert len(session.calls) == 3
    assert client.top_starred_total(n=3) == 1800


def test_top_starred_pages_through_full_pages() -> None:
    repos = [(i, 10_000 - i) for i in range(250)]
    client, session = client_with(
        star_pages({"ai": repos, "machine-learning": [], "deep-learning": []})
    )

    top = client.top_starred(n=1000)

    assert len(top) == 250
    ai_pages = [
        c["params"]["page"] for c in session.calls if c["params"]["q"] == "topic:ai"
    ]
    assert ai_pages == ["1", "2", "3"]
    assert session.calls[0]["params"]["per_page"] == "100"


def test_top_n_beyond_the_search_cap_is_refused() -> None:
    client, _ = client_with(counts({}))
    with pytest.raises(ValueError, match="1000"):
        client.top_starred(n=1001)


# --- ingestion --------------------------------------------------------------


class FakeGitHub:
    def __init__(self, months: dict[date, int] | None = None, stars: int = 0) -> None:
        self.months = months or {}
        self.stars = stars
        self.star_calls = 0

    def count_months(self, months: Sequence[date]) -> Iterator[MonthlyCount]:
        for m in months:
            yield MonthlyCount(m, self.months[m])

    def top_starred_total(self) -> int:
        self.star_calls += 1
        return self.stars

    def close(self) -> None:
        pass


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def rows(conn: Connection, series_id: str) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.observed_at, o.as_of, o.value::bigint, o.reconstructed, s.domain
            FROM observations o JOIN series s ON s.id = o.series_id
            WHERE s.series_id = %s
            ORDER BY o.observed_at, o.as_of
            """,
            (series_id,),
        )
        return cur.fetchall()


def test_activity_backfill_is_reconstructed(migrated: Connection) -> None:
    feb = date(2024, 2, 1)
    ingest_backfill(
        migrated, ingest_github.SERIES, FakeGitHub({JAN: 2249, feb: 2400}), JAN, feb
    )

    assert rows(migrated, "GITHUB_AI_ACTIVITY") == [
        (JAN, feb, 2249, True, "ai"),
        (feb, date(2024, 3, 1), 2400, True, "ai"),
    ]


def test_stars_snapshot_once_per_month(migrated: Connection) -> None:
    fake = FakeGitHub(stars=48_000_000)

    first = ingest_stars_snapshot(migrated, fake, date(2026, 10, 12))
    again = ingest_stars_snapshot(migrated, fake, date(2026, 10, 30))
    november = ingest_stars_snapshot(
        migrated, FakeGitHub(stars=48_500_000), date(2026, 11, 2)
    )

    assert (first.inserted, again.fetched, november.inserted) == (1, 0, 1)
    assert fake.star_calls == 1  # the second October run didn't search at all
    assert rows(migrated, STARS_SERIES.series_id) == [
        (date(2026, 10, 1), date(2026, 10, 12), 48_000_000, False, "ai"),
        (date(2026, 11, 1), date(2026, 11, 2), 48_500_000, False, "ai"),
    ]
    with migrated.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM sources WHERE name = 'GitHub'")
        assert cur.fetchone() == (1,)


def test_zero_stars_are_refused(migrated: Connection) -> None:
    with pytest.raises(SourceAPIError, match="refusing to store"):
        ingest_stars_snapshot(migrated, FakeGitHub(stars=0), date(2026, 10, 12))


# --- command line -----------------------------------------------------------


def test_cli_routes_stars_and_activity(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_db() -> None:
        raise AssertionError("dry run touched the database")

    monkeypatch.setattr(monthly_ingest, "connection", no_db)
    monkeypatch.setattr(ingest_github, "connection", no_db)
    fake = FakeGitHub(
        {date(2026, 9, 1): 5000, date(2026, 8, 1): 4800}, stars=48_000_000
    )

    assert (
        ingest_github.main(
            ["stars", "--dry-run"],
            client_factory=lambda: fake,
            today=lambda: date(2026, 10, 12),
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == (
        "GITHUB_AI_STARS 2026-10  48000000  as_of=2026-10-12"
    )

    assert (
        ingest_github.main(
            ["--dry-run", "--months", "2"],
            client_factory=lambda: fake,
            today=lambda: date(2026, 10, 12),
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [
        "2026-08     4800  as_of=2026-10-12  live",
        "2026-09     5000  as_of=2026-10-12  live",
    ]
