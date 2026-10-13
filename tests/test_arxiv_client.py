"""Tests for the arXiv client and ARXIV_AI_SUBMISSIONS ingestion.

Client tests replay sample Atom feeds offline; ingestion tests use a fake
client and a throwaway database (skipped if Postgres isn't reachable).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import ingest_arxiv
from eco_prediction.ai.ingest_arxiv import (
    SERIES,
    ingest_backfill,
    ingest_live,
    live_window,
)
from eco_prediction.ai.sources.arxiv_client import (
    ArxivClient,
    MonthlyCount,
    iter_months,
    last_complete_month,
    parse_total_results,
    submissions_query,
)
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.data.fred_client import RateLimiter
from eco_prediction.db.migrate import migrate
from eco_prediction.db.queries import get_series_value_as_of

JAN, FEB, MAR, APR = (date(2024, m, 1) for m in (1, 2, 3, 4))


def feed(total: int) -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/</id>
  <title>arXiv Search Results</title>
  <opensearch:itemsPerPage>1</opensearch:itemsPerPage>
  <opensearch:totalResults>{total}</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry><id>http://arxiv.org/abs/2401.00001v1</id><title>A paper</title></entry>
</feed>"""


# The shape arXiv really sends, with status 500 for an overloaded server.
ERROR_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/</id>
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>https://arxiv.org/api/errors</id>
    <title>Error</title>
    <summary>The server encountered an internal error.</summary>
  </entry>
</feed>"""


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code
        self.headers: dict[str, str] = {}


class FakeSession:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def client_with(
    *responses: FakeResponse, sleeps: list[float] | None = None
) -> tuple[ArxivClient, FakeSession]:
    session = FakeSession(*responses)
    client = ArxivClient(
        session=session,  # type: ignore[arg-type]
        rate_limiter=RateLimiter(max_calls=1000, period=60),
        sleep=(sleeps if sleeps is not None else []).append,
    )
    return client, session


class FakeArxiv(ArxivClient):
    """Answers count_submissions from a dict and records the months asked."""

    def __init__(self, counts: dict[date, int]) -> None:
        super().__init__(sleep=lambda _: None)
        self.counts = counts
        self.asked: list[date] = []

    def count_submissions(self, month: date, categories: Any = None) -> int:
        self.asked.append(month)
        return self.counts[month]


# --- client -----------------------------------------------------------------


def test_query_covers_the_whole_month_in_every_category() -> None:
    assert submissions_query(date(2024, 2, 17), ("cs.AI", "cs.LG", "cs.CL")) == (
        "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND "
        "submittedDate:[202402010000 TO 202402292359]"
    )
    assert submissions_query(date(2023, 12, 5), ("cs.CL",)).endswith(
        "[202312010000 TO 202312312359]"
    )


def test_count_submissions_reads_total_results() -> None:
    client, session = client_with(FakeResponse(feed(4066)))

    assert client.count_submissions(JAN) == 4066
    call = session.calls[0]
    assert call["url"] == "https://export.arxiv.org/api/query"
    # max_results=0 makes arXiv answer 500, so one result is requested.
    assert call["params"]["max_results"] == "1"
    assert "cat:cs.LG" in call["params"]["search_query"]


def test_monthly_counts_one_query_per_month() -> None:
    client, session = client_with(*(FakeResponse(feed(n)) for n in (10, 20, 30)))

    assert client.monthly_counts(date(2023, 12, 9), date(2024, 2, 1)) == [
        MonthlyCount(date(2023, 12, 1), 10),
        MonthlyCount(JAN, 20),
        MonthlyCount(FEB, 30),
    ]
    assert len(session.calls) == 3


def test_error_feed_with_status_200_raises() -> None:
    with pytest.raises(SourceAPIError, match="internal error"):
        parse_total_results(ERROR_FEED)


def test_error_feed_with_status_500_is_retried_then_reported() -> None:
    client, session = client_with(FakeResponse(ERROR_FEED, 500), FakeResponse(feed(5)))
    assert client.count_submissions(JAN) == 5
    assert len(session.calls) == 2

    client, _ = client_with(*[FakeResponse(ERROR_FEED, 500)] * 7)
    with pytest.raises(SourceAPIError, match="500: The server encountered"):
        client.count_submissions(JAN)


def test_rate_limited_backoff_outlasts_a_block_of_minutes() -> None:
    sleeps: list[float] = []
    client, _ = client_with(
        *[FakeResponse("", 429)] * 6, FakeResponse(feed(5)), sleeps=sleeps
    )

    assert client.count_submissions(JAN) == 5
    assert sleeps == [15.0, 30.0, 60.0, 120.0, 240.0, 300.0]
    assert ArxivClient().rate_limiter.period == 5.0


def test_malformed_feeds_raise() -> None:
    with pytest.raises(SourceAPIError, match="Invalid XML"):
        parse_total_results("<html>maintenance</html")
    with pytest.raises(SourceAPIError, match="no totalResults"):
        parse_total_results(
            '<feed xmlns="http://www.w3.org/2005/Atom"><id>x</id></feed>'
        )


def test_month_helpers() -> None:
    assert list(iter_months(date(2023, 11, 30), date(2024, 1, 1))) == [
        date(2023, 11, 1),
        date(2023, 12, 1),
        JAN,
    ]
    assert list(iter_months(FEB, JAN)) == []
    assert last_complete_month(date(2024, 3, 1)) == FEB
    assert last_complete_month(date(2024, 3, 31)) == FEB
    assert last_complete_month(date(2024, 1, 15)) == date(2023, 12, 1)
    assert live_window(date(2024, 4, 12), 3) == (JAN, MAR)
    assert live_window(date(2024, 4, 12), 1) == (MAR, MAR)


# --- ingestion --------------------------------------------------------------


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def stored(conn: Connection) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.observed_at, o.as_of, o.value::int, o.reconstructed
            FROM observations o JOIN series s ON s.id = o.series_id
            WHERE s.series_id = %s
            ORDER BY o.observed_at, o.as_of
            """,
            (SERIES.series_id,),
        )
        return cur.fetchall()


def test_backfill_stores_reconstructed_rows_known_after_each_month(
    migrated: Connection,
) -> None:
    result = ingest_backfill(migrated, FakeArxiv({JAN: 100, FEB: 120}), JAN, FEB)

    assert (result.fetched, result.inserted) == (2, 2)
    assert stored(migrated) == [
        (JAN, FEB, 100, True),
        (FEB, MAR, 120, True),
    ]
    with migrated.cursor() as cur:
        cur.execute(
            """
            SELECT s.domain, s.units, s.frequency, src.name
            FROM series s JOIN sources src ON src.id = s.source_id
            WHERE s.series_id = %s
            """,
            (SERIES.series_id,),
        )
        assert cur.fetchone() == ("ai", "Papers", "Monthly", "arXiv")
    # A backtest standing on the last day of January can't see January yet.
    assert (
        get_series_value_as_of(migrated, SERIES.series_id, JAN, date(2024, 1, 31))
        is None
    )
    assert get_series_value_as_of(migrated, SERIES.series_id, JAN, FEB) == 100


def test_backfill_skips_stored_months_so_it_resumes(migrated: Connection) -> None:
    ingest_backfill(migrated, FakeArxiv({JAN: 100}), JAN, JAN)
    fake = FakeArxiv({JAN: 999, FEB: 120, MAR: 130})

    result = ingest_backfill(migrated, fake, JAN, MAR)

    assert fake.asked == [FEB, MAR]
    assert (result.fetched, result.inserted) == (2, 2)
    assert stored(migrated)[0] == (JAN, FEB, 100, True)


def test_live_stores_new_and_changed_months_only(migrated: Connection) -> None:
    ingest_backfill(migrated, FakeArxiv({JAN: 100, FEB: 120}), JAN, FEB)
    today = date(2024, 4, 12)

    # JAN unchanged, FEB revised (a late cross-list), MAR new.
    result = ingest_live(migrated, FakeArxiv({JAN: 100, FEB: 121, MAR: 130}), today)

    assert (result.fetched, result.inserted) == (3, 2)
    assert stored(migrated) == [
        (JAN, FEB, 100, True),
        (FEB, MAR, 120, True),
        (FEB, today, 121, False),
        (MAR, today, 130, False),
    ]

    # Same counts the next day: nothing new.
    again = ingest_live(
        migrated, FakeArxiv({JAN: 100, FEB: 121, MAR: 130}), date(2024, 4, 13)
    )
    assert again.inserted == 0


def test_zero_count_is_refused(migrated: Connection) -> None:
    with pytest.raises(SourceAPIError, match="refusing to store"):
        ingest_backfill(migrated, FakeArxiv({JAN: 100, FEB: 0}), JAN, FEB)
    with pytest.raises(SourceAPIError, match="refusing to store"):
        ingest_live(migrated, FakeArxiv({JAN: 0, FEB: 1, MAR: 1}), date(2024, 4, 2))


# --- command line -----------------------------------------------------------


def test_dry_run_prints_rows_without_a_database(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_connected() -> None:
        raise AssertionError("dry run touched the database")

    fake = FakeArxiv({JAN: 100, FEB: 120, MAR: 130})
    monkeypatch.setattr(ingest_arxiv, "connection", fail_if_connected)

    code = ingest_arxiv.main(
        ["--backfill", "--start", "2024-01-01", "--dry-run"],
        client_factory=lambda: fake,
        today=lambda: APR,
    )
    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        "2024-01      100  as_of=2024-02-01  reconstructed",
        "2024-02      120  as_of=2024-03-01  reconstructed",
        "2024-03      130  as_of=2024-04-01  reconstructed",
    ]

    code = ingest_arxiv.main(
        ["--dry-run", "--months", "2"],
        client_factory=lambda: fake,
        today=lambda: date(2024, 4, 12),
    )
    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        "2024-02      120  as_of=2024-04-12  live",
        "2024-03      130  as_of=2024-04-12  live",
    ]


def test_api_failure_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    class Down(FakeArxiv):
        def count_submissions(self, month: date, categories: Any = None) -> int:
            raise SourceAPIError("arxiv query failed after 6 attempts", source="arxiv")

    code = ingest_arxiv.main(
        ["--dry-run"], client_factory=lambda: Down({}), today=lambda: APR
    )
    assert code == 1
    assert "error: arxiv query failed" in capsys.readouterr().err
