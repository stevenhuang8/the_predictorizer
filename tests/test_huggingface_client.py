"""Tests for the Hugging Face Hub client and HF_NEW_MODELS ingestion.

Client tests replay fake model listings offline; ingestion tests use a
throwaway database (skipped if Postgres isn't reachable).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date
from typing import Any

import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import ingest_huggingface
from eco_prediction.ai.ingest_huggingface import SERIES
from eco_prediction.ai.monthly_ingest import ingest_backfill, ingest_live
from eco_prediction.ai.months import MonthlyCount
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.sources.huggingface_client import FIRST_MONTH, HuggingFaceClient
from eco_prediction.data.fred_client import RateLimiter
from eco_prediction.db.migrate import migrate

JUL, AUG, SEP, OCT = (date(2026, m, 1) for m in (7, 8, 9, 10))
NEXT = "https://huggingface.co/api/models?sort=createdAt&cursor="


class FakeResponse:
    def __init__(
        self,
        body: Any,
        status_code: int = 200,
        next_url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.links = {"next": {"url": next_url, "rel": "next"}} if next_url else {}
        self.text = str(body)

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def models(*created: str) -> list[dict[str, str]]:
    return [{"id": f"org/m{i}", "createdAt": c} for i, c in enumerate(created)]


def pages(*batches: list[dict[str, str]]) -> list[FakeResponse]:
    """Linked pages: each but the last points to the next."""
    return [
        FakeResponse(b, next_url=f"{NEXT}{i + 1}" if i < len(batches) - 1 else None)
        for i, b in enumerate(batches)
    ]


def client_with(
    *responses: FakeResponse,
    api_key: str | None = None,
    sleeps: list[float] | None = None,
) -> tuple[HuggingFaceClient, FakeSession]:
    session = FakeSession(*responses)
    client = HuggingFaceClient(
        api_key=api_key,
        session=session,  # type: ignore[arg-type]
        rate_limiter=RateLimiter(max_calls=1000, period=60),
        sleep=(sleeps if sleeps is not None else []).append,
    )
    return client, session


# --- listing ----------------------------------------------------------------


def test_listing_follows_the_cursor_links() -> None:
    client, session = client_with(
        *pages(
            models("2026-10-12T10:00:00.000Z", "2026-10-11T09:00:00.000Z"),
            models("2026-10-10T08:00:00.000Z"),
        )
    )

    created = list(client.iter_created())

    assert [c.isoformat() for c in created] == [
        "2026-10-12T10:00:00+00:00",
        "2026-10-11T09:00:00+00:00",
        "2026-10-10T08:00:00+00:00",
    ]
    first, second = session.calls
    assert first["url"] == "https://huggingface.co/api/models"
    assert first["params"] == {
        "sort": "createdAt",
        "direction": "-1",
        "limit": "1000",
        "expand[]": "createdAt",
    }
    # The next link already carries the parameters and cursor.
    assert second["url"] == f"{NEXT}1"
    assert second["params"] == {}


def test_counts_by_utc_month_and_stops_once_past_the_earliest() -> None:
    client, session = client_with(
        *pages(
            models(
                "2026-10-03T00:00:00.000Z",  # October: not asked for
                "2026-09-30T23:59:59.000Z",
                "2026-09-01T00:00:00.000Z",
            ),
            models(
                "2026-08-31T23:00:00.000Z",
                "2026-08-15T00:00:00.000Z",
                "2026-08-02T00:00:00.000Z",
            ),
            models("2026-07-29T00:00:00.000Z"),  # far enough past August
            models("2026-07-01T00:00:00.000Z"),  # never fetched
        )
    )

    counts = list(client.count_months([AUG, SEP]))

    assert counts == [MonthlyCount(SEP, 2), MonthlyCount(AUG, 3)]
    assert len(session.calls) == 3


def test_a_month_stays_open_for_slightly_out_of_order_models() -> None:
    client, _ = client_with(
        *pages(
            models(
                "2026-09-02T00:00:00.000Z",
                "2026-08-31T00:00:00.000Z",  # August, within the slack
                "2026-09-01T12:00:00.000Z",  # September, listed late
                "2026-08-20T00:00:00.000Z",
            )
        )
    )

    assert list(client.count_months([SEP])) == [MonthlyCount(SEP, 2)]


def test_listing_that_runs_out_still_reports_every_month() -> None:
    client, _ = client_with(*pages(models("2026-09-05T00:00:00.000Z")))

    assert list(client.count_months([JUL, SEP])) == [
        MonthlyCount(SEP, 1),
        MonthlyCount(JUL, 0),
    ]


def test_months_before_createdAt_is_meaningful_are_refused() -> None:
    client, session = client_with()

    with pytest.raises(ValueError, match="before 2022-04-01"):
        list(client.count_months([date(2022, 3, 1), date(2022, 4, 1)]))
    assert session.calls == []


def test_missing_createdAt_raises() -> None:
    client, _ = client_with(*pages([{"id": "org/broken"}]))

    with pytest.raises(SourceAPIError, match="org/broken"):
        list(client.iter_created())


def test_invalid_json_raises() -> None:
    client, _ = client_with(FakeResponse(ValueError("not json")))

    with pytest.raises(SourceAPIError, match="Invalid JSON"):
        list(client.iter_created())


# --- rate limits and auth ---------------------------------------------------


def test_default_rate_limit_stays_under_hf_fixed_window() -> None:
    limiter = HuggingFaceClient().rate_limiter
    assert (limiter.max_calls, limiter.period) == (450, 300.0)


def test_429_waits_for_the_window_reset() -> None:
    sleeps: list[float] = []
    client, session = client_with(
        FakeResponse("rate limited", 429, headers={"ratelimit": '"api";r=0;t=187'}),
        *pages(models("2026-09-05T00:00:00.000Z")),
        sleeps=sleeps,
    )

    assert len(list(client.iter_created())) == 1
    assert sleeps == [187.0]
    assert len(session.calls) == 2


def test_token_is_sent_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    client, session = client_with(*pages([]))
    list(client.iter_created())
    assert session.calls[0]["headers"] == {}

    client, session = client_with(*pages([]), api_key="hf_abc")
    list(client.iter_created())
    assert session.calls[0]["headers"] == {"Authorization": "Bearer hf_abc"}

    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert HuggingFaceClient.from_env().api_key == "hf_env"
    monkeypatch.setenv("HF_TOKEN", "")
    assert HuggingFaceClient.from_env().api_key is None


# --- ingestion --------------------------------------------------------------


class FakeHF:
    """A MonthlyCounter that yields newest first, like the real client."""

    def __init__(self, counts: dict[date, int], fail_after: int | None = None) -> None:
        self.counts = counts
        self.fail_after = fail_after
        self.closed = False

    def count_months(self, months: Sequence[date]) -> Iterator[MonthlyCount]:
        for i, m in enumerate(sorted(months, reverse=True)):
            if self.fail_after is not None and i == self.fail_after:
                raise SourceAPIError("huggingface models failed after 6 attempts")
            yield MonthlyCount(m, self.counts[m])

    def close(self) -> None:
        self.closed = True


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
            WHERE s.series_id = 'HF_NEW_MODELS'
            ORDER BY o.observed_at, o.as_of
            """
        )
        return cur.fetchall()


def test_backfill_and_live_rows(migrated: Connection) -> None:
    result = ingest_backfill(
        migrated, SERIES, FakeHF({JUL: 120_000, AUG: 130_000}), JUL, AUG
    )
    assert (result.fetched, result.inserted) == (2, 2)

    today = date(2026, 10, 12)
    live = ingest_live(
        migrated, SERIES, FakeHF({JUL: 119_500, AUG: 130_000, SEP: 140_000}), today
    )
    assert (live.fetched, live.inserted) == (3, 2)

    assert stored(migrated) == [
        (JUL, AUG, 120_000, True),
        (JUL, today, 119_500, False),  # models deleted since
        (AUG, SEP, 130_000, True),
        (SEP, today, 140_000, False),
    ]
    with migrated.cursor() as cur:
        cur.execute(
            """
            SELECT s.domain, s.units, src.name
            FROM series s JOIN sources src ON src.id = s.source_id
            WHERE s.series_id = 'HF_NEW_MODELS'
            """
        )
        assert cur.fetchone() == ("ai", "Models", "Hugging Face Hub")


def test_interrupted_backfill_keeps_the_months_it_counted(
    migrated: Connection,
) -> None:
    with pytest.raises(SourceAPIError):
        ingest_backfill(
            migrated,
            SERIES,
            FakeHF({JUL: 1, AUG: 2, SEP: 3}, fail_after=1),
            JUL,
            SEP,
        )
    migrated.rollback()  # what `connection()` does on the error

    assert stored(migrated) == [(SEP, OCT, 3, True)]

    # The rerun counts only the missing months.
    result = ingest_backfill(migrated, SERIES, FakeHF({JUL: 1, AUG: 2}), JUL, SEP)
    assert result.fetched == 2
    assert [r[0] for r in stored(migrated)] == [JUL, AUG, SEP]


# --- command line -----------------------------------------------------------


def test_cli_refuses_a_start_before_april_2022(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        ingest_huggingface.main(
            ["--backfill", "--start", "2021-01-01", "--dry-run"],
            client_factory=lambda: FakeHF({}),
        )
    assert f"can't be before {FIRST_MONTH}" in capsys.readouterr().err


def test_cli_dry_run_prints_newest_first_and_closes_the_client(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeHF({JUL: 120_000, AUG: 130_000, SEP: 140_000})

    code = ingest_huggingface.main(
        ["--dry-run"], client_factory=lambda: fake, today=lambda: date(2026, 10, 12)
    )

    assert code == 0
    assert fake.closed
    assert capsys.readouterr().out.splitlines() == [
        "2026-09   140000  as_of=2026-10-12  live",
        "2026-08   130000  as_of=2026-10-12  live",
        "2026-07   120000  as_of=2026-10-12  live",
    ]
