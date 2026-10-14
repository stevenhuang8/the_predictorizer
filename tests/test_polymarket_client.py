"""Tests for the Polymarket client, the platform tables and their ingestion.

Client tests replay fake Gamma and CLOB responses offline; ingestion tests
use a throwaway database (skipped if Postgres isn't reachable).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import ingest_polymarket
from eco_prediction.ai.ingest_polymarket import history_end, ingest
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.sources.polymarket_client import (
    PAGE_SIZE,
    PolymarketClient,
    daily_prices,
    parse_market,
)
from eco_prediction.data.fred_client import RateLimiter
from eco_prediction.db.migrate import migrate

TODAY = date(2026, 10, 14)
EVENT = {"id": "25009", "slug": "openai-hardware", "title": "OpenAI hardware by...?"}


def market(id: str, **fields: Any) -> dict[str, Any]:
    """A Gamma market as the API sends it: list fields are JSON strings."""
    raw: dict[str, Any] = {
        "id": id,
        "question": f"Question {id}?",
        "description": f"Resolves yes if {id} happens.",
        "slug": f"market-{id}",
        "conditionId": f"0xcond{id}",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.25", "0.75"]',
        "clobTokenIds": json.dumps([f"yes{id}", f"no{id}"]),
        "volumeNum": 50_000,
        "startDate": "2026-10-10T12:00:00Z",
        "endDate": "2027-01-01T04:59:00Z",
        "closed": False,
        "umaResolutionStatus": None,
        "closedTime": None,
    }
    raw.update(fields)
    return raw


def resolved(id: str, prices: str, closed_time: str, **fields: Any) -> dict[str, Any]:
    return market(
        id,
        outcomePrices=prices,
        closed=True,
        umaResolutionStatus="resolved",
        closedTime=closed_time,
        **fields,
    )


def at(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp())


class FakeResponse:
    def __init__(self, body: Any, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = json.dumps(body)

    def json(self) -> Any:
        return self._body


class FakeSession:
    """Answers Gamma `events` pages and CLOB `prices-history` by token."""

    def __init__(
        self,
        open_events: list[dict[str, Any]],
        closed_events: list[dict[str, Any]] | None = None,
        history: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.events = {"false": open_events, "true": closed_events or []}
        self.history = history or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any], **kwargs: Any) -> FakeResponse:
        self.calls.append((url, params))
        if url.endswith("/events"):
            events = self.events[params["closed"]]
            start = params["offset"]
            return FakeResponse(events[start : start + params["limit"]])
        if url.endswith("/prices-history"):
            return FakeResponse({"history": self.history.get(params["market"], [])})
        return FakeResponse({"error": "not found"}, 404)

    def history_calls(self) -> list[str]:
        return [p["market"] for u, p in self.calls if u.endswith("/prices-history")]

    def close(self) -> None:
        pass


def client_with(session: FakeSession) -> PolymarketClient:
    return PolymarketClient(
        session=session,  # type: ignore[arg-type]
        rate_limiter=RateLimiter(1000, 1.0),
        sleep=lambda s: None,
    )


def event(*markets: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {**EVENT, **fields, "markets": list(markets)}


# --- parsing markets ----------------------------------------------------------


def test_open_market_has_its_yes_price_and_no_outcome() -> None:
    m = parse_market(market("1"), EVENT)

    assert m is not None
    assert (m.id, m.question, m.yes_token) == ("1", "Question 1?", "yes1")
    assert (m.closed, m.outcome, m.yes_price) == (False, None, 0.25)
    assert m.url == "https://polymarket.com/event/openai-hardware"
    assert m.opened_at == datetime(2026, 10, 10, 12, tzinfo=UTC)
    assert m.closes_at == datetime(2027, 1, 1, 4, 59, tzinfo=UTC)
    assert m.metadata["event_title"] == "OpenAI hardware by...?"
    assert m.metadata["condition_id"] == "0xcond1"


@pytest.mark.parametrize(
    ("prices", "outcome"),
    [('["1", "0"]', "yes"), ('["0", "1"]', "no"), ('["0.5", "0.5"]', None)],
)
def test_resolution_comes_from_the_final_prices(
    prices: str, outcome: str | None
) -> None:
    m = parse_market(resolved("2", prices, "2026-01-01 07:14:25+00"), EVENT)

    assert m is not None
    assert m.outcome == outcome
    assert m.closed_at == datetime(2026, 1, 1, 7, 14, 25, tzinfo=UTC)
    assert m.yes_price is None


def test_a_proposed_resolution_is_not_an_outcome_yet() -> None:
    raw = resolved("3", '["1", "0"]', "2026-01-01 07:14:25+00")
    raw["umaResolutionStatus"] = "proposed"

    m = parse_market(raw, EVENT)

    assert m is not None and m.closed and m.outcome is None


@pytest.mark.parametrize(
    "fields",
    [
        {"outcomes": '["Up", "Down"]'},
        {"outcomes": '["Yes", "No", "Maybe"]'},
        {"clobTokenIds": None, "outcomePrices": None},  # inactive placeholder
    ],
)
def test_non_binary_and_untradable_markets_are_skipped(fields: dict[str, Any]) -> None:
    assert parse_market(market("4", **fields), EVENT) is None


@pytest.mark.parametrize(
    ("fields", "drop"),
    [
        ({"outcomePrices": "not json"}, None),
        ({"outcomePrices": '"0.5"'}, None),
        ({}, "id"),
    ],
)
def test_malformed_markets_raise(fields: dict[str, Any], drop: str | None) -> None:
    raw = market("5", **fields)
    if drop:
        del raw[drop]

    with pytest.raises(SourceAPIError, match="bad field"):
        parse_market(raw, EVENT)


# --- listing and history ------------------------------------------------------


def test_events_are_paged_and_markets_deduped_and_filtered_by_volume() -> None:
    full_page = [event(market(str(i))) for i in range(PAGE_SIZE)]
    session = FakeSession(
        open_events=[*full_page, event(market("thin", volumeNum=9_999))],
        closed_events=[
            event(resolved("0", '["1", "0"]', "2026-01-01 00:00:00+00")),  # again
            event(market("old", closed=True), slug="old-event"),
        ],
    )

    markets = client_with(session).ai_markets(min_volume=10_000)

    assert [m.id for m in markets] == [*map(str, range(PAGE_SIZE)), "old"]
    assert [(p["closed"], p["offset"]) for _, p in session.calls] == [
        ("false", 0),
        ("false", 100),
        ("true", 0),
    ]
    assert all(p["tag_slug"] == "ai" for _, p in session.calls)


def test_price_history_is_parsed_and_sorted() -> None:
    session = FakeSession(
        [],
        history={"tok": [{"t": at(TODAY, 5), "p": 0.4}, {"t": at(TODAY), "p": 0.3}]},
    )

    points = client_with(session).price_history("tok")

    assert points == [
        (datetime(2026, 10, 14, tzinfo=UTC), 0.3),
        (datetime(2026, 10, 14, 5, tzinfo=UTC), 0.4),
    ]
    url, params = session.calls[0]
    assert url == "https://clob.polymarket.com/prices-history"
    assert params == {"market": "tok", "interval": "max", "fidelity": 1440}


def test_bad_history_raises() -> None:
    session = FakeSession([], history={"tok": [{"t": "soon", "p": 0.4}]})

    with pytest.raises(SourceAPIError, match="bad body"):
        client_with(session).price_history("tok")


def test_daily_prices_keep_each_utc_days_last_price_before_the_cutoff() -> None:
    d1, d2 = date(2026, 10, 12), date(2026, 10, 13)
    history = [
        (datetime(2026, 10, 12, 23, tzinfo=UTC), 0.31),
        (datetime(2026, 10, 12, 1, tzinfo=UTC), 0.30),
        (datetime(2026, 10, 13, 0, tzinfo=UTC), 0.35),
        (datetime(2026, 10, 14, 0, tzinfo=UTC), 0.40),  # today: incomplete
    ]

    assert daily_prices(history, TODAY) == {d1: 0.31, d2: 0.35}


def test_history_ends_yesterday_or_on_the_close_day() -> None:
    open_m = parse_market(market("1"), EVENT)
    closed_m = parse_market(
        resolved("2", '["0", "1"]', "2026-01-01 07:14:25+00"), EVENT
    )
    assert open_m is not None and closed_m is not None

    assert history_end(open_m, TODAY) == date(2026, 10, 13)
    assert history_end(closed_m, TODAY) == date(2026, 1, 1)


# --- ingestion ----------------------------------------------------------------


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def questions(conn: Connection) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT external_id, resolved_outcome, resolved_at, prices_through
            FROM platform_questions ORDER BY external_id
            """
        )
        return cur.fetchall()


def prices(conn: Connection, external_id: str) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.as_of, p.probability::float8, p.reconstructed
            FROM platform_probabilities p
            JOIN platform_questions q ON q.id = p.question_id
            WHERE q.external_id = %s ORDER BY p.as_of
            """,
            (external_id,),
        )
        return cur.fetchall()


D10, D11, D12, D13 = (date(2026, 10, d) for d in (10, 11, 12, 13))


def two_markets_session() -> FakeSession:
    return FakeSession(
        open_events=[event(market("open"))],
        closed_events=[
            event(resolved("done", '["1", "0"]', "2026-10-11 18:00:00+00")),
        ],
        history={
            "yesopen": [
                {"t": at(D10, 12), "p": 0.10},
                {"t": at(D12), "p": 0.20},
                {"t": at(TODAY), "p": 0.99},  # today: left to the live price
            ],
            "yesdone": [
                {"t": at(D10), "p": 0.50},
                {"t": at(D11), "p": 0.60},
                {"t": at(D12), "p": 1.00},  # after the close day
            ],
        },
    )


def test_first_run_stores_questions_history_and_live_prices(
    migrated: Connection,
) -> None:
    session = two_markets_session()

    result = ingest(migrated, client_with(session), TODAY)

    assert (result.markets, result.new_markets) == (2, 2)
    assert (result.histories, result.history_rows, result.live_rows) == (2, 4, 1)
    assert questions(migrated) == [
        ("done", "yes", datetime(2026, 10, 11, 18, tzinfo=UTC), D11),
        ("open", None, None, TODAY),
    ]
    assert prices(migrated, "open") == [
        (D10, 0.10, True),
        (D12, 0.20, True),
        (TODAY, 0.25, False),
    ]
    assert prices(migrated, "done") == [(D10, 0.50, True), (D11, 0.60, True)]


def test_daily_runs_only_fetch_history_for_gaps(migrated: Connection) -> None:
    ingest(migrated, client_with(two_markets_session()), TODAY)

    # The next day: the open market's live price covers today, so no history.
    tomorrow = TODAY + timedelta(days=1)
    session = two_markets_session()
    result = ingest(migrated, client_with(session), tomorrow)
    assert session.history_calls() == []
    assert result.live_rows == 1
    assert questions(migrated)[1][3] == tomorrow

    # A missed day leaves a gap, so the open market's history is fetched again.
    later = tomorrow + timedelta(days=2)
    session = two_markets_session()
    ingest(migrated, client_with(session), later)
    assert session.history_calls() == ["yesopen"]
    assert questions(migrated)[1][3] == later


def test_a_new_resolution_updates_the_question(migrated: Connection) -> None:
    pk, _, new = upsert_platform_question(
        migrated, PlatformQuestion("polymarket", "x", "Will X?")
    )
    assert new
    resolved_at = datetime(2026, 10, 1, tzinfo=UTC)
    pk2, _, new2 = upsert_platform_question(
        migrated,
        PlatformQuestion(
            "polymarket", "x", "Will X?", resolved_at=resolved_at, resolved_outcome="no"
        ),
    )
    assert (pk2, new2) == (pk, False)
    assert questions(migrated) == [("x", "no", resolved_at, None)]


@pytest.mark.parametrize(
    "cols",
    [
        {"platform": "kalshi"},
        {"resolved_outcome": "yes"},  # an outcome needs a resolution time
        {"resolved_outcome": "maybe", "resolved_at": datetime(2026, 10, 1, tzinfo=UTC)},
    ],
)
def test_platform_question_constraints(
    migrated: Connection, cols: dict[str, Any]
) -> None:
    cols = {"platform": "polymarket", "external_id": "1", "title": "q", **cols}
    names, marks = ", ".join(cols), ", ".join(["%s"] * len(cols))
    with pytest.raises(psycopg2.IntegrityError), migrated.cursor() as cur:
        cur.execute(
            f"INSERT INTO platform_questions ({names}) VALUES ({marks})",
            tuple(cols.values()),
        )


def test_probabilities_must_be_between_0_and_1(migrated: Connection) -> None:
    pk, _, _ = upsert_platform_question(
        migrated, PlatformQuestion("polymarket", "x", "Will X?")
    )
    with pytest.raises(psycopg2.IntegrityError), migrated.cursor() as cur:
        cur.execute(
            "INSERT INTO platform_probabilities VALUES (%s, %s, 1.5, false)",
            (pk, TODAY),
        )


# --- command line -------------------------------------------------------------


def test_cli_dry_run_lists_markets_without_history(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = two_markets_session()

    code = ingest_polymarket.main(
        ["--dry-run"],
        client_factory=lambda: client_with(session),
        today=lambda: TODAY,
    )

    assert code == 0
    assert session.history_calls() == []
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "2 markets"
    assert "open 0.250" in out[0] and "Question open?" in out[0]
    assert "resolved yes" in out[1]


def test_cli_rejects_a_negative_min_volume() -> None:
    with pytest.raises(SystemExit):
        ingest_polymarket.main(["--min-volume", "-1", "--dry-run"])


def test_cli_reports_api_errors(capsys: pytest.CaptureFixture[str]) -> None:
    session = FakeSession([event(market("bad", outcomePrices="nope"))])

    code = ingest_polymarket.main(
        ["--dry-run"], client_factory=lambda: client_with(session)
    )

    assert code == 1
    assert "bad field" in capsys.readouterr().err
