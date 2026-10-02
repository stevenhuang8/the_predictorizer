"""Tests for the FRED/ALFRED client.

Unit tests use a fake session and fake clock, so they run offline and instantly.
Integration tests hit the live API and are skipped unless FRED_API_KEY is set.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import pytest
import requests
from dotenv import load_dotenv

from eco_prediction.data import fred_client
from eco_prediction.data.fred_client import (
    REALTIME_END_MAX,
    REALTIME_START_MIN,
    FREDAPIError,
    FREDClient,
    Observation,
    RateLimiter,
    drop_unchanged,
)

API_KEY = "a" * 32


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    """Replays queued responses (or raises queued exceptions) and records calls."""

    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, str], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        pass


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(session: FakeSession, **kwargs: Any) -> tuple[FREDClient, list[float]]:
    sleeps: list[float] = []
    client = FREDClient(
        API_KEY,
        session=session,  # type: ignore[arg-type]
        rate_limiter=RateLimiter(max_calls=1000, period=60),
        sleep=sleeps.append,
        **kwargs,
    )
    return client, sleeps


def obs_row(day: str, realtime_start: str, value: str) -> dict[str, str]:
    return {
        "date": day,
        "realtime_start": realtime_start,
        "realtime_end": "9999-12-31",
        "value": value,
    }


def vintages_response(*dates: str) -> FakeResponse:
    return FakeResponse(body={"count": len(dates), "vintage_dates": list(dates)})


# --- observations -----------------------------------------------------------


def test_observations_preserve_vintages_and_missing_values() -> None:
    session = FakeSession(
        vintages_response("2024-02-02", "2024-03-08"),
        FakeResponse(
            body={
                "count": 3,
                "observations": [
                    obs_row("2024-01-01", "2024-03-08", "3.8"),
                    obs_row("2024-01-01", "2024-02-02", "3.7"),
                    obs_row("2024-02-01", "2024-03-08", "."),
                ],
            }
        ),
    )
    client, _ = make_client(session)

    obs = client.get_series_observations("UNRATE")

    assert obs == [
        Observation(date(2024, 1, 1), date(2024, 2, 2), 3.7),
        Observation(date(2024, 1, 1), date(2024, 3, 8), 3.8),
        Observation(date(2024, 2, 1), date(2024, 3, 8), None),
    ]
    assert session.calls[0]["url"].endswith("/series/vintagedates")
    params = session.calls[1]["params"]
    assert params["realtime_start"] == REALTIME_START_MIN
    assert params["realtime_end"] == REALTIME_END_MAX
    assert params["api_key"] == API_KEY
    assert params["file_type"] == "json"


def test_observations_paginate_until_count_reached() -> None:
    session = FakeSession(
        vintages_response("2024-02-01", "2024-03-01"),
        FakeResponse(
            body={
                "count": 3,
                "observations": [obs_row("2024-01-01", "2024-02-01", "1")] * 2,
            }
        ),
        FakeResponse(
            body={
                "count": 3,
                "observations": [obs_row("2024-02-01", "2024-03-01", "2")],
            }
        ),
    )
    client, _ = make_client(session)

    assert len(client.get_series_observations("X")) == 3
    assert [c["params"]["offset"] for c in session.calls[1:]] == ["0", "2"]


def test_observations_with_vintage_dates_and_date_objects() -> None:
    session = FakeSession(FakeResponse(body={"count": 0, "observations": []}))
    client, _ = make_client(session)

    client.get_series_observations(
        "X",
        vintage_dates=[date(2020, 1, 1), "2021-01-01"],
        observation_start=date(2019, 1, 1),
    )

    params = session.calls[0]["params"]
    assert params["vintage_dates"] == "2020-01-01,2021-01-01"
    assert params["observation_start"] == "2019-01-01"
    assert "realtime_start" not in params


def test_vintage_dates_and_realtime_are_mutually_exclusive() -> None:
    client, _ = make_client(FakeSession())
    with pytest.raises(ValueError):
        client.get_series_observations(
            "X", realtime_start="2020-01-01", vintage_dates=["2020-01-01"]
        )


def test_series_info_parses_metadata() -> None:
    session = FakeSession(
        FakeResponse(
            body={
                "seriess": [
                    {
                        "id": "CPIAUCSL",
                        "title": "Consumer Price Index",
                        "units": "Index 1982-1984=100",
                        "frequency": "Monthly",
                        "seasonal_adjustment": "Seasonally Adjusted",
                        "observation_start": "1947-01-01",
                        "observation_end": "2026-08-01",
                        "last_updated": "2026-09-11 07:45:02-05",
                    }
                ]
            }
        )
    )
    client, _ = make_client(session)

    info = client.get_series_info("CPIAUCSL")

    assert info.series_id == "CPIAUCSL"
    assert info.frequency == "Monthly"
    assert info.observation_start == date(1947, 1, 1)
    assert info.notes is None


def test_vintage_dates_paginate() -> None:
    session = FakeSession(
        FakeResponse(body={"count": 3, "vintage_dates": ["2024-01-02", "2024-01-03"]}),
        FakeResponse(body={"count": 3, "vintage_dates": ["2024-01-04"]}),
    )
    client, _ = make_client(session)

    dates = client.get_vintage_dates("X")

    assert dates == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    assert [c["params"]["offset"] for c in session.calls] == ["0", "2"]


def test_vintage_dates_filtered_locally() -> None:
    # FRED returns 500 for a period with no vintages, so the range is never sent.
    session = FakeSession(vintages_response("2024-01-02", "2024-01-03", "2024-01-04"))
    client, _ = make_client(session)

    dates = client.get_vintage_dates("X", date(2024, 1, 3), "2024-01-03")
    assert dates == [date(2024, 1, 3)]
    assert session.calls[0]["params"]["realtime_start"] == REALTIME_START_MIN

    session.responses.append(vintages_response("2024-01-02"))
    assert client.get_vintage_dates("X", realtime_start=date(2030, 1, 1)) == []


def test_many_vintages_fetched_in_windows_without_clipped_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fred_client, "MAX_VINTAGES_PER_REQUEST", 2)
    d1, d2, d3, d4 = "2024-01-10", "2024-02-10", "2024-03-10", "2024-04-10"
    p1, p2, p3 = "2024-01-01", "2024-02-01", "2024-03-01"
    session = FakeSession(
        vintages_response(d1, d2, d3, d4),
        FakeResponse(
            body={
                "count": 3,
                "observations": [
                    obs_row(p1, d1, "1.0"),
                    obs_row(p1, d2, "1.1"),
                    obs_row(p2, d2, "2.0"),
                ],
            }
        ),
        # FRED dates values already in effect to the window's start (d3).
        FakeResponse(
            body={
                "count": 4,
                "observations": [
                    obs_row(p1, d3, "1.1"),
                    obs_row(p2, d3, "2.0"),
                    obs_row(p2, d4, "2.1"),
                    obs_row(p3, d4, "3.0"),
                ],
            }
        ),
    )
    client, _ = make_client(session)

    obs = client.get_series_observations("X")

    windows = [
        (c["params"]["realtime_start"], c["params"]["realtime_end"])
        for c in session.calls[1:]
    ]
    assert windows == [(REALTIME_START_MIN, d2), (d3, REALTIME_END_MAX)]
    assert [(o.observed_at.isoformat(), o.as_of.isoformat(), o.value) for o in obs] == [
        (p1, d1, 1.0),
        (p1, d2, 1.1),
        (p2, d2, 2.0),
        (p2, d4, 2.1),
        (p3, d4, 3.0),
    ]


def test_drop_unchanged_against_prior_values() -> None:
    jan, feb = date(2024, 1, 1), date(2024, 2, 1)
    as_of = date(2024, 3, 1)
    obs = [
        Observation(jan, as_of, 1.0),  # same as stored: dropped
        Observation(feb, as_of, None),  # stored as missing too: dropped
        Observation(feb, date(2024, 4, 1), 2.0),  # revision: kept
    ]

    assert drop_unchanged(obs, {jan: 1.0, feb: None}) == [obs[2]]


# --- errors and retries -----------------------------------------------------


def test_client_error_raises_immediately_without_retry() -> None:
    session = FakeSession(
        FakeResponse(
            400,
            {
                "error_code": 400,
                "error_message": "Bad Request. The series does not exist.",
            },
        )
    )
    client, sleeps = make_client(session)

    with pytest.raises(FREDAPIError, match="does not exist") as exc_info:
        client.get_series_info("NOPE")

    assert exc_info.value.status_code == 400
    assert len(session.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "failure",
    [FakeResponse(503, {}), FakeResponse(429, {}), requests.ConnectionError("boom")],
)
def test_transient_failures_are_retried(failure: FakeResponse | Exception) -> None:
    session = FakeSession(failure, failure, vintages_response("2024-01-05"))
    client, sleeps = make_client(session, backoff_base=1.0)

    assert client.get_vintage_dates("X") == [date(2024, 1, 5)]
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_backoff_is_capped_and_honors_retry_after() -> None:
    session = FakeSession(
        FakeResponse(429, {}, headers={"Retry-After": "10"}),
        FakeResponse(503, {}),
        FakeResponse(503, {}),
        vintages_response(),
    )
    client, sleeps = make_client(session, backoff_base=2.0, backoff_max=5.0)

    client.get_vintage_dates("X")

    assert sleeps == [10.0, 4.0, 5.0]


def test_gives_up_after_max_retries_and_redacts_api_key() -> None:
    err = requests.ConnectionError(
        f"Max retries exceeded with url: /fred/series?api_key={API_KEY}"
    )
    session = FakeSession(*[err] * 3)
    client, _ = make_client(session, max_retries=2)

    with pytest.raises(FREDAPIError, match="after 3 attempts") as exc_info:
        client.get_series_info("X")

    assert API_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


# --- rate limiting ----------------------------------------------------------


def test_rate_limiter_blocks_once_window_is_full() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_calls=3, period=60, clock=clock, sleep=clock.sleep)

    for _ in range(3):
        limiter.acquire()
    assert clock.sleeps == []

    limiter.acquire()  # 4th call must wait for the first to age out
    assert clock.sleeps == [60.0]


def test_rate_limiter_sliding_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_calls=2, period=60, clock=clock, sleep=clock.sleep)

    limiter.acquire()  # t=0
    clock.now = 45
    limiter.acquire()  # t=45
    limiter.acquire()  # waits until t=60, when the t=0 call expires
    assert clock.sleeps == [15.0]
    assert clock.now == 60


def test_rapid_client_requests_are_throttled() -> None:
    clock = FakeClock()
    session = FakeSession(*[vintages_response() for _ in range(125)])
    client = FREDClient(
        API_KEY,
        session=session,  # type: ignore[arg-type]
        # Default limits: FRED's 120 requests/minute.
        rate_limiter=RateLimiter(clock=clock, sleep=clock.sleep),
    )

    for _ in range(125):
        client.get_vintage_dates("X")

    assert len(session.calls) == 125
    assert clock.now >= 60  # requests 121-125 had to wait for the window to roll


# --- live API ---------------------------------------------------------------

load_dotenv()
live = pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY not set"
)


@live
def test_live_cpi_series_info() -> None:
    with FREDClient.from_env() as fred:
        info = fred.get_series_info("CPIAUCSL")
    assert info.series_id == "CPIAUCSL"
    assert info.frequency.startswith("Monthly")


@live
def test_live_cpi_vintages_are_preserved() -> None:
    with FREDClient.from_env() as fred:
        obs = fred.get_series_observations(
            "CPIAUCSL", observation_start="2020-01-01", observation_end="2020-12-01"
        )
    jan = [o for o in obs if o.observed_at == date(2020, 1, 1)]
    # CPI is revised (seasonal factors are re-estimated every year), so January 2020 has several vintages.
    assert len(jan) > 1
    assert jan[0].as_of >= date(2020, 2, 1)  # first published after the reference month
    assert jan == sorted(jan, key=lambda o: o.as_of)


@live
def test_live_unknown_series_raises() -> None:
    with FREDClient.from_env() as fred, pytest.raises(FREDAPIError) as exc_info:
        fred.get_series_info("THIS_SERIES_DOES_NOT_EXIST_XYZ")
    assert exc_info.value.status_code == 400
