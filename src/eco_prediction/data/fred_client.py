"""Client for the FRED / ALFRED web API (St. Louis Fed).

Usage:
    from eco_prediction.data.fred_client import FREDClient

    with FREDClient.from_env() as fred:
        info = fred.get_series_info("CPIAUCSL")
        obs = fred.get_series_observations("CPIAUCSL")  # every vintage

Observations come back as `Observation(observed_at, as_of, value)` rows that
map one-to-one onto the `observations` table: `as_of` is the date a value was
first published (ALFRED's `realtime_start`) and `value` is None where FRED
reports a missing value ('.').

By default `get_series_observations` requests the full real-time period, so
every revision is returned. Pass `realtime_start`/`realtime_end` to narrow it,
or `vintage_dates` to get the series as it stood on specific dates. Series
with more vintages than FRED allows in one request (e.g. the daily yield
spreads) are fetched in several windows transparently.

Requests are rate limited to 120 per minute per client (FRED's documented
limit) and retried with exponential backoff on 429, 5xx, timeouts and
connection errors. Other API errors raise `FREDAPIError` immediately.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Any, Self

import requests
from dotenv import load_dotenv

from eco_prediction.db.connection import ConfigError

BASE_URL = "https://api.stlouisfed.org/fred"

# ALFRED's sentinels for "all of real time": passing both returns every vintage.
REALTIME_START_MIN = "1776-07-04"
REALTIME_END_MAX = "9999-12-31"

# FRED rejects a real-time period spanning more vintages than this.
MAX_VINTAGES_PER_REQUEST = 2000

# FRED's maximum `limit` for observations and vintage dates respectively.
_PAGE_SIZE = 100_000
_VINTAGE_PAGE_SIZE = 10_000
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

DateLike = str | date


class FREDAPIError(Exception):
    """FRED rejected the request, or it still failed after all retries."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Observation:
    observed_at: date  # period the value describes
    as_of: date  # vintage date: when this value was published
    value: float | None  # None = FRED reported missing ('.')


@dataclass(frozen=True)
class SeriesInfo:
    series_id: str
    name: str
    units: str
    frequency: str
    seasonal_adjustment: str
    observation_start: date
    observation_end: date
    last_updated: str
    notes: str | None


class RateLimiter:
    """Sliding-window limiter: at most `max_calls` per `period` seconds.

    `clock` and `sleep` are injectable so tests don't have to wait.
    """

    def __init__(
        self,
        max_calls: int = 120,
        period: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a call is allowed, then record it."""
        with self._lock:
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                self._sleep(self.period - (now - self._calls[0]))


def fred_api_key() -> str:
    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise ConfigError("FRED_API_KEY is not set (see .env.example)")
    return key


class FREDClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        rate_limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session = session or requests.Session()
        self._sleep = sleep

    @classmethod
    def from_env(cls, **kwargs: Any) -> FREDClient:
        return cls(fred_api_key(), **kwargs)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def get_series_info(self, series_id: str) -> SeriesInfo:
        """Current metadata: title, units, frequency, seasonal adjustment."""
        data = self._get("series", {"series_id": series_id})
        try:
            s = data["seriess"][0]
        except (KeyError, IndexError):
            raise FREDAPIError(
                f"No series metadata returned for {series_id!r}"
            ) from None
        return SeriesInfo(
            series_id=s["id"],
            name=s["title"],
            units=s["units"],
            frequency=s["frequency"],
            seasonal_adjustment=s["seasonal_adjustment"],
            observation_start=date.fromisoformat(s["observation_start"]),
            observation_end=date.fromisoformat(s["observation_end"]),
            last_updated=s["last_updated"],
            notes=s.get("notes"),
        )

    def get_series_observations(
        self,
        series_id: str,
        realtime_start: DateLike | None = None,
        realtime_end: DateLike | None = None,
        *,
        vintage_dates: Iterable[DateLike] | None = None,
        observation_start: DateLike | None = None,
        observation_end: DateLike | None = None,
    ) -> list[Observation]:
        """Observations with vintage dates, sorted by (observed_at, as_of).

        With no real-time arguments, every vintage is fetched. `vintage_dates`
        can't be combined with `realtime_start`/`realtime_end`.

        In real-time mode the vintage dates are looked up first, and a period
        with more than MAX_VINTAGES_PER_REQUEST of them is fetched in windows
        that each start on a vintage date. FRED reports a value already in
        effect when a window opens as published on the window's first day, so
        those repeats are dropped and every row keeps its true publish date.
        Rows in the first window are still clipped to `realtime_start`.
        """
        params: dict[str, str] = {"series_id": series_id}
        if observation_start is not None:
            params["observation_start"] = _iso(observation_start)
        if observation_end is not None:
            params["observation_end"] = _iso(observation_end)

        if vintage_dates is not None:
            if realtime_start is not None or realtime_end is not None:
                raise ValueError(
                    "Pass either vintage_dates or realtime_start/realtime_end, not both"
                )
            params["vintage_dates"] = ",".join(_iso(d) for d in vintage_dates)
            windows = [params]
        else:
            start = _iso(realtime_start or REALTIME_START_MIN)
            end = _iso(realtime_end or REALTIME_END_MAX)
            vintages = self.get_vintage_dates(series_id, start, end)
            windows = [
                {**params, "realtime_start": ws, "realtime_end": we}
                for ws, we in _realtime_windows(start, end, vintages)
            ]

        rows: list[dict[str, Any]] = []
        for window in windows:
            rows.extend(self._get_paged("series/observations", window, "observations"))
        observations = [_parse_observation(r) for r in rows]
        observations.sort(key=lambda o: (o.observed_at, o.as_of))
        if len(windows) > 1:
            observations = drop_unchanged(observations)
        return observations

    def get_vintage_dates(
        self,
        series_id: str,
        realtime_start: DateLike | None = None,
        realtime_end: DateLike | None = None,
    ) -> list[date]:
        """Dates on which the series was published or revised, ascending.

        FRED answers 500 instead of an empty list when no vintage falls in the
        requested period, so the full list is always fetched and filtered here.
        """
        params = {
            "series_id": series_id,
            "realtime_start": REALTIME_START_MIN,
            "realtime_end": REALTIME_END_MAX,
        }
        dates = self._get_paged(
            "series/vintagedates", params, "vintage_dates", _VINTAGE_PAGE_SIZE
        )
        start = _iso(realtime_start or REALTIME_START_MIN)
        end = _iso(realtime_end or REALTIME_END_MAX)
        # ISO date strings compare in date order.
        return [date.fromisoformat(d) for d in sorted(dates) if start <= d <= end]

    def _get_paged(
        self,
        endpoint: str,
        params: dict[str, str],
        key: str,
        page_size: int = _PAGE_SIZE,
    ) -> list[Any]:
        """Collect `key` from every page of an offset-paginated endpoint."""
        items: list[Any] = []
        while True:
            page = self._get(
                endpoint,
                {
                    **params,
                    "limit": str(page_size),
                    "offset": str(len(items)),
                    "sort_order": "asc",
                },
            )
            batch = page.get(key, [])
            items.extend(batch)
            if not batch or len(items) >= int(page.get("count", 0)):
                return items

    def _get(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        """GET a FRED endpoint as JSON, with rate limiting and retries."""
        url = f"{self.base_url}/{endpoint}"
        query = {**params, "api_key": self.api_key, "file_type": "json"}
        last_error = ""
        retry_after: float | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(self._backoff(attempt, retry_after))
                retry_after = None
            self.rate_limiter.acquire()
            try:
                resp = self.session.get(url, params=query, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                # requests' messages embed the full URL, api_key included.
                last_error = self._redact(f"{type(exc).__name__}: {exc}")
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()  # type: ignore[no-any-return]
                except ValueError:
                    raise FREDAPIError(
                        f"Invalid JSON from FRED {endpoint}", 200
                    ) from None

            message = self._redact(_error_message(resp))
            if resp.status_code not in _RETRYABLE_STATUS:
                raise FREDAPIError(
                    f"FRED {endpoint} failed ({resp.status_code}): {message}",
                    resp.status_code,
                )
            last_error = f"{resp.status_code}: {message}"
            retry_after = _retry_after(resp)

        raise FREDAPIError(
            f"FRED {endpoint} failed after {self.max_retries + 1} attempts ({last_error})"
        )

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self.backoff_base * 2 ** (attempt - 1), self.backoff_max)
        return max(delay, retry_after or 0.0)

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text


def drop_unchanged(
    observations: Iterable[Observation],
    prior: Mapping[date, float | None] | None = None,
) -> list[Observation]:
    """Drop rows whose value repeats the previous vintage of the same period.

    `observations` must be sorted by (observed_at, as_of). `prior` seeds the
    previous value per period, e.g. the latest vintage already stored, so rows
    FRED re-reports at the start of a later real-time window are dropped too.
    A repeated row changes no point-in-time result.
    """
    last = dict(prior or {})
    kept = []
    for o in observations:
        if o.observed_at in last and last[o.observed_at] == o.value:
            continue
        last[o.observed_at] = o.value
        kept.append(o)
    return kept


def _realtime_windows(
    start: str, end: str, vintages: Sequence[date]
) -> list[tuple[str, str]]:
    """Split [start, end] so no window spans more than MAX_VINTAGES_PER_REQUEST vintages.

    Each window after the first starts on its first vintage date, so the rows
    FRED clips to that start are exact repeats that `drop_unchanged` removes.
    """
    if len(vintages) <= MAX_VINTAGES_PER_REQUEST:
        return [(start, end)]
    chunks = [
        vintages[i : i + MAX_VINTAGES_PER_REQUEST]
        for i in range(0, len(vintages), MAX_VINTAGES_PER_REQUEST)
    ]
    return [
        (
            start if i == 0 else chunk[0].isoformat(),
            end if i == len(chunks) - 1 else chunk[-1].isoformat(),
        )
        for i, chunk in enumerate(chunks)
    ]


def _iso(d: DateLike) -> str:
    return d.isoformat() if isinstance(d, date) else d


def _parse_observation(row: dict[str, Any]) -> Observation:
    raw = row["value"]
    return Observation(
        observed_at=date.fromisoformat(row["date"]),
        as_of=date.fromisoformat(row["realtime_start"]),
        value=None if raw == "." else float(raw),
    )


def _error_message(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("error_message", resp.text))
    except ValueError:
        return resp.text[:500]


def _retry_after(resp: requests.Response) -> float | None:
    try:
        return float(resp.headers["Retry-After"])
    except (KeyError, ValueError):
        return None
