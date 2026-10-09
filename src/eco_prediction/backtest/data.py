"""Point-in-time data access for backtests.

Usage:
    store = VintageStore.from_db()                    # loads vintages lazily, cached
    view = PointInTimeData(store, TARGETS["cpi_yoy"], cutoff=date(2020, 3, 31))
    cpi = view.series("CPIAUCSL")                     # as known on 2020-03-31
    yoy = view.target_history()                       # CPI YoY %, as known then

A `VintageStore` holds every vintage of each series in memory, so a backtest
reads the database once per series rather than once per forecast date.
`PointInTimeData` is the only way models see data: it returns, for each
period, the latest vintage published on or before its cutoff, and records the
latest `as_of` it served so the harness can prove nothing later leaked.

ALFRED's vintages of many series start years after the series does (2009-2014
for oil, gasoline, claims and the yield spreads), so a strict read before the
first vintage finds nothing. `VintageStore(..., backfill=True)` treats each
period in a series' first vintage as published `release_lag` days after the
period date, if that is earlier than the first vintage. The lag is the 90th
percentile delay, rounded up, of periods first published after the first
vintage. That is exact for unrevised series (market prices) and leaks only
later revisions for revised ones. Later vintages keep their real dates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

# Quantile of observed release delays used as a series' backfill lag.
RELEASE_LAG_QUANTILE = 0.9

from eco_prediction.db.connection import connection
from eco_prediction.db.queries import get_vintage_history


class LookaheadError(Exception):
    """Data published after the cutoff reached a model."""


def _level(series: pd.Series) -> pd.Series:
    return series.dropna()


def _yoy_percent(series: pd.Series) -> pd.Series:
    monthly = series.asfreq("MS")  # a gap must not shift the 12-month lag
    return (monthly.pct_change(12, fill_method=None) * 100).dropna()


@dataclass(frozen=True)
class Target:
    """A forecast target: a source series and how it becomes the quantity scored."""

    name: str  # matches the forecast_target enum
    series_id: str
    transform: Callable[[pd.Series], pd.Series]
    description: str


TARGETS: dict[str, Target] = {
    "cpi_yoy": Target(
        "cpi_yoy", "CPIAUCSL", _yoy_percent, "CPI-U all items (SA), YoY % change"
    ),
    "unemployment": Target(
        "unemployment", "UNRATE", _level, "Unemployment rate, % (SA)"
    ),
}


def _empty_known() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observed_at": pd.Series(dtype="datetime64[ns]"),
            "as_of": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
        }
    )


def release_lag(history: pd.DataFrame) -> int | None:
    """Days from period date to first release, for periods newer than the first vintage.

    Returns the RELEASE_LAG_QUANTILE of those delays, rounded up, or None if
    there are no newer periods. Older history added by a later vintage is not
    a release and is ignored.
    """
    if history.empty:
        return None
    first_vintage = history["as_of"].min()
    newest_in_first = history.loc[
        history["as_of"] == first_vintage, "observed_at"
    ].max()
    released = history[history["value"].notna()].groupby("observed_at")["as_of"].min()
    later = released[released.index > newest_in_first]
    if later.empty:
        return None
    delays = (later - later.index.to_series(index=later.index)).dt.days
    return math.ceil(delays.quantile(RELEASE_LAG_QUANTILE))


def backfill_first_vintage(history: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Date each first-vintage row `lag_days` after its period, if that is earlier."""
    if history.empty:
        return history
    first_vintage = history["as_of"].min()
    assumed = history["observed_at"] + pd.Timedelta(days=lag_days)
    earlier = (history["as_of"] == first_vintage) & (assumed < first_vintage)
    out = history.copy()
    out.loc[earlier, "as_of"] = assumed[earlier]
    return out.sort_values(["observed_at", "as_of"], ignore_index=True)


class VintageStore:
    """Every vintage of each series, loaded once and queried as of any date.

    Pass `histories` (series_id -> frame with observed_at, as_of, value, as
    returned by `get_vintage_history`) directly, or a `loader` that fetches one
    series on first use.

    `backfill=True` dates periods older than a series' first vintage by its
    release lag (see the module docstring). Lags are estimated per series, or
    taken from `release_lags` (days); the lags used are in `self.release_lags`.
    A series whose lag can't be estimated is left as it is.
    """

    def __init__(
        self,
        histories: Mapping[str, pd.DataFrame] | None = None,
        loader: Callable[[str], pd.DataFrame] | None = None,
        *,
        backfill: bool = False,
        release_lags: Mapping[str, int] | None = None,
    ) -> None:
        self._loader = loader
        self.backfill = backfill
        self.release_lags: dict[str, int] = dict(release_lags or {})
        self._histories: dict[str, pd.DataFrame] = {}
        for series_id, frame in (histories or {}).items():
            self._histories[series_id] = self._prepare(series_id, frame)

    @classmethod
    def from_db(
        cls, *, backfill: bool = False, release_lags: Mapping[str, int] | None = None
    ) -> VintageStore:
        """Load series from the pooled database connection as they are needed."""

        def load(series_id: str) -> pd.DataFrame:
            with connection() as conn:
                return get_vintage_history(conn, series_id)

        return cls(loader=load, backfill=backfill, release_lags=release_lags)

    def _prepare(self, series_id: str, frame: pd.DataFrame) -> pd.DataFrame:
        history = self._normalize(frame)
        if not self.backfill:
            return history
        if series_id not in self.release_lags:
            lag = release_lag(history)
            if lag is None:
                return history
            self.release_lags[series_id] = lag
        return backfill_first_vintage(history, self.release_lags[series_id])

    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "observed_at": pd.to_datetime(frame["observed_at"]),
                "as_of": pd.to_datetime(frame["as_of"]),
                "value": frame["value"].astype("float64"),
            }
        )
        return out.sort_values(["observed_at", "as_of"], ignore_index=True)

    def history(self, series_id: str) -> pd.DataFrame:
        """All vintages of a series, sorted by (observed_at, as_of). Empty if unknown."""
        if series_id not in self._histories:
            raw = self._loader(series_id) if self._loader else _empty_known()
            self._histories[series_id] = self._prepare(series_id, raw)
        return self._histories[series_id]

    def known(self, series_id: str, cutoff: date) -> pd.DataFrame:
        """Latest vintage of each period published on or before `cutoff`."""
        history = self.history(series_id)
        published = history[history["as_of"] <= pd.Timestamp(cutoff)]
        return published.drop_duplicates("observed_at", keep="last")

    def first_release(self, series_id: str, period: date) -> date | None:
        """When a non-missing value for `period` was first published, if ever."""
        history = self.history(series_id)
        rows = history[
            (history["observed_at"] == pd.Timestamp(period)) & history["value"].notna()
        ]
        return None if rows.empty else rows["as_of"].iloc[0].date()

    def until(self, cutoff: date) -> VintageStore:
        """A store holding only the vintages published on or before `cutoff`.

        Series load lazily, already backfilled, so nothing later can be read
        from it whatever date is asked for.
        """
        limit = pd.Timestamp(cutoff)

        def load(series_id: str) -> pd.DataFrame:
            history = self.history(series_id)
            return history[history["as_of"] <= limit]

        return VintageStore(loader=load)

    def latest_vintage(self, series_id: str) -> date | None:
        history = self.history(series_id)
        return None if history.empty else history["as_of"].max().date()


class PointInTimeData:
    """Read-only view of the data as known at the end of `cutoff`.

    `start` limits the periods returned (a rolling training window). Nothing
    published after `cutoff` is ever returned; `max_as_of_seen` records the
    latest vintage actually served.
    """

    def __init__(
        self,
        store: VintageStore,
        target: Target,
        cutoff: date,
        start: date | None = None,
    ) -> None:
        self.store = store
        self.target = target
        self.cutoff = cutoff
        self.start = start
        self.max_as_of_seen: date | None = None

    def series(self, series_id: str, start: date | None = None) -> pd.Series:
        """A series as known at the cutoff, indexed by period (DatetimeIndex)."""
        return self._clip(self._raw(series_id), start)

    def target_history(self, start: date | None = None) -> pd.Series:
        """The target quantity (e.g. CPI YoY %) as known at the cutoff."""
        # Transform the unclipped series, so a YoY at the window's start still
        # has its year-earlier value.
        raw = self._raw(self.target.series_id)
        return self._clip(self.target.transform(raw), start).rename(self.target.name)

    def _raw(self, series_id: str) -> pd.Series:
        known = self.store.known(series_id, self.cutoff)
        if not known.empty:
            newest = known["as_of"].max().date()
            if newest > self.cutoff:  # can't happen via VintageStore; cheap to check
                raise LookaheadError(
                    f"{series_id}: vintage {newest} is after cutoff {self.cutoff}"
                )
            if self.max_as_of_seen is None or newest > self.max_as_of_seen:
                self.max_as_of_seen = newest
        values = pd.Series(
            known["value"].to_numpy(),
            index=pd.DatetimeIndex(known["observed_at"], name="observed_at"),
            name=series_id,
            dtype="float64",
        )
        return values

    def _clip(self, series: pd.Series, start: date | None) -> pd.Series:
        bounds = [d for d in (self.start, start) if d is not None]
        if bounds:
            series = series[series.index >= pd.Timestamp(max(bounds))]
        return series
