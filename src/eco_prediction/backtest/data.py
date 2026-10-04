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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

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


class VintageStore:
    """Every vintage of each series, loaded once and queried as of any date.

    Pass `histories` (series_id -> frame with observed_at, as_of, value, as
    returned by `get_vintage_history`) directly, or a `loader` that fetches one
    series on first use.
    """

    def __init__(
        self,
        histories: Mapping[str, pd.DataFrame] | None = None,
        loader: Callable[[str], pd.DataFrame] | None = None,
    ) -> None:
        self._loader = loader
        self._histories: dict[str, pd.DataFrame] = {}
        for series_id, frame in (histories or {}).items():
            self._histories[series_id] = self._normalize(frame)

    @classmethod
    def from_db(cls) -> VintageStore:
        """Load series from the pooled database connection as they are needed."""

        def load(series_id: str) -> pd.DataFrame:
            with connection() as conn:
                return get_vintage_history(conn, series_id)

        return cls(loader=load)

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
            self._histories[series_id] = self._normalize(raw)
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
