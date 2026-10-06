"""Feature vectors built from the economic series as known on a date.

Usage:
    from eco_prediction.features.engineering import FeatureEngineer, drop_sparse

    fe = FeatureEngineer()
    store = VintageStore.from_db()
    fe.build_features_as_of(store, date(2020, 3, 31))   # {'t10y2y': 0.47, ...}
    X = fe.build_matrix(store, [date(2019, 12, 31), date(2020, 1, 31), ...])
    X, dropped = drop_sparse(X, max_missing=0.5)

Every feature comes from data published on or before its as-of date. Each row
of `build_matrix` is built from the vintages known on that row's date, so
training rows see the same unrevised, ragged-edge data a live forecast would.
Inside a backtest, pass the model's `PointInTimeData` view as the source: a
date after its cutoff raises `LookaheadError`.

Features describe the latest period available, not a fixed calendar month.
At the end of March, CPI is known through February, so `cpi_yoy` is
February's year-over-year change and `cpi_yoy_lag1` January's. With harness
cutoffs (month ends) each series sits the same distance behind every row.

Daily and weekly series are averaged to calendar months, counting only months
that have ended by the as-of date; the yield spreads and jobless claims also
keep their latest daily or weekly value. A monthly gap of up to `ffill_limit`
months is forward filled. A series whose latest value is older than
`max_stale_months` (or `max_stale_days` for a daily or weekly value) yields
None for its features rather than an out-of-date number.

Values are floats, or None when they can't be computed (short history, stale
or missing series), so a feature dict can be stored as JSON as it is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any
from weakref import WeakKeyDictionary

import pandas as pd

from eco_prediction.backtest.data import LookaheadError, PointInTimeData, VintageStore

Features = dict[str, float | None]

TREND_ORIGIN = date(1990, 1, 1)

# Series the features are built from, by sampling frequency.
MONTHLY_SERIES = (
    "CPIAUCSL",
    "UNRATE",
    "PPIACO",
    "CES0500000003",
    "MICH",
    "CUSR0000SEHA",
)
WEEKLY_SERIES = ("GASREGW", "ICSA")
DAILY_SERIES = ("T10Y2Y", "T10Y3M", "DCOILWTICO")
FEATURE_SERIES = MONTHLY_SERIES + WEEKLY_SERIES + DAILY_SERIES

TARGET_LAGS = 3  # lags kept for the target variables, CPI YoY and unemployment


def _months_between(start: pd.Timestamp | date, end: pd.Timestamp | date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def _value(x: Any) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def _published(series: pd.Series | None, as_of: date) -> pd.Series:
    """The series up to as_of, sorted. Periods after as_of are dropped."""
    if series is None or series.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    series = series.astype("float64").sort_index()
    return series[series.index <= pd.Timestamp(as_of)]


def _empty_monthly() -> pd.Series:
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([], freq="MS"))


def to_monthly(
    series: pd.Series | None, as_of: date, ffill_limit: int = 1
) -> pd.Series:
    """A monthly series on a gap-free month-start index, short gaps forward filled."""
    s = _published(series, as_of)
    if s.empty:
        return _empty_monthly()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    s = s[~s.index.duplicated(keep="last")]
    return s.asfreq("MS").ffill(limit=ffill_limit)


def monthly_mean(
    series: pd.Series | None, as_of: date, ffill_limit: int = 1
) -> pd.Series:
    """Calendar-month means of a daily or weekly series, months ended by as_of only."""
    s = _published(series, as_of).dropna()
    if s.empty:
        return _empty_monthly()
    months = pd.DatetimeIndex(s.index).to_period("M")
    means = s.groupby(months).mean()
    periods = pd.PeriodIndex(means.index)
    ended = periods.end_time.normalize() <= pd.Timestamp(as_of)
    means = means[ended]
    if means.empty:
        return _empty_monthly()
    means.index = periods[ended].to_timestamp()
    return means.asfreq("MS").ffill(limit=ffill_limit)


def pct_change(monthly: pd.Series, months: int) -> pd.Series:
    """Percent change over `months` on a gap-free monthly series."""
    return (monthly / monthly.shift(months) - 1) * 100


def _store(source: VintageStore | PointInTimeData, as_of: date) -> VintageStore:
    """The store behind a source; a view refuses dates after its cutoff."""
    if isinstance(source, PointInTimeData):
        if as_of > source.cutoff:
            raise LookaheadError(
                f"features as of {as_of} requested from a view with cutoff {source.cutoff}"
            )
        return source.store
    return source


class FeatureEngineer:
    """Builds one feature dict per as-of date. The feature names are fixed."""

    def __init__(
        self,
        *,
        ffill_limit: int = 1,
        max_stale_months: int = 3,
        max_stale_days: int = 14,
    ) -> None:
        self.ffill_limit = ffill_limit
        self.max_stale_months = max_stale_months
        self.max_stale_days = max_stale_days
        # A row depends only on its date, so it's built once per store.
        self._rows: WeakKeyDictionary[VintageStore, dict[date, Features]] = (
            WeakKeyDictionary()
        )

    def build_features(
        self, series_data: Mapping[str, pd.Series], as_of_date: date
    ) -> Features:
        """Feature vector from raw series as known on as_of_date.

        `series_data` maps series codes to values indexed by period. Missing
        series give None features; every feature name is always present.
        """
        get = series_data.get
        features: Features = {}

        # Yield curve: latest daily spread.
        features["t10y2y"] = self._last_value(get("T10Y2Y"), as_of_date)
        features["t10y3m"] = self._last_value(get("T10Y3M"), as_of_date)

        # Prices: changes in monthly averages (oil, gasoline) or levels (PPI).
        for name, monthly in (
            ("oil", self._mean(get("DCOILWTICO"), as_of_date)),
            ("gas", self._mean(get("GASREGW"), as_of_date)),
            ("ppi", self._monthly(get("PPIACO"), as_of_date)),
        ):
            latest = self._latest(monthly, as_of_date)
            for suffix, months in (("mom", 1), ("3m", 3), ("yoy", 12)):
                change = pct_change(monthly, months)
                features[f"{name}_{suffix}"] = self._at(change, latest)

        # Inflation (a target): CPI YoY with lags, and the latest monthly change.
        cpi = self._monthly(get("CPIAUCSL"), as_of_date)
        latest = self._latest(cpi, as_of_date)
        cpi_yoy = pct_change(cpi, 12)
        features["cpi_yoy"] = self._at(cpi_yoy, latest)
        for lag in range(1, TARGET_LAGS + 1):
            features[f"cpi_yoy_lag{lag}"] = self._at(cpi_yoy, latest, lag)
        features["cpi_mom"] = self._at(pct_change(cpi, 1), latest)

        # Unemployment (a target): level, change, 3-month average, lags.
        unrate = self._monthly(get("UNRATE"), as_of_date)
        latest = self._latest(unrate, as_of_date)
        features["unrate"] = self._at(unrate, latest)
        features["unrate_diff1"] = self._at(unrate.diff(), latest)
        features["unrate_ma3"] = self._at(unrate.rolling(3).mean(), latest)
        for lag in range(1, TARGET_LAGS + 1):
            features[f"unrate_lag{lag}"] = self._at(unrate, latest, lag)

        features.update(self._claims(get("ICSA"), as_of_date))

        # Other V1 series: wage growth, rent inflation, inflation expectations.
        for name, series_id in (
            ("wages_yoy", "CES0500000003"),
            ("rent_yoy", "CUSR0000SEHA"),
        ):
            monthly = self._monthly(get(series_id), as_of_date)
            features[name] = self._at(
                pct_change(monthly, 12), self._latest(monthly, as_of_date)
            )
        mich = self._monthly(get("MICH"), as_of_date)
        features["mich"] = self._at(mich, self._latest(mich, as_of_date))

        # Calendar: month of the as-of date, and a linear trend.
        for month in range(1, 13):
            features[f"month_{month}"] = float(as_of_date.month == month)
        features["trend_months"] = float(_months_between(TREND_ORIGIN, as_of_date))

        return features

    def build_features_as_of(
        self, source: VintageStore | PointInTimeData, as_of_date: date
    ) -> Features:
        """Feature vector from the vintages known on as_of_date. Cached per store."""
        rows = self._rows.setdefault(_store(source, as_of_date), {})
        if as_of_date not in rows:
            series = self.series_as_of(source, as_of_date)
            rows[as_of_date] = self.build_features(series, as_of_date)
        return dict(rows[as_of_date])

    def build_matrix(
        self, source: VintageStore | PointInTimeData, as_of_dates: Iterable[date]
    ) -> pd.DataFrame:
        """One row per as-of date (DatetimeIndex `as_of`), each from data known then."""
        dates = sorted(set(as_of_dates))
        rows = [self.build_features_as_of(source, d) for d in dates]
        index = pd.DatetimeIndex(dates, name="as_of")
        if not rows:
            return pd.DataFrame(index=index, dtype="float64")
        return pd.DataFrame(rows, index=index, dtype="float64")

    @staticmethod
    def series_as_of(
        source: VintageStore | PointInTimeData, as_of_date: date
    ) -> dict[str, pd.Series]:
        """Every feature series as known on as_of_date, indexed by period."""
        store = _store(source, as_of_date)
        out = {}
        for series_id in FEATURE_SERIES:
            known = store.known(series_id, as_of_date)
            out[series_id] = pd.Series(
                known["value"].to_numpy(),
                index=pd.DatetimeIndex(known["observed_at"], name="observed_at"),
                name=series_id,
                dtype="float64",
            )
        return out

    def _monthly(self, series: pd.Series | None, as_of: date) -> pd.Series:
        return to_monthly(series, as_of, self.ffill_limit)

    def _mean(self, series: pd.Series | None, as_of: date) -> pd.Series:
        return monthly_mean(series, as_of, self.ffill_limit)

    def _latest(self, monthly: pd.Series, as_of: date) -> pd.Timestamp | None:
        """Latest month with a value, unless it's more than max_stale_months old."""
        valid = monthly.dropna()
        if valid.empty:
            return None
        last = valid.index[-1]
        if _months_between(last, as_of) > self.max_stale_months:
            return None
        return last

    @staticmethod
    def _at(
        series: pd.Series, latest: pd.Timestamp | None, lag: int = 0
    ) -> float | None:
        """Value `lag` months before the latest month, if there is one."""
        if latest is None:
            return None
        return _value(series.get(latest - pd.DateOffset(months=lag)))

    def _last_value(self, series: pd.Series | None, as_of: date) -> float | None:
        s = _published(series, as_of).dropna()
        if s.empty or (as_of - s.index[-1].date()).days > self.max_stale_days:
            return None
        return float(s.iloc[-1])

    def _claims(self, series: pd.Series | None, as_of: date) -> Features:
        """Initial claims: latest week, 4-week average, and its YoY % change."""
        out: Features = {"claims": None, "claims_4wk": None, "claims_4wk_yoy": None}
        s = _published(series, as_of).dropna()
        if s.empty or (as_of - s.index[-1].date()).days > self.max_stale_days:
            return out
        out["claims"] = float(s.iloc[-1])
        recent = self._four_weeks(s, s.index[-1])
        if recent is None:
            return out
        out["claims_4wk"] = recent
        year_ago = self._four_weeks(s, s.index[-1] - pd.Timedelta(weeks=52))
        if year_ago is not None:
            out["claims_4wk_yoy"] = (recent / year_ago - 1) * 100
        return out

    @staticmethod
    def _four_weeks(weekly: pd.Series, end: pd.Timestamp) -> float | None:
        """Mean of the four weeks ending at `end`, if all four are present."""
        window = weekly[
            (weekly.index > end - pd.Timedelta(weeks=4)) & (weekly.index <= end)
        ]
        return float(window.mean()) if len(window) == 4 else None


def drop_sparse(
    matrix: pd.DataFrame, max_missing: float = 0.5
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns missing in more than `max_missing` of rows. Returns (kept, dropped)."""
    if not 0 <= max_missing <= 1:
        raise ValueError("max_missing must be between 0 and 1")
    missing = matrix.isna().mean()
    dropped = missing.index[missing > max_missing].tolist()
    return matrix.drop(columns=dropped), dropped
