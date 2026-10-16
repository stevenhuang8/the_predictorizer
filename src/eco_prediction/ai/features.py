"""Feature vectors built from the AI series as known on a date (AI task 12).

Usage:
    from eco_prediction.ai.features import AIFeatureEngineer, build_ai_features

    build_ai_features(conn, date(2026, 10, 15))                 # {'arxiv': 30120.0, ...}
    build_ai_features(conn, date(2026, 10, 15), "HF_NEW_MODELS")  # + target_* aliases

    fe = AIFeatureEngineer()
    store = VintageStore.from_db()
    X = fe.build_matrix(store, [date(2025, 1, 31), date(2025, 2, 28), ...])
    save_feature_snapshot(conn, as_of, features)   # db.forecasts, as for the economy

Mirrors the economy's `features/engineering.py`: every feature comes from
data published on or before the as-of date (pass a `PointInTimeData` view
inside a backtest and a later date raises `LookaheadError`), features describe
each series' latest available month, and lags count back from that month.
A series whose latest month is more than `max_stale_months` before the as-of
month, or that hasn't started yet (SOTA series begin in 2023-2025), gives
None for its features. Every feature name is always present.

Per series (prefix in brackets):
- **Counts** (`arxiv`, `hf`, `github`, `stars`): the level, its natural log,
  lags 1/3/6/12, % growth over 1, 3 and 12 months, and `<p>_months_behind`
  (latest month to as-of month).
- **Step series**, monthly running maxima (`gpqa`, `swebench`, `arc2` in
  percent; `flops` in log10 FLOP): the level, lags 1/3/6/12, changes over 1,
  3 and 12 months (points, or log10 units), `<p>_months_since_change`,
  `<p>_last_jump` (the size of the latest increase), `<p>_months_behind`,
  and for percentages `<p>_headroom` (100 - level).
- **Cross-series**: correlation of HF's and of GitHub's monthly log changes
  with arXiv's over the last 12 months (`corr12_hf_arxiv`,
  `corr12_github_arxiv`), their log ratios to arXiv, and
  `sota_increases_12m`, how many monthly increases the three benchmarks had
  in the last 12 months together.
- **Calendar**: `month_1`..`month_12`, `quarter_1`..`quarter_4` of the as-of
  date, and `trend_months` since 2010-01.

With a `target_series`, that series' features are repeated under a `target`
prefix (`target`, `target_log`, `target_lag1`, ...), so one model can be
trained across several series of a kind with the same column names.

Reconstructed rows (`observations.reconstructed`, backfilled from timestamped
records) are read like any vintage at their `as_of`, the first day after their
month: before live pulls began they are the only history there is. Backtests
over that period inherit their hindsight (AI_NOTES/Task7.md).

Values are floats or None, so a feature dict can be stored as JSON as is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.db.queries import get_vintage_history
from eco_prediction.features.engineering import (
    Features,
    _months_between,
    _store,
    pct_change,
    to_monthly,
)

COUNT_SERIES = {
    "arxiv": "ARXIV_AI_SUBMISSIONS",
    "hf": "HF_NEW_MODELS",
    "github": "GITHUB_AI_ACTIVITY",
    "stars": "GITHUB_AI_STARS",
}
STEP_SERIES = {
    "gpqa": "EPOCH_GPQA_SOTA",
    "swebench": "EPOCH_SWEBENCH_SOTA",
    "arc2": "EPOCH_ARC2_SOTA",
    "flops": "EPOCH_LARGEST_FLOPS",
}
PERCENT_SERIES = ("gpqa", "swebench", "arc2")
PREFIXES = {**COUNT_SERIES, **STEP_SERIES}
SERIES_PREFIX = {series_id: prefix for prefix, series_id in PREFIXES.items()}
AI_FEATURE_SERIES = tuple(PREFIXES.values())

LAGS = (1, 3, 6, 12)
CHANGES = (("1m", 1), ("3m", 3), ("12m", 12))
CORRELATION_MONTHS = 12
TREND_ORIGIN = date(2010, 1, 1)


def _float(x: object) -> float | None:
    if x is None:
        return None
    value = float(x)  # type: ignore[arg-type]
    return None if math.isnan(value) or math.isinf(value) else value


def store_from_conn(conn: Connection) -> VintageStore:
    """A store that loads each AI series from `conn` on first use."""
    return VintageStore(loader=lambda series_id: get_vintage_history(conn, series_id))


class AIFeatureEngineer:
    """Builds one AI feature dict per as-of date. The feature names are fixed."""

    def __init__(self, *, ffill_limit: int = 1, max_stale_months: int = 3) -> None:
        self.ffill_limit = ffill_limit
        self.max_stale_months = max_stale_months
        self._rows: WeakKeyDictionary[VintageStore, dict[date, Features]] = (
            WeakKeyDictionary()
        )

    # --- building --------------------------------------------------------------

    def build_features(
        self,
        series_data: Mapping[str, pd.Series],
        as_of_date: date,
        target_series: str | None = None,
    ) -> Features:
        """Feature vector from raw series (series id -> values by period)."""
        monthly = {
            prefix: to_monthly(series_data.get(series_id), as_of_date, self.ffill_limit)
            for prefix, series_id in PREFIXES.items()
        }
        latest = {p: self._latest(m, as_of_date) for p, m in monthly.items()}
        features: Features = {}
        for prefix in COUNT_SERIES:
            features.update(
                self._count(prefix, monthly[prefix], latest[prefix], as_of_date)
            )
        for prefix in STEP_SERIES:
            features.update(
                self._step(prefix, monthly[prefix], latest[prefix], as_of_date)
            )
        features.update(self._cross(monthly, latest))
        features.update(self._calendar(as_of_date))
        return self._with_target(features, target_series)

    def build_features_as_of(
        self,
        source: VintageStore | PointInTimeData,
        as_of_date: date,
        target_series: str | None = None,
    ) -> Features:
        """Feature vector from the vintages known on as_of_date. Cached per store."""
        rows = self._rows.setdefault(_store(source, as_of_date), {})
        if as_of_date not in rows:
            series = self.series_as_of(source, as_of_date)
            rows[as_of_date] = self.build_features(series, as_of_date)
        return self._with_target(dict(rows[as_of_date]), target_series)

    def build_matrix(
        self,
        source: VintageStore | PointInTimeData,
        as_of_dates: Iterable[date],
        target_series: str | None = None,
    ) -> pd.DataFrame:
        """One row per as-of date (DatetimeIndex `as_of`), each from data known then."""
        dates = sorted(set(as_of_dates))
        rows = [self.build_features_as_of(source, d, target_series) for d in dates]
        index = pd.DatetimeIndex(dates, name="as_of")
        if not rows:
            return pd.DataFrame(index=index, dtype="float64")
        return pd.DataFrame(rows, index=index, dtype="float64")

    @staticmethod
    def series_as_of(
        source: VintageStore | PointInTimeData, as_of_date: date
    ) -> dict[str, pd.Series]:
        """Every AI series as known on as_of_date, indexed by period."""
        store = _store(source, as_of_date)
        out = {}
        for series_id in AI_FEATURE_SERIES:
            known = store.known(series_id, as_of_date)
            out[series_id] = pd.Series(
                known["value"].to_numpy(),
                index=pd.DatetimeIndex(known["observed_at"], name="observed_at"),
                name=series_id,
                dtype="float64",
            )
        return out

    # --- per kind ----------------------------------------------------------------

    def _count(
        self, p: str, s: pd.Series, latest: pd.Timestamp | None, as_of: date
    ) -> Features:
        level = _at(s, latest)
        out: Features = {
            p: level,
            f"{p}_log": math.log(level) if level and level > 0 else None,
        }
        for lag in LAGS:
            out[f"{p}_lag{lag}"] = _at(s, latest, lag)
        for suffix, months in CHANGES:
            out[f"{p}_growth_{suffix}"] = _at(pct_change(s, months), latest)
        out[f"{p}_months_behind"] = _behind(latest, as_of)
        return out

    def _step(
        self, p: str, s: pd.Series, latest: pd.Timestamp | None, as_of: date
    ) -> Features:
        level = _at(s, latest)
        out: Features = {p: level}
        for lag in LAGS:
            out[f"{p}_lag{lag}"] = _at(s, latest, lag)
        for suffix, months in CHANGES:
            out[f"{p}_chg_{suffix}"] = _at(s.diff(months), latest)
        since, jump = _last_change(s, latest)
        out[f"{p}_months_since_change"] = since
        out[f"{p}_last_jump"] = jump
        out[f"{p}_months_behind"] = _behind(latest, as_of)
        if p in PERCENT_SERIES:
            out[f"{p}_headroom"] = None if level is None else 100 - level
        return out

    def _cross(
        self,
        monthly: Mapping[str, pd.Series],
        latest: Mapping[str, pd.Timestamp | None],
    ) -> Features:
        out: Features = {}
        for other in ("hf", "github"):
            end = _common_latest(latest["arxiv"], latest[other])
            out[f"corr12_{other}_arxiv"] = _log_change_corr(
                monthly[other], monthly["arxiv"], end
            )
            a, b = _at(monthly[other], end), _at(monthly["arxiv"], end)
            out[f"log_ratio_{other}_arxiv"] = (
                math.log(a / b) if a and b and a > 0 and b > 0 else None
            )
        increases = [
            _increases(monthly[p], latest[p]) for p in PERCENT_SERIES if latest[p]
        ]
        known = [n for n in increases if n is not None]
        out["sota_increases_12m"] = float(sum(known)) if known else None
        return out

    @staticmethod
    def _calendar(as_of: date) -> Features:
        out: Features = {}
        for month in range(1, 13):
            out[f"month_{month}"] = float(as_of.month == month)
        quarter = (as_of.month - 1) // 3 + 1
        for q in range(1, 5):
            out[f"quarter_{q}"] = float(quarter == q)
        out["trend_months"] = float(_months_between(TREND_ORIGIN, as_of))
        return out

    @staticmethod
    def _with_target(features: Features, target_series: str | None) -> Features:
        if target_series is None:
            return features
        if target_series not in SERIES_PREFIX:
            raise ValueError(f"no AI features for series {target_series!r}")
        prefix = SERIES_PREFIX[target_series]
        aliases = {
            "target" + name[len(prefix) :]: value
            for name, value in features.items()
            if name == prefix or name.startswith(prefix + "_")
        }
        return {**features, **aliases}

    def _latest(self, monthly: pd.Series, as_of: date) -> pd.Timestamp | None:
        """Latest month with a value, unless it's more than max_stale_months old."""
        valid = monthly.dropna()
        if valid.empty:
            return None
        last = valid.index[-1]
        if _months_between(last, as_of) > self.max_stale_months:
            return None
        return last


# --- helpers -------------------------------------------------------------------


def _at(series: pd.Series, latest: pd.Timestamp | None, lag: int = 0) -> float | None:
    """Value `lag` months before the latest month, if there is one."""
    if latest is None:
        return None
    return _float(series.get(latest - pd.DateOffset(months=lag)))


def _behind(latest: pd.Timestamp | None, as_of: date) -> float | None:
    return None if latest is None else float(_months_between(latest, as_of))


def _last_change(
    series: pd.Series, latest: pd.Timestamp | None
) -> tuple[float | None, float | None]:
    """(months from the latest change to the latest month, size of that change).

    With no change in the series' history: (months since its first value, None).
    """
    if latest is None:
        return None, None
    s = series[series.index <= latest].dropna()
    changed = s.diff()
    changed = changed[changed != 0].dropna()
    if changed.empty:
        return float(_months_between(s.index[0], latest)), None
    when = changed.index[-1]
    return float(_months_between(when, latest)), float(changed.iloc[-1])


def _common_latest(
    a: pd.Timestamp | None, b: pd.Timestamp | None
) -> pd.Timestamp | None:
    return None if a is None or b is None else min(a, b)


def _log_change_corr(
    x: pd.Series, y: pd.Series, end: pd.Timestamp | None
) -> float | None:
    """Correlation of monthly log changes over the CORRELATION_MONTHS ending at
    `end`; None unless every month is present and both vary."""
    if end is None:
        return None
    start = end - pd.DateOffset(months=CORRELATION_MONTHS)
    window = pd.DataFrame({"x": x, "y": y})
    window = window[(window.index >= start) & (window.index <= end)]
    if (window <= 0).any().any():
        return None
    changes = window.transform(np.log).diff().dropna()
    if len(changes) < CORRELATION_MONTHS or (changes.std() == 0).any():
        return None
    return _float(changes["x"].corr(changes["y"]))


def _increases(series: pd.Series, latest: pd.Timestamp | None) -> int | None:
    """Monthly increases in the 12 months up to latest; None without a full year."""
    if latest is None:
        return None
    window = series[
        (series.index > latest - pd.DateOffset(months=12)) & (series.index <= latest)
    ]
    previous = series.get(latest - pd.DateOffset(months=12))
    if previous is None or pd.isna(previous) or window.isna().any():
        return None
    steps = pd.concat([pd.Series([previous]), window.reset_index(drop=True)]).diff()
    return int((steps > 0).sum())


def build_ai_features(
    conn: Connection, as_of_date: date, target_series: str | None = None
) -> Features:
    """AI features as known on as_of_date, read from `conn`."""
    return AIFeatureEngineer().build_features_as_of(
        store_from_conn(conn), as_of_date, target_series
    )
