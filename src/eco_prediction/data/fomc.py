"""FOMC meeting calendar, the fed funds target, and decision outcomes.

Usage:
    from eco_prediction.data.fomc import FOMC_MEETINGS, decision, fed_funds_target

    store = VintageStore.from_db(backfill=True, release_lags=RELEASE_LAGS)
    fed_funds_target(store, date(2024, 9, 30))       # daily midpoint, as known then
    decision(store, date(2024, 9, 18), date(2024, 9, 30))  # 'cut'
    meetings_between(date(2024, 1, 1), date(2024, 12, 31))

`FOMC_MEETINGS` holds the decision day (the last day) of every scheduled
meeting from 1994, when the FOMC started announcing its decisions, to 2027.
They were taken from federalreserve.gov's calendar pages on 2026-10-08:
fomchistorical{year}.htm for 1994-2020 and fomccalendars.htm for 2021-2027.
Left out: unscheduled meetings and conference calls (e.g. 2020-03-03 and
2020-03-15), the cancelled 2020-03-17/18 meeting, the 2025-08-22 notation
vote, and 2003-09-15, a meeting with an agenda but no policy statement.
Future dates are the Fed's published schedule and can change.

The target is DFEDTAR (a single rate) to 2008-12-15, then the midpoint of the
range, DFEDTARU - 0.125 (every range since has been 25 bp wide), so the
series is continuous. A meeting's decision compares the target the day before
the decision day with the day after it; since 2008 changes take effect the day
after the announcement. Moves between meetings (2001, 2008, March 2020) are in
the target but are not decisions of any scheduled meeting.

DFEDTAR has a single ALFRED vintage (2008-12-15), so its release lag can't be
estimated; pass `RELEASE_LAGS` to `VintageStore` so its history is dated one
day after each period, as it was public.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from eco_prediction.backtest.data import VintageStore

Outcome = Literal["cut", "hold", "hike"]
OUTCOMES: tuple[Outcome, ...] = ("cut", "hold", "hike")

TARGET_SERIES = "DFEDTAR"  # to 2008-12-15
RANGE_UPPER_SERIES = "DFEDTARU"  # from 2008-12-16
RANGE_START = date(2008, 12, 16)
RANGE_HALF_WIDTH = 0.125
RELEASE_LAGS = {TARGET_SERIES: 1, RANGE_UPPER_SERIES: 1}

_MEETINGS = {
    1994: [(2, 4), (3, 22), (5, 17), (7, 6), (8, 16), (9, 27), (11, 15), (12, 20)],
    1995: [(2, 1), (3, 28), (5, 23), (7, 6), (8, 22), (9, 26), (11, 15), (12, 19)],
    1996: [(1, 31), (3, 26), (5, 21), (7, 3), (8, 20), (9, 24), (11, 13), (12, 17)],
    1997: [(2, 5), (3, 25), (5, 20), (7, 2), (8, 19), (9, 30), (11, 12), (12, 16)],
    1998: [(2, 4), (3, 31), (5, 19), (7, 1), (8, 18), (9, 29), (11, 17), (12, 22)],
    1999: [(2, 3), (3, 30), (5, 18), (6, 30), (8, 24), (10, 5), (11, 16), (12, 21)],
    2000: [(2, 2), (3, 21), (5, 16), (6, 28), (8, 22), (10, 3), (11, 15), (12, 19)],
    2001: [(1, 31), (3, 20), (5, 15), (6, 27), (8, 21), (10, 2), (11, 6), (12, 11)],
    2002: [(1, 30), (3, 19), (5, 7), (6, 26), (8, 13), (9, 24), (11, 6), (12, 10)],
    2003: [(1, 29), (3, 18), (5, 6), (6, 25), (8, 12), (9, 16), (10, 28), (12, 9)],
    2004: [(1, 28), (3, 16), (5, 4), (6, 30), (8, 10), (9, 21), (11, 10), (12, 14)],
    2005: [(2, 2), (3, 22), (5, 3), (6, 30), (8, 9), (9, 20), (11, 1), (12, 13)],
    2006: [(1, 31), (3, 28), (5, 10), (6, 29), (8, 8), (9, 20), (10, 25), (12, 12)],
    2007: [(1, 31), (3, 21), (5, 9), (6, 28), (8, 7), (9, 18), (10, 31), (12, 11)],
    2008: [(1, 30), (3, 18), (4, 30), (6, 25), (8, 5), (9, 16), (10, 29), (12, 16)],
    2009: [(1, 28), (3, 18), (4, 29), (6, 24), (8, 12), (9, 23), (11, 4), (12, 16)],
    2010: [(1, 27), (3, 16), (4, 28), (6, 23), (8, 10), (9, 21), (11, 3), (12, 14)],
    2011: [(1, 26), (3, 15), (4, 27), (6, 22), (8, 9), (9, 21), (11, 2), (12, 13)],
    2012: [(1, 25), (3, 13), (4, 25), (6, 20), (8, 1), (9, 13), (10, 24), (12, 12)],
    2013: [(1, 30), (3, 20), (5, 1), (6, 19), (7, 31), (9, 18), (10, 30), (12, 18)],
    2014: [(1, 29), (3, 19), (4, 30), (6, 18), (7, 30), (9, 17), (10, 29), (12, 17)],
    2015: [(1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 17), (10, 28), (12, 16)],
    2016: [(1, 27), (3, 16), (4, 27), (6, 15), (7, 27), (9, 21), (11, 2), (12, 14)],
    2017: [(2, 1), (3, 15), (5, 3), (6, 14), (7, 26), (9, 20), (11, 1), (12, 13)],
    2018: [(1, 31), (3, 21), (5, 2), (6, 13), (8, 1), (9, 26), (11, 8), (12, 19)],
    2019: [(1, 30), (3, 20), (5, 1), (6, 19), (7, 31), (9, 18), (10, 30), (12, 11)],
    2020: [(1, 29), (4, 29), (6, 10), (7, 29), (9, 16), (11, 5), (12, 16)],
    2021: [(1, 27), (3, 17), (4, 28), (6, 16), (7, 28), (9, 22), (11, 3), (12, 15)],
    2022: [(1, 26), (3, 16), (5, 4), (6, 15), (7, 27), (9, 21), (11, 2), (12, 14)],
    2023: [(2, 1), (3, 22), (5, 3), (6, 14), (7, 26), (9, 20), (11, 1), (12, 13)],
    2024: [(1, 31), (3, 20), (5, 1), (6, 12), (7, 31), (9, 18), (11, 7), (12, 18)],
    2025: [(1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (10, 29), (12, 10)],
    2026: [(1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 16), (10, 28), (12, 9)],
    2027: [(1, 27), (3, 17), (4, 28), (6, 9), (7, 28), (9, 15), (10, 27), (12, 8)],
}
FOMC_MEETINGS: tuple[date, ...] = tuple(
    date(year, month, day) for year, days in _MEETINGS.items() for month, day in days
)


def meetings_between(start: date, end: date) -> list[date]:
    """Scheduled decision days in [start, end]."""
    return [m for m in FOMC_MEETINGS if start <= m <= end]


def fed_funds_target(store: VintageStore, as_of: date) -> pd.Series:
    """Daily fed funds target (midpoint of the range since 2008), as known on as_of."""
    parts = []
    for series_id, shift in (
        (TARGET_SERIES, 0.0),
        (RANGE_UPPER_SERIES, -RANGE_HALF_WIDTH),
    ):
        known = store.known(series_id, as_of)
        parts.append(
            pd.Series(
                known["value"].to_numpy() + shift,
                index=pd.DatetimeIndex(known["observed_at"], name="observed_at"),
            )
        )
    target = pd.concat(parts).dropna().sort_index()
    return target[~target.index.duplicated(keep="last")].rename("fed_funds_target")


def value_at(series: pd.Series, when: pd.Timestamp) -> float:
    """The latest value on or before `when`; NaN if there is none."""
    position = int(series.index.searchsorted(when, side="right"))  # sorted index
    return float(series.iloc[position - 1]) if position else float("nan")


def classify(before: float, after: float) -> Outcome:
    if after > before:
        return "hike"
    if after < before:
        return "cut"
    return "hold"


def decision(
    store: VintageStore,
    meeting: date,
    as_of: date,
    target: pd.Series | None = None,
) -> Outcome | None:
    """What `meeting` decided, as known on as_of; None if not known yet.

    Pass `target` (from `fed_funds_target`) to reuse it across meetings.
    """
    if target is None:
        target = fed_funds_target(store, as_of)
    day_after = pd.Timestamp(meeting + timedelta(days=1))
    if target.empty or target.index[-1] < day_after:
        return None
    before = value_at(target, pd.Timestamp(meeting - timedelta(days=1)))
    if math.isnan(before):
        return None
    return classify(before, value_at(target, day_after))
