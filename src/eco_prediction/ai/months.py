"""Calendar-month helpers for the AI series, which are all monthly.

A month is represented by its first day.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class MonthlyCount:
    month: date  # first day of the month
    count: float  # a count, or for Epoch's step series the month's running max


def month_start(d: date) -> date:
    return d.replace(day=1)


def next_month(d: date) -> date:
    return date(d.year + d.month // 12, d.month % 12 + 1, 1)


def iter_months(start: date, end: date) -> Iterator[date]:
    """First day of each month from `start`'s through `end`'s."""
    month, last = month_start(start), month_start(end)
    while month <= last:
        yield month
        month = next_month(month)


def last_complete_month(today: date) -> date:
    """First day of the latest month that has fully ended before `today`."""
    return month_start(month_start(today) - timedelta(days=1))
