"""Shared ingestion for the AI domain's monthly series (arXiv, Hugging Face,
GitHub, Epoch AI).

Only months that have ended are stored; the current month is never stored
half-counted. Two kinds of rows:
- **Backfill**: one row per month that has no rows yet, with `as_of` = the
  first day of the next month, when the count was complete, and
  `reconstructed = true`. The value is today's count, which can differ from
  what was countable then (late cross-lists, deleted models). Each month is
  committed as soon as it's counted, and months already stored are skipped, so
  an interrupted backfill resumes.
- **Live**: re-counts the last few ended months and stores a row with
  `as_of` = today and `reconstructed = false` for each month that is new,
  whose count changed since its latest row, or that has no live row yet. The
  last case confirms a backfilled month with a genuine vintage even when the
  count is unchanged: AI questions resolve to a month's first live value
  (AI tasks 9 and 11), so every month a live pull has seen must have one.

A count of zero is refused: every count series here has had activity in every
month since its first, so a zero means the source failed, not a quiet month.
Series whose zero is real (a benchmark score) set `allow_zero`; negatives are
always refused.

Each source module defines a `MonthlySeries` and calls `run_cli` with a client
that implements `MonthlyCounter`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from psycopg2.extensions import connection as Connection

from eco_prediction.ai.months import (
    MonthlyCount,
    iter_months,
    last_complete_month,
    month_start,
    next_month,
)
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.store import (
    AIObservation,
    AISeriesInfo,
    ensure_source,
    insert_ai_observations,
    latest_stored,
    upsert_ai_series,
)
from eco_prediction.db.connection import ConfigError, connection


class MonthlyCounter(Protocol):
    def count_months(self, months: Sequence[date]) -> Iterable[MonthlyCount]:
        """Counts for `months` (ended, ascending), each yielded once it's known,
        in any order."""
        ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class MonthlySeries:
    source: tuple[str, str, str]  # name, url, description for `sources`
    info: AISeriesInfo
    first_month: date  # earliest month the source can count; backfill default
    live_months: int = 3  # ended months a live pull re-counts
    allow_zero: bool = False  # True when 0 is a real value, e.g. a 0% score
    value_format: str = ".0f"  # how dry runs and progress print a value


@dataclass(frozen=True)
class IngestResult:
    mode: str  # "backfill" or "live"
    fetched: int  # months counted
    inserted: int  # rows new to the database


def register_series(conn: Connection, series: MonthlySeries) -> int:
    source_pk = ensure_source(conn, *series.source)
    return upsert_ai_series(conn, source_pk, series.info)


def backfill_row(series: MonthlySeries, c: MonthlyCount) -> AIObservation:
    """A reconstructed row, known as of the first day after its month."""
    _check(series, c)
    return AIObservation(c.month, next_month(c.month), c.count, reconstructed=True)


def live_rows(
    series: MonthlySeries,
    counts: Iterable[MonthlyCount],
    prior: Mapping[date, float | None],
    today: date,
    confirmed: Collection[date] = frozenset(),
) -> list[AIObservation]:
    """Live rows for months that are new, changed, or not yet `confirmed` (no
    live row so far)."""
    rows = []
    for c in counts:
        _check(series, c)
        if prior.get(c.month) != c.count or c.month not in confirmed:
            rows.append(AIObservation(c.month, today, c.count, reconstructed=False))
    return rows


def live_months_stored(conn: Connection, series_pk: int) -> set[date]:
    """Months with at least one live (not reconstructed) row."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT observed_at FROM observations"
            " WHERE series_id = %s AND NOT reconstructed",
            (series_pk,),
        )
        return {row[0] for row in cur.fetchall()}


def ingest_backfill(
    conn: Connection,
    series: MonthlySeries,
    counter: MonthlyCounter,
    start: date,
    end: date,
    *,
    on_row: Callable[[AIObservation, int], None] | None = None,
) -> IngestResult:
    """Backfill the months from `start` through `end` that have no rows.

    Commits after registering the series and after each month, so an
    interrupted backfill keeps what it has counted. `on_row(row, inserted)`
    is called after each commit, for progress output.
    """
    series_pk = register_series(conn, series)
    conn.commit()
    _, stored = latest_stored(conn, series_pk)
    missing = [m for m in iter_months(start, end) if m not in stored]
    fetched = inserted = 0
    if missing:
        for c in counter.count_months(missing):
            row = backfill_row(series, c)
            n = insert_ai_observations(conn, series_pk, [row])
            conn.commit()
            fetched += 1
            inserted += n
            if on_row:
                on_row(row, n)
    return IngestResult("backfill", fetched, inserted)


def ingest_live(
    conn: Connection,
    series: MonthlySeries,
    counter: MonthlyCounter,
    today: date,
    months: int | None = None,
) -> IngestResult:
    """Re-count the last `months` ended months and store what changed. No commit."""
    series_pk = register_series(conn, series)
    _, prior = latest_stored(conn, series_pk)
    counts = list(
        counter.count_months(live_months(today, months or series.live_months))
    )
    rows = live_rows(series, counts, prior, today, live_months_stored(conn, series_pk))
    inserted = insert_ai_observations(conn, series_pk, rows)
    return IngestResult("live", len(counts), inserted)


def live_months(today: date, months: int) -> list[date]:
    """The `months` most recent months that have ended, ascending."""
    end = start = last_complete_month(today)
    for _ in range(months - 1):
        start = last_complete_month(start)
    return list(iter_months(start, end))


def _check(series: MonthlySeries, c: MonthlyCount) -> None:
    if c.count < 0 or (c.count == 0 and not series.allow_zero):
        raise SourceAPIError(
            f"{series.info.series_id}: source returned {c.count} for "
            f"{c.month:%Y-%m}; refusing to store it",
            source=series.source[0],
        )


def format_row(row: AIObservation, value_format: str = ".0f") -> str:
    kind = "reconstructed" if row.reconstructed else "live"
    value = format(row.value, f">7{value_format}")
    return f"{row.observed_at:%Y-%m}  {value}  as_of={row.as_of}  {kind}"


def run_cli(
    argv: Sequence[str] | None,
    series: MonthlySeries,
    *,
    description: str,
    client_factory: Callable[[], MonthlyCounter],
    today: Callable[[], date] = date.today,
) -> int:
    """`--backfill`, `--start`, `--months` and `--dry-run` for one series."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="fill every ended month since --start that has no rows yet",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=series.first_month,
        help=f"first month to backfill (default {series.first_month})",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=series.live_months,
        help=f"ended months a live pull re-counts (default {series.live_months})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rows without writing to the database",
    )
    args = parser.parse_args(argv)
    if args.months < 1:
        parser.error("--months must be at least 1")
    if month_start(args.start) < series.first_month:
        parser.error(f"--start can't be before {series.first_month}")

    run_date = today()
    sid = series.info.series_id
    if args.backfill:
        months = list(iter_months(args.start, last_complete_month(run_date)))
    else:
        months = live_months(run_date, args.months)

    try:
        client = client_factory()
        try:
            if args.dry_run:
                for c in client.count_months(months):
                    rows = (
                        [backfill_row(series, c)]
                        if args.backfill
                        else live_rows(series, [c], {}, run_date)
                    )
                    for row in rows:
                        print(format_row(row, series.value_format), flush=True)
                return 0
            with connection() as conn:
                if args.backfill:
                    result = ingest_backfill(
                        conn,
                        series,
                        client,
                        months[0],
                        months[-1],
                        on_row=lambda row, n: print(
                            f"{sid} {format_row(row, series.value_format)}  inserted={n}",
                            flush=True,
                        ),
                    )
                else:
                    result = ingest_live(conn, series, client, run_date, args.months)
        finally:
            client.close()
    except (ConfigError, SourceAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{sid} {result.mode} fetched={result.fetched} inserted={result.inserted}")
    return 0
