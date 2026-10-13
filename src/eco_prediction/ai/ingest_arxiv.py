"""Load monthly arXiv AI submission counts (series ARXIV_AI_SUBMISSIONS).

Usage:
    uv run python -m eco_prediction.ai.ingest_arxiv               # live pull
    uv run python -m eco_prediction.ai.ingest_arxiv --backfill    # 2010 to last month
    uv run python -m eco_prediction.ai.ingest_arxiv --backfill --dry-run

The value for a month is the number of papers first submitted that month in
cs.AI, cs.LG or cs.CL (see `ArxivClient`). Only months that have ended are
stored; the current month is never stored half-counted.

Two kinds of rows:
- **Backfill** (`--backfill`): one row per month that has no rows yet, with
  `as_of` = the first day of the next month, when the count was complete, and
  `reconstructed = true`. The value is today's count, which can differ a
  little from what was countable then (late cross-lists, withdrawals).
  Months already stored are skipped, so an interrupted backfill resumes.
- **Live** (default): re-counts the last LIVE_MONTHS ended months and stores a
  row with `as_of` = today and `reconstructed = false` for each month that is
  new or whose count changed since its latest row.

`--dry-run` fetches and prints the counts without touching the database.
The backfill is committed one year at a time.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from psycopg2.extensions import connection as Connection

from eco_prediction.ai.sources.arxiv_client import (
    AI_CATEGORIES,
    ArxivClient,
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

ARXIV_SOURCE = (
    "arXiv",
    "https://arxiv.org",
    "arXiv preprint server (export.arxiv.org search API)",
)

SERIES = AISeriesInfo(
    series_id="ARXIV_AI_SUBMISSIONS",
    name=(
        "arXiv submissions in " + ", ".join(AI_CATEGORIES) + " (primary or "
        "cross-listed, each paper once), by month of first submission"
    ),
    units="Papers",
    frequency="Monthly",
)

BACKFILL_START = date(2010, 1, 1)
LIVE_MONTHS = 3


@dataclass(frozen=True)
class ArxivIngestResult:
    mode: str  # "backfill" or "live"
    fetched: int  # months counted
    inserted: int  # rows new to the database


def register_series(conn: Connection) -> int:
    source_pk = ensure_source(conn, *ARXIV_SOURCE)
    return upsert_ai_series(conn, source_pk, SERIES)


def backfill_rows(counts: Sequence[MonthlyCount]) -> list[AIObservation]:
    """Reconstructed rows, each known as of the first day after its month."""
    return [
        AIObservation(c.month, next_month(c.month), c.count, reconstructed=True)
        for c in _checked(counts)
    ]


def live_rows(
    counts: Sequence[MonthlyCount], prior: dict[date, float | None], today: date
) -> list[AIObservation]:
    """Live rows for months that are new or whose count changed."""
    return [
        AIObservation(c.month, today, c.count, reconstructed=False)
        for c in _checked(counts)
        if prior.get(c.month) != c.count
    ]


def ingest_backfill(
    conn: Connection, client: ArxivClient, start: date, end: date
) -> ArxivIngestResult:
    """Backfill the months from `start` through `end` that have no rows. No commit."""
    series_pk = register_series(conn)
    _, stored = latest_stored(conn, series_pk)
    missing = [m for m in iter_months(start, end) if m not in stored]
    counts = [MonthlyCount(m, client.count_submissions(m)) for m in missing]
    inserted = insert_ai_observations(conn, series_pk, backfill_rows(counts))
    return ArxivIngestResult("backfill", len(counts), inserted)


def ingest_live(
    conn: Connection, client: ArxivClient, today: date, months: int = LIVE_MONTHS
) -> ArxivIngestResult:
    """Re-count the last `months` ended months and store what changed. No commit."""
    series_pk = register_series(conn)
    _, prior = latest_stored(conn, series_pk)
    counts = client.monthly_counts(*live_window(today, months))
    inserted = insert_ai_observations(conn, series_pk, live_rows(counts, prior, today))
    return ArxivIngestResult("live", len(counts), inserted)


def live_window(today: date, months: int) -> tuple[date, date]:
    """First and last of the `months` most recent months that have ended."""
    end = start = last_complete_month(today)
    for _ in range(months - 1):
        start = last_complete_month(start)
    return start, end


def _checked(counts: Sequence[MonthlyCount]) -> Sequence[MonthlyCount]:
    """Reject zero counts: these categories have had papers every month since
    2010, so a zero is the search index failing, not a real month."""
    for c in counts:
        if c.count <= 0:
            raise SourceAPIError(
                f"arxiv returned {c.count} papers for {c.month:%Y-%m}; "
                "refusing to store it",
                source="arxiv",
            )
    return counts


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[], ArxivClient] = ArxivClient.from_env,
    today: Callable[[], date] = date.today,
) -> int:
    parser = argparse.ArgumentParser(
        description="Load monthly arXiv AI submission counts into the database."
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="fill every ended month since --start that has no rows yet",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=BACKFILL_START,
        help=f"first month to backfill (default {BACKFILL_START})",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=LIVE_MONTHS,
        help=f"ended months a live pull re-counts (default {LIVE_MONTHS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the counts without writing to the database",
    )
    args = parser.parse_args(argv)
    if args.months < 1:
        parser.error("--months must be at least 1")

    run_date = today()
    end = last_complete_month(run_date)
    if args.backfill:
        start = month_start(args.start)
    else:
        start, end = live_window(run_date, args.months)

    try:
        with client_factory() as client:
            if args.dry_run:
                _dry_run(client, start, end, run_date, backfill=args.backfill)
                return 0
            if args.backfill:
                results = []
                for year in range(start.year, end.year + 1):
                    chunk_start = max(start, date(year, 1, 1))
                    chunk_end = min(end, date(year, 12, 1))
                    with connection() as conn:
                        results.append(
                            ingest_backfill(conn, client, chunk_start, chunk_end)
                        )
                    print(
                        f"{SERIES.series_id} backfill {year}: "
                        f"fetched={results[-1].fetched} inserted={results[-1].inserted}",
                        flush=True,
                    )
                fetched = sum(r.fetched for r in results)
                inserted = sum(r.inserted for r in results)
            else:
                with connection() as conn:
                    result = ingest_live(conn, client, run_date, args.months)
                fetched, inserted = result.fetched, result.inserted
    except (ConfigError, SourceAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mode = "backfill" if args.backfill else "live"
    print(f"{SERIES.series_id} {mode} fetched={fetched} inserted={inserted}")
    return 0


def _dry_run(
    client: ArxivClient, start: date, end: date, run_date: date, *, backfill: bool
) -> None:
    """Print each month's row as it would be stored, as soon as it is counted."""
    for month in iter_months(start, end):
        counts = [MonthlyCount(month, client.count_submissions(month))]
        rows = backfill_rows(counts) if backfill else live_rows(counts, {}, run_date)
        for r in rows:
            kind = "reconstructed" if r.reconstructed else "live"
            print(
                f"{r.observed_at:%Y-%m}  {r.value:>7.0f}  as_of={r.as_of}  {kind}",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
