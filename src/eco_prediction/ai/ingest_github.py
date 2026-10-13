"""Load GitHub AI activity: new AI repos per month, and a monthly stars snapshot.

Usage:
    uv run python -m eco_prediction.ai.ingest_github                      # live repo counts
    uv run python -m eco_prediction.ai.ingest_github --backfill           # 2017-01 on
    uv run python -m eco_prediction.ai.ingest_github --backfill --dry-run
    uv run python -m eco_prediction.ai.ingest_github stars                # stars snapshot
    uv run python -m eco_prediction.ai.ingest_github stars --dry-run

Two series (see `GitHubClient` for how each is counted):
- **GITHUB_AI_ACTIVITY**, the forecast target `github_ai_activity`: public
  repos created each month with topic ai, machine-learning or deep-learning,
  each repo once. Backfill and live rows follow `ai/monthly_ingest.py`. A
  backfill is 7 searches per month from 2017-01, ~31 minutes with a token.
- **GITHUB_AI_STARS**: stars summed over the 1,000 most-starred such repos.
  Only today's star counts exist, so there's no backfill: `stars` stores one
  snapshot per calendar month, dated to the month it was taken in, and does
  nothing if this month already has one. ~30 searches, about a minute.

`GITHUB_TOKEN` in `.env` raises search from 10 to 30 requests a minute (the client keeps to 9 and 27).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import date
from typing import Protocol

from psycopg2.extensions import connection as Connection

from eco_prediction.ai.monthly_ingest import (
    IngestResult,
    MonthlyCounter,
    MonthlySeries,
    run_cli,
)
from eco_prediction.ai.months import month_start
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.sources.github_client import (
    AI_TOPICS,
    FIRST_MONTH,
    TOP_N,
    GitHubClient,
)
from eco_prediction.ai.store import (
    AIObservation,
    AISeriesInfo,
    ensure_source,
    insert_ai_observations,
    latest_stored,
    upsert_ai_series,
)
from eco_prediction.db.connection import ConfigError, connection

GITHUB_SOURCE = (
    "GitHub",
    "https://github.com",
    "GitHub repository search API (api.github.com/search/repositories)",
)
_TOPICS = ", ".join(AI_TOPICS)

SERIES = MonthlySeries(
    source=GITHUB_SOURCE,
    info=AISeriesInfo(
        series_id="GITHUB_AI_ACTIVITY",
        name=(
            f"New public GitHub repos with topic {_TOPICS} (each repo once, "
            "forks excluded), by month created"
        ),
        units="Repositories",
        frequency="Monthly",
    ),
    first_month=FIRST_MONTH,
)

STARS_SERIES = AISeriesInfo(
    series_id="GITHUB_AI_STARS",
    name=(
        f"Stars on the {TOP_N:,} most-starred GitHub repos with topic {_TOPICS} "
        "(monthly snapshot)"
    ),
    units="Stars",
    frequency="Monthly",
)


class StarCounter(Protocol):
    def top_starred_total(self) -> int: ...

    def close(self) -> None: ...


def stars_row(total: int, today: date) -> AIObservation:
    if total <= 0:
        raise SourceAPIError(
            f"{STARS_SERIES.series_id}: github returned {total} stars; "
            "refusing to store it",
            source="github",
        )
    return AIObservation(month_start(today), today, total, reconstructed=False)


def ingest_stars_snapshot(
    conn: Connection, client: StarCounter, today: date
) -> IngestResult:
    """Store this month's stars snapshot unless it's already taken. No commit."""
    source_pk = ensure_source(conn, *GITHUB_SOURCE)
    series_pk = upsert_ai_series(conn, source_pk, STARS_SERIES)
    _, stored = latest_stored(conn, series_pk)
    if month_start(today) in stored:
        return IngestResult("snapshot", 0, 0)
    row = stars_row(client.top_starred_total(), today)
    return IngestResult("snapshot", 1, insert_ai_observations(conn, series_pk, [row]))


def stars_main(
    argv: Sequence[str],
    *,
    client_factory: Callable[[], StarCounter],
    today: Callable[[], date],
) -> int:
    parser = argparse.ArgumentParser(
        prog="ingest_github stars",
        description="Store this month's snapshot of stars on the top AI repos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the snapshot without writing to the database",
    )
    args = parser.parse_args(argv)
    run_date = today()
    sid = STARS_SERIES.series_id
    try:
        client = client_factory()
        try:
            if args.dry_run:
                row = stars_row(client.top_starred_total(), run_date)
                print(
                    f"{sid} {row.observed_at:%Y-%m}  {row.value:.0f}  as_of={row.as_of}"
                )
                return 0
            with connection() as conn:
                result = ingest_stars_snapshot(conn, client, run_date)
        finally:
            client.close()
    except (ConfigError, SourceAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.fetched == 0:
        print(f"{sid} snapshot for {run_date:%Y-%m} already stored")
    else:
        print(f"{sid} snapshot inserted={result.inserted}")
    return 0


class GitHubCounter(MonthlyCounter, StarCounter, Protocol):
    pass


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], GitHubCounter] = GitHubClient.from_env,
    today: Callable[[], date] = date.today,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["stars"]:
        return stars_main(args[1:], client_factory=client_factory, today=today)
    return run_cli(
        args,
        SERIES,
        description=(
            "Load monthly counts of new AI repos on GitHub into the database "
            "(or `stars` for the stars snapshot)."
        ),
        client_factory=client_factory,
        today=today,
    )


if __name__ == "__main__":
    sys.exit(main())
