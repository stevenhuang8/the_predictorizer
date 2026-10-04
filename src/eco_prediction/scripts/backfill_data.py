"""Backfill the full vintage history of every V1 series, resumably.

Usage:
    uv run python -m eco_prediction.scripts.backfill_data              # V1, from 1990
    uv run python -m eco_prediction.scripts.backfill_data CPIAUCSL     # just these
    uv run python -m eco_prediction.scripts.backfill_data --start 1980-01-01
    uv run python -m eco_prediction.scripts.backfill_data --restart    # ignore checkpoint
    uv run python -m eco_prediction.scripts.backfill_data --report     # coverage only

Each series is fetched with every ALFRED vintage for periods from `--start` on
(`ingest_series(full=True)`) and committed in its own transaction. After each
commit the series is recorded in a JSON checkpoint, so an interrupted run picks
up at the first unfinished series. The checkpoint belongs to one start/end
range; a run with a different range starts over. Failed series are not
checkpointed and are retried on the next run. Inserts skip rows already
stored, so re-running is always safe.

Rate limiting and retries are the FRED client's (120 requests/minute, backoff
on 429/5xx). Requesting the full real-time period returns every revision, so no
monthly snapshots are needed.

When the run ends, a coverage report shows per series the first and last
period, the first vintage (the earliest date point-in-time queries can see),
and the row and vintage counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from psycopg2.extensions import connection as Connection

from eco_prediction.data.fred_client import FREDAPIError, FREDClient
from eco_prediction.data.ingest import V1_SERIES, ingest_series
from eco_prediction.db.connection import ConfigError, connection

log = logging.getLogger("eco_prediction.backfill")

DEFAULT_START = date(1990, 1, 1)
DEFAULT_CHECKPOINT = Path("logs/backfill_checkpoint.json")
DEFAULT_LOG_FILE = Path("logs/backfill.log")

# A series whose first period is later than this after --start is flagged.
_COVERAGE_SLACK = timedelta(days=366)


class Checkpoint:
    """Series finished for one observation range, persisted as JSON."""

    def __init__(
        self, path: Path, start: date, end: date | None, *, restart: bool = False
    ) -> None:
        self.path = path
        self.range = {"start": start.isoformat(), "end": end and end.isoformat()}
        self.completed: dict[str, dict[str, Any]] = {}
        if restart or not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable checkpoint %s: %s", path, exc)
            return
        if data.get("range") != self.range:
            log.info(
                "Checkpoint %s is for range %s, not %s; starting over",
                path,
                data.get("range"),
                self.range,
            )
            return
        self.completed = dict(data.get("completed", {}))

    def is_done(self, series_id: str) -> bool:
        return series_id in self.completed

    def mark_done(self, series_id: str, **details: Any) -> None:
        self.completed[series_id] = {
            **details,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"range": self.range, "completed": self.completed}, indent=2)
        )
        tmp.replace(self.path)  # atomic, so an interrupt can't leave half a file


@dataclass(frozen=True)
class Coverage:
    series_id: str
    rows: int
    vintages: int
    first_period: date | None
    last_period: date | None
    first_vintage: date | None
    last_vintage: date | None


def backfill(
    client: FREDClient,
    series_ids: Sequence[str],
    checkpoint: Checkpoint,
    *,
    start: date,
    end: date | None = None,
) -> dict[str, Exception]:
    """Backfill each unfinished series. Returns the errors of those that failed."""
    failures: dict[str, Exception] = {}
    todo = [s for s in series_ids if not checkpoint.is_done(s)]
    skipped = len(series_ids) - len(todo)
    if skipped:
        log.info("Resuming: %d of %d series already done", skipped, len(series_ids))

    run_started = time.monotonic()
    for i, series_id in enumerate(todo, start=skipped + 1):
        log.info("[%d/%d] %s: fetching", i, len(series_ids), series_id)
        t0 = time.monotonic()
        try:
            with connection() as conn:
                result = ingest_series(
                    conn,
                    client,
                    series_id,
                    full=True,
                    observation_start=start,
                    observation_end=end,
                )
        except FREDAPIError as exc:
            log.error("[%d/%d] %s: FAILED %s", i, len(series_ids), series_id, exc)
            failures[series_id] = exc
            continue
        elapsed = time.monotonic() - t0
        checkpoint.mark_done(
            series_id,
            fetched=result.fetched,
            inserted=result.inserted,
            seconds=round(elapsed, 1),
        )
        log.info(
            "[%d/%d] %s: fetched=%d inserted=%d in %.1fs (run %.0fs)",
            i,
            len(series_ids),
            series_id,
            result.fetched,
            result.inserted,
            elapsed,
            time.monotonic() - run_started,
        )
    return failures


def coverage(conn: Connection, series_ids: Sequence[str]) -> list[Coverage]:
    """Stored rows per series, in the order given; unknown series have zero rows."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.series_id, count(o.id), count(DISTINCT o.as_of),
                   min(o.observed_at), max(o.observed_at), min(o.as_of), max(o.as_of)
            FROM series s
            LEFT JOIN observations o ON o.series_id = s.id
            WHERE s.series_id = ANY(%s)
            GROUP BY s.series_id
            """,
            (list(series_ids),),
        )
        found = {row[0]: Coverage(*row) for row in cur.fetchall()}
    return [found.get(s, Coverage(s, 0, 0, None, None, None, None)) for s in series_ids]


def format_report(rows: Sequence[Coverage], start: date) -> str:
    header = (
        f"{'series':15} {'rows':>7} {'vintages':>8}  "
        f"{'first period':12} {'last period':12} {'first vintage':13} last vintage"
    )
    lines = [header, "-" * len(header)]
    for c in rows:
        flag = ""
        if c.rows == 0:
            flag = "  <- no data"
        elif c.first_period and c.first_period > start + _COVERAGE_SLACK:
            flag = f"  <- starts after {start.year}"
        lines.append(
            f"{c.series_id:15} {c.rows:>7} {c.vintages:>8}  "
            f"{_d(c.first_period):12} {_d(c.last_period):12} "
            f"{_d(c.first_vintage):13} {_d(c.last_vintage)}{flag}"
        )
    total = sum(c.rows for c in rows)
    lines.append(f"total rows: {total}")
    return "\n".join(lines)


def _d(d: date | None) -> str:
    return d.isoformat() if d else "-"


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill FRED series and every vintage, resumably."
    )
    parser.add_argument(
        "series",
        nargs="*",
        default=list(V1_SERIES),
        help="FRED series IDs (default: V1 set)",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help=f"first period to load (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, help="last period to load (default: latest)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"checkpoint file (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--restart", action="store_true", help="ignore the checkpoint and redo all"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"also log here (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--report", action="store_true", help="only print the coverage report"
    )
    args = parser.parse_args(argv)
    setup_logging(args.log_file)

    failures: dict[str, Exception] = {}
    try:
        if not args.report:
            checkpoint = Checkpoint(
                args.checkpoint, args.start, args.end, restart=args.restart
            )
            with FREDClient.from_env() as client:
                failures = backfill(
                    client, args.series, checkpoint, start=args.start, end=args.end
                )
        with connection() as conn:
            rows = coverage(conn, args.series)
    except ConfigError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted; rerun to resume from the checkpoint")
        return 130

    print(format_report(rows, args.start))
    for series_id, error in failures.items():
        print(f"{series_id:15} FAILED  {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
