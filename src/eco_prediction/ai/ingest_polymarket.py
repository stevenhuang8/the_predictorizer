"""Mirror AI-tagged Polymarket markets and their daily probabilities.

Usage:
    uv run python -m eco_prediction.ai.ingest_polymarket                 # daily run
    uv run python -m eco_prediction.ai.ingest_polymarket --dry-run       # list markets only
    uv run python -m eco_prediction.ai.ingest_polymarket --no-history    # skip history
    uv run python -m eco_prediction.ai.ingest_polymarket --min-volume 50000

One command does everything, so it can run daily:
1. Lists every binary market in an AI-tagged event, open or closed, with at
   least `--min-volume` USD traded (default $10k), and upserts each into
   `platform_questions`: question, resolution criteria, dates and outcome.
2. Fills price history (`reconstructed = true`): one price a UTC day, from
   the market's opening through yesterday, or through its close day if it
   has closed.
3. Stores today's Yes price for each open market (`reconstructed = false`).

`prices_through` is the day through which a market's prices are stored with
no gaps, from history or live. History is only fetched when that's behind
yesterday (or the close day), so the first run fetches every market's history
(~2,500 requests, ~7 minutes) and a daily run after that only fetches new
markets, newly closed ones, and any gap left by a missed day.

A day already stored is never overwritten, so a live price beats the history
for the same day. Each market is committed on its own, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from psycopg2.extensions import connection as Connection

from eco_prediction.ai.platform_store import (
    PlatformProbability,
    PlatformQuestion,
    insert_platform_probabilities,
    set_prices_through,
    upsert_platform_question,
)
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.sources.polymarket_client import (
    Market,
    PolymarketClient,
    daily_prices,
)
from eco_prediction.db.connection import ConfigError, connection

PLATFORM = "polymarket"
MIN_VOLUME = 10_000.0


@dataclass
class IngestResult:
    markets: int = 0
    new_markets: int = 0
    live_rows: int = 0
    histories: int = 0  # markets whose history was fetched
    history_rows: int = 0


def to_question(m: Market) -> PlatformQuestion:
    return PlatformQuestion(
        platform=PLATFORM,
        external_id=m.id,
        title=m.question,
        resolution_criteria=m.description or None,
        url=m.url,
        opened_at=m.opened_at,
        closes_at=m.closes_at,
        resolved_at=m.closed_at if m.outcome else None,
        resolved_outcome=m.outcome,
        metadata=m.metadata,
    )


def history_end(m: Market, today: date) -> date:
    """The last day whose price history should be stored: yesterday, or the
    close day if the market closed before then."""
    end = today - timedelta(days=1)
    if m.closed and m.closed_at:
        end = min(end, m.closed_at.date())
    return end


def ingest(
    conn: Connection,
    client: PolymarketClient,
    today: date,
    *,
    min_volume: float = MIN_VOLUME,
    history: bool = True,
    on_market: Callable[[Market, int], None] | None = None,
) -> IngestResult:
    """Run steps 1-3 above, committing after each market. `on_market(market,
    history_rows)` is called after each commit, for progress output."""
    result = IngestResult()
    for m in client.ai_markets(min_volume):
        pk, through, new = upsert_platform_question(conn, to_question(m))
        result.markets += 1
        result.new_markets += new
        end = history_end(m, today)
        inserted = 0
        if history and (through is None or through < end):
            prices = daily_prices(
                client.price_history(m.yes_token), end + timedelta(days=1)
            )
            inserted = insert_platform_probabilities(
                conn,
                pk,
                (PlatformProbability(day, p, True) for day, p in prices.items()),
            )
            through = end
            set_prices_through(conn, pk, through)
            result.histories += 1
            result.history_rows += inserted
        if m.yes_price is not None:
            result.live_rows += insert_platform_probabilities(
                conn, pk, [PlatformProbability(today, m.yes_price, False)]
            )
            # Today's live price extends a gapless run, so tomorrow needs no history.
            if through == today - timedelta(days=1):
                set_prices_through(conn, pk, today)
        conn.commit()
        if on_market:
            on_market(m, inserted)
    return result


def format_market(m: Market) -> str:
    if m.outcome:
        state = f"resolved {m.outcome}"
    elif m.closed:
        state = "closed"
    else:
        state = f"open {m.yes_price:.3f}" if m.yes_price is not None else "open"
    closes = f"{m.closes_at:%Y-%m-%d}" if m.closes_at else "?"
    return f"{m.id:>8}  {closes}  {state:<14}  ${m.volume:>12,.0f}  {m.question}"


def _print_history(m: Market, inserted: int) -> None:
    if inserted:
        print(f"{format_market(m)}  history+{inserted}", flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], PolymarketClient] = PolymarketClient.from_env,
    today: Callable[[], date] = date.today,
) -> int:
    parser = argparse.ArgumentParser(
        description="Mirror AI-tagged Polymarket markets and daily probabilities."
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=MIN_VOLUME,
        help=f"minimum USD traded (default {MIN_VOLUME:,.0f})",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="store markets and today's prices, but fetch no price history",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the markets without fetching history or writing anything",
    )
    args = parser.parse_args(argv)
    if args.min_volume < 0:
        parser.error("--min-volume can't be negative")

    run_date = today()
    try:
        client = client_factory()
        try:
            if args.dry_run:
                markets = client.ai_markets(args.min_volume)
                for m in markets:
                    print(format_market(m), flush=True)
                print(f"{len(markets)} markets")
                return 0
            with connection() as conn:
                result = ingest(
                    conn,
                    client,
                    run_date,
                    min_volume=args.min_volume,
                    history=not args.no_history,
                    on_market=_print_history,
                )
        finally:
            client.close()
    except (ConfigError, SourceAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"polymarket markets={result.markets} new={result.new_markets} "
        f"live_rows={result.live_rows} histories={result.histories} "
        f"history_rows={result.history_rows}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
