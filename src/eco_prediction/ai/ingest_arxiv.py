"""Load monthly arXiv AI submission counts (series ARXIV_AI_SUBMISSIONS).

Usage:
    uv run python -m eco_prediction.ai.ingest_arxiv               # live pull
    uv run python -m eco_prediction.ai.ingest_arxiv --backfill    # 2010 to last month
    uv run python -m eco_prediction.ai.ingest_arxiv --backfill --dry-run

The value for a month is the number of papers first submitted that month in
cs.AI, cs.LG or cs.CL (see `ArxivClient`). Backfill and live rows follow
`ai/monthly_ingest.py`. A backfill is one request per month, ~17 minutes.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from datetime import date

from eco_prediction.ai.monthly_ingest import MonthlyCounter, MonthlySeries, run_cli
from eco_prediction.ai.sources.arxiv_client import AI_CATEGORIES, ArxivClient
from eco_prediction.ai.store import AISeriesInfo

SERIES = MonthlySeries(
    source=(
        "arXiv",
        "https://arxiv.org",
        "arXiv preprint server (export.arxiv.org search API)",
    ),
    info=AISeriesInfo(
        series_id="ARXIV_AI_SUBMISSIONS",
        name=(
            "arXiv submissions in " + ", ".join(AI_CATEGORIES) + " (primary or "
            "cross-listed, each paper once), by month of first submission"
        ),
        units="Papers",
        frequency="Monthly",
    ),
    first_month=date(2010, 1, 1),
)


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], MonthlyCounter] = ArxivClient.from_env,
    today: Callable[[], date] = date.today,
) -> int:
    return run_cli(
        argv,
        SERIES,
        description="Load monthly arXiv AI submission counts into the database.",
        client_factory=client_factory,
        today=today,
    )


if __name__ == "__main__":
    sys.exit(main())
