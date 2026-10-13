"""Load monthly new public model counts from the Hugging Face Hub (HF_NEW_MODELS).

Usage:
    uv run python -m eco_prediction.ai.ingest_huggingface               # live pull
    uv run python -m eco_prediction.ai.ingest_huggingface --backfill    # 2022-04 on
    uv run python -m eco_prediction.ai.ingest_huggingface --backfill --dry-run

The value for a month is the number of models created that month (UTC) that
are public and still exist (see `HuggingFaceClient`). Backfill and live rows
follow `ai/monthly_ingest.py`. History starts in April 2022, the first month
`createdAt` can count. A backfill lists every model since then, newest first,
and commits each month as the listing passes it; a live pull lists back
through the last 3 ended months.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from datetime import date

from eco_prediction.ai.monthly_ingest import MonthlyCounter, MonthlySeries, run_cli
from eco_prediction.ai.sources.huggingface_client import FIRST_MONTH, HuggingFaceClient
from eco_prediction.ai.store import AISeriesInfo

SERIES = MonthlySeries(
    source=(
        "Hugging Face Hub",
        "https://huggingface.co",
        "Hugging Face Hub model listing (huggingface.co/api/models)",
    ),
    info=AISeriesInfo(
        series_id="HF_NEW_MODELS",
        name=(
            "New models on the Hugging Face Hub by month created "
            "(public and not deleted as of the count)"
        ),
        units="Models",
        frequency="Monthly",
    ),
    first_month=FIRST_MONTH,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], MonthlyCounter] = HuggingFaceClient.from_env,
    today: Callable[[], date] = date.today,
) -> int:
    return run_cli(
        argv,
        SERIES,
        description="Load monthly Hugging Face new-model counts into the database.",
        client_factory=client_factory,
        today=today,
    )


if __name__ == "__main__":
    sys.exit(main())
