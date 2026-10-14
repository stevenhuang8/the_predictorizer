"""Load Epoch AI step series: benchmark SOTA and the largest training run.

Usage:
    uv run python -m eco_prediction.ai.ingest_epoch                     # live, all four
    uv run python -m eco_prediction.ai.ingest_epoch --backfill          # all four
    uv run python -m eco_prediction.ai.ingest_epoch --backfill --dry-run
    uv run python -m eco_prediction.ai.ingest_epoch gpqa --backfill     # one series

Four series, all monthly running maxima: a month's value is the best among
models released by that month's last day, as Epoch's data stands on the day
it's downloaded (see `EpochClient`).
- **EPOCH_GPQA_SOTA**, **EPOCH_SWEBENCH_SOTA**, **EPOCH_ARC2_SOTA**: best
  score so far on GPQA Diamond, SWE-bench Verified and ARC-AGI-2, in percent.
  Forecast target `benchmark_sota`. Each starts in the month its benchmark was
  published; the first month's value includes older models evaluated since.
- **EPOCH_LARGEST_FLOPS**: largest training compute so far among notable
  models, as log10(FLOP), from 2010-01. Forecast target `largest_training_run`.

Backfill and live rows follow `ai/monthly_ingest.py`. Every run downloads the
two files once (~5 MB) and shares them across the series, so a live pull
re-checks the last 12 ended months instead of 3: a model released months ago
but evaluated only now changes those months, and the change is stored as a
live vintage.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import partial

from eco_prediction.ai.monthly_ingest import MonthlySeries, run_cli
from eco_prediction.ai.months import MonthlyCount
from eco_prediction.ai.sources.epoch_client import (
    ARC_AGI_2,
    GPQA_DIAMOND,
    SWE_BENCH_VERIFIED,
    Benchmark,
    EpochClient,
    Result,
    running_max,
)
from eco_prediction.ai.store import AISeriesInfo

EPOCH_SOURCE = (
    "Epoch AI",
    "https://epoch.ai/data",
    "Epoch AI Benchmarking Hub and Notable AI Models datasets (CC BY 4.0)",
)
LIVE_MONTHS = 12


@dataclass(frozen=True)
class EpochSeries:
    series: MonthlySeries
    load: Callable[[EpochClient], list[Result]]


def _sota(series_id: str, benchmark: Benchmark, first_month: date) -> EpochSeries:
    return EpochSeries(
        MonthlySeries(
            source=EPOCH_SOURCE,
            info=AISeriesInfo(
                series_id=series_id,
                name=(
                    f"Best {benchmark.name} score among models released by "
                    "month end (Epoch AI)"
                ),
                units="Percent",
                frequency="Monthly",
            ),
            first_month=first_month,
            live_months=LIVE_MONTHS,
            allow_zero=True,
            value_format=".2f",
        ),
        lambda client: client.benchmark_results(benchmark),
    )


SERIES: dict[str, EpochSeries] = {
    "gpqa": _sota("EPOCH_GPQA_SOTA", GPQA_DIAMOND, date(2023, 11, 1)),
    "swe_bench": _sota("EPOCH_SWEBENCH_SOTA", SWE_BENCH_VERIFIED, date(2024, 8, 1)),
    "arc_agi_2": _sota("EPOCH_ARC2_SOTA", ARC_AGI_2, date(2025, 3, 1)),
    "compute": EpochSeries(
        MonthlySeries(
            source=EPOCH_SOURCE,
            info=AISeriesInfo(
                series_id="EPOCH_LARGEST_FLOPS",
                name=(
                    "Largest training compute among notable models published by "
                    "month end (Epoch AI)"
                ),
                units="log10(FLOP)",
                frequency="Monthly",
            ),
            first_month=date(2010, 1, 1),
            live_months=LIVE_MONTHS,
            value_format=".2f",
        ),
        lambda client: client.training_runs(),
    ),
}


class EpochCounter:
    """`MonthlyCounter` for one series over a client shared by all of them."""

    def __init__(self, client: EpochClient, series: EpochSeries) -> None:
        self.client = client
        self.series = series

    def count_months(self, months: Sequence[date]) -> Iterable[MonthlyCount]:
        return running_max(self.series.load(self.client), months)

    def close(self) -> None:
        pass  # `main` closes the shared client


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], EpochClient] = EpochClient.from_env,
    today: Callable[[], date] = date.today,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    keys = list(SERIES)
    if args[:1] and args[0] in SERIES:
        keys, args = [args[0]], args[1:]
    client = client_factory()
    try:
        status = 0
        for key in keys:
            series = SERIES[key]
            if len(keys) > 1 and "--dry-run" in args:
                print(f"== {series.series.info.series_id}", flush=True)
            status |= run_cli(
                args,
                series.series,
                description=(
                    "Load Epoch AI step series (benchmark SOTA, largest training "
                    f"run) into the database. Series: {', '.join(SERIES)}."
                ),
                client_factory=partial(EpochCounter, client, series),
                today=today,
            )
        return status
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
