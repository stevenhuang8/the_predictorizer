"""Client for Epoch AI's datasets: benchmark scores and notable models' compute.

Usage:
    from eco_prediction.ai.sources.epoch_client import GPQA_DIAMOND, EpochClient

    with EpochClient.from_env() as epoch:
        results = epoch.benchmark_results(GPQA_DIAMOND)
        months = running_max(results, [date(2026, 8, 1), date(2026, 9, 1)])

Epoch has no API, only static downloads that it refreshes as it adds models:
- `benchmark_data.zip`: one CSV per benchmark, one row per model run, with the
  model's release date. GPQA Diamond and SWE-bench Verified are Epoch's own
  runs; ARC-AGI-2 is copied from the ARC Prize leaderboard. The score column
  is each benchmark's headline metric, as `benchmark_metadata.csv` names it.
- `notable_ai_models.csv`: one row per notable model, with its publication
  date and estimated training compute in FLOP (blank when Epoch has none).
  Estimates of every confidence level are used, "Speculative" included,
  because frontier labs rarely disclose compute.

Each file is downloaded once per client and cached. Rows without a date or a
value are skipped; a value or date that doesn't parse, or a missing column,
raises `SourceAPIError`, since that means Epoch changed its format.

The values are a model's own: `running_max` turns them into the step series
"best so far", by the month each model was released.
"""

from __future__ import annotations

import csv
import io
import math
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from eco_prediction.ai.months import MonthlyCount, next_month
from eco_prediction.ai.sources.base_client import BaseAPIClient, SourceAPIError

BASE_URL = "https://epoch.ai/data"
BENCHMARK_ZIP = "benchmark_data.zip"
MODELS_CSV = "notable_ai_models.csv"


@dataclass(frozen=True)
class Benchmark:
    name: str
    file: str  # CSV inside benchmark_data.zip
    score_column: str  # headline metric, a fraction from 0 to 1


GPQA_DIAMOND = Benchmark(
    "GPQA Diamond", "gpqa_diamond.csv", "Best score (across scorers)"
)
SWE_BENCH_VERIFIED = Benchmark(
    "SWE-bench Verified", "swe_bench_verified.csv", "Best score (across scorers)"
)
ARC_AGI_2 = Benchmark("ARC-AGI-2", "arc_agi_2_external.csv", "Score")


@dataclass(frozen=True)
class Result:
    released: date
    value: float
    model: str


class EpochClient(BaseAPIClient):
    SOURCE = "epoch"
    BASE_URL = BASE_URL
    MAX_CALLS = 10  # two static files; this only guards against a loop
    PERIOD = 60.0

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("timeout", 120.0)
        super().__init__(**kwargs)
        self._files: dict[str, bytes] = {}

    def benchmark_results(self, benchmark: Benchmark) -> list[Result]:
        """Every run of `benchmark` with a release date, as a percent (0 to 100)."""
        try:
            with zipfile.ZipFile(io.BytesIO(self._download(BENCHMARK_ZIP))) as zf:
                text = zf.read(benchmark.file).decode("utf-8-sig")
        except (zipfile.BadZipFile, KeyError) as exc:
            raise SourceAPIError(
                f"{BENCHMARK_ZIP}: can't read {benchmark.file} ({exc})",
                source=self.SOURCE,
            ) from None
        return [
            Result(r.released, r.value * 100, r.model)
            for r in _parse(
                text,
                benchmark.file,
                model="Model version",
                released="Release date",
                value=benchmark.score_column,
            )
        ]

    def training_runs(self) -> list[Result]:
        """Every notable model with a compute estimate, as log10(FLOP)."""
        text = self._download(MODELS_CSV).decode("utf-8-sig")
        results = []
        for r in _parse(
            text,
            MODELS_CSV,
            model="Model",
            released="Publication date",
            value="Training compute (FLOP)",
        ):
            if r.value <= 0:
                raise SourceAPIError(
                    f"{MODELS_CSV}: {r.model!r} has training compute {r.value}",
                    source=self.SOURCE,
                )
            results.append(Result(r.released, math.log10(r.value), r.model))
        return results

    def _download(self, name: str) -> bytes:
        if name not in self._files:
            self._files[name] = self._get(name).content
        return self._files[name]


def _parse(
    text: str, file: str, *, model: str, released: str, value: str
) -> list[Result]:
    reader = csv.DictReader(io.StringIO(text))
    missing = {model, released, value} - set(reader.fieldnames or ())
    if missing:
        raise SourceAPIError(
            f"{file}: missing column(s) {sorted(missing)}", source="epoch"
        )
    results = []
    for row in reader:
        if not row[released].strip() or not row[value].strip():
            continue
        try:
            results.append(
                Result(
                    date.fromisoformat(row[released].strip()),
                    float(row[value]),
                    row[model],
                )
            )
        except ValueError as exc:
            raise SourceAPIError(
                f"{file}: bad row for {row[model]!r} ({exc})", source="epoch"
            ) from None
    if not results:
        raise SourceAPIError(f"{file}: no usable rows", source="epoch")
    return results


def running_max(
    results: Iterable[Result], months: Sequence[date]
) -> list[MonthlyCount]:
    """For each month, the best value among models released by its last day,
    rounded to 4 decimals. Months before the first release are left out."""
    ordered = sorted(results, key=lambda r: r.released)
    out = []
    i, best = 0, -math.inf
    for month in sorted(months):
        end = next_month(month)
        while i < len(ordered) and ordered[i].released < end:
            best = max(best, ordered[i].value)
            i += 1
        if i:
            out.append(MonthlyCount(month, round(best, 4)))
    return out
