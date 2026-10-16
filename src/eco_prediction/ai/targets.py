"""The AI numeric series as backtest targets (AI task 13).

    from eco_prediction.ai.targets import AI_TARGETS, kind_of

    bt.run(AIRandomWalk(), AI_TARGETS["HF_NEW_MODELS"], horizon_months=3)
    kind_of("EPOCH_SWEBENCH_SOTA")   # 'percent'

Each target is the series' level (no transform), so the economy's
`WalkForwardBacktest` runs on it unchanged. A target is named by its series
id, not its `forecast_target`: the three benchmarks share `benchmark_sota`.

Kinds decide how models treat a series:
- `count` (arXiv, HF, GitHub): multiplicative growth, modelled on a log scale;
- `percent` (benchmark SOTA): a running maximum, between its last value and 100;
- `log10` (largest training run, log10 FLOP): a running maximum, unbounded.

The harness resolves a backtest forecast with the period's first stored
vintage. Over history that's the reconstructed backfill, while live questions
resolve to the first live value (`ai/resolutions.py`); the two agree once
live pulls have run.
"""

from __future__ import annotations

from typing import Literal

from eco_prediction.ai.questions import NUMERIC_TARGETS
from eco_prediction.backtest.data import Target, _level

Kind = Literal["count", "percent", "log10"]

_KINDS: dict[str, Kind] = {"Percent": "percent", "log10(FLOP)": "log10"}

KINDS: dict[str, Kind] = {
    t.series.series_id: _KINDS.get(t.series.units, "count") for t in NUMERIC_TARGETS
}

AI_TARGETS: dict[str, Target] = {
    t.series.series_id: Target(
        t.series.series_id, t.series.series_id, _level, t.series.name
    )
    for t in NUMERIC_TARGETS
}


def kind_of(series_id: str) -> Kind:
    if series_id not in KINDS:
        raise ValueError(f"{series_id!r} is not an AI numeric target")
    return KINDS[series_id]
