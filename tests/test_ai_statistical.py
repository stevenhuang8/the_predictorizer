"""Tests for ARIMA/ETS on the AI series (AI task 14).

Count series are synthetic arXiv-like data: 3% monthly growth from 1,000 in
2018-01 with a December dip and seeded noise, each month known on the 1st of
the next. Step series are made by hand.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from eco_prediction.ai.baselines import AIRandomWalk
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.statistical_models import AIARIMA, AIETS, scale_for
from eco_prediction.ai.targets import AI_TARGETS
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import WalkForwardBacktest, summarize
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.statistical import ARIMAModel

ARXIV = "ARXIV_AI_SUBMISSIONS"
GPQA = "EPOCH_GPQA_SOTA"
FLOPS = "EPOCH_LARGEST_FLOPS"
START = date(2018, 1, 1)
CUTOFF = date(2026, 1, 14)  # knows through 2025-12


def monthly(start: date, values: list[float]) -> pd.DataFrame:
    months = list(iter_months(start, date(2100, 1, 1)))[: len(values)]
    return pd.DataFrame(
        [(m, next_month(m), v) for m, v in zip(months, values)],
        columns=["observed_at", "as_of", "value"],
    )


def arxiv_like(n: int = 96, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    months = np.arange(n)
    december = np.where(months % 12 == 11, -0.08, 0.0)
    log = math.log(1000) + 0.03 * months + december + rng.normal(0, 0.02, n)
    return list(np.exp(log))


def store(**series: list[float] | tuple[date, list[float]]) -> VintageStore:
    frames = {}
    for series_id, spec in series.items():
        start, values = spec if isinstance(spec, tuple) else (START, spec)
        frames[series_id] = monthly(start, values)
    return VintageStore(frames)


def view(s: VintageStore, series_id: str, cutoff: date = CUTOFF) -> PointInTimeData:
    return PointInTimeData(s, AI_TARGETS[series_id], cutoff)


# --- scales -----------------------------------------------------------------


def test_count_scale_is_log_with_an_offset_only_for_zeros() -> None:
    s = pd.Series([10.0, 100.0])
    log = scale_for("count", s)
    assert log.forward(s).tolist() == pytest.approx([math.log(10), math.log(100)])
    assert log.inverse(math.log(100)) == pytest.approx(100)
    with_zero = scale_for("count", pd.Series([0.0, 5.0]))
    assert with_zero.forward(pd.Series([0.0])).iloc[0] == 0.0  # log(0 + 1)
    assert with_zero.inverse(0.0) == pytest.approx(0.0)


def test_percent_scale_is_log_headroom() -> None:
    pct = scale_for("percent", pd.Series([50.0]))
    assert pct.decreasing
    assert pct.forward(pd.Series([90.0])).iloc[0] == pytest.approx(math.log(10.1))
    assert math.isfinite(pct.forward(pd.Series([100.0])).iloc[0])
    assert pct.inverse(math.log(10.1)) == pytest.approx(90.0)
    assert pct.inverse(-50.0) <= 100.1  # never far past 100


def test_log10_scale_is_identity() -> None:
    s = scale_for("log10", pd.Series([26.0]))
    assert s.forward(pd.Series([26.5])).iloc[0] == 26.5 and s.inverse(27.0) == 27.0


# --- counts on the log scale ----------------------------------------------------


def test_arima_forecasts_counts_on_the_log_scale() -> None:
    values = arxiv_like()
    data = view(store(**{ARXIV: values}), ARXIV)
    model = AIARIMA()
    model.fit(data)
    f = model.predict(data, date(2026, 3, 1))
    assert f.lower < f.point < f.upper  # type: ignore[operator]
    # The economy's ARIMA run directly on log values gives the same forecast.
    logs = store(**{ARXIV: list(np.log(values))})
    raw = ARIMAModel(season_length=12)
    raw.fit(view(logs, ARXIV))
    r = raw.predict(view(logs, ARXIV), date(2026, 3, 1))
    assert f.point == pytest.approx(math.exp(r.point))
    # A normal interval on logs is symmetric in ratio terms.
    assert math.log(f.upper / f.point) == pytest.approx(  # type: ignore[operator]
        math.log(f.point / f.lower),
        rel=1e-6,  # type: ignore[operator]
    )
    assert data.max_as_of_seen == date(2026, 1, 1)  # reads went through the view


def test_ets_fits_and_predicts_counts() -> None:
    data = view(store(**{ARXIV: arxiv_like()}), ARXIV)
    model = AIETS()
    model.fit(data)
    f = model.predict(data, date(2026, 6, 1))
    last = arxiv_like()[-1]
    assert last < f.point < last * 1.5  # still growing, ~3% a month


def test_a_zero_month_is_handled() -> None:
    values = arxiv_like()
    values[10] = 0.0
    data = view(store(**{ARXIV: values}), ARXIV)
    model = AIARIMA()
    model.fit(data)
    assert model.scale is not None and model.scale.name == "log(y + 1)"
    assert math.isfinite(model.predict(data, date(2026, 3, 1)).point)


# --- step series ----------------------------------------------------------------


def test_percent_forecasts_stay_between_the_last_value_and_100() -> None:
    # Headroom halves every six months: 60 -> 80 -> 90 -> ... -> 99.06.
    values = [100 - 40 * 0.5 ** (i / 6) for i in range(40)]
    data = view(store(**{GPQA: (date(2022, 9, 1), values)}), GPQA)
    for model in (AIARIMA(), AIETS()):
        model.fit(data)
        f = model.predict(data, date(2026, 7, 1))
        assert values[-1] <= f.lower <= f.point <= f.upper <= 100  # type: ignore[operator]


def test_log10_forecasts_are_floored_at_the_record() -> None:
    values = [25.0 + 0.05 * (i // 6) for i in range(36)]  # a step up every 6 months
    data = view(store(**{FLOPS: (date(2023, 1, 1), values)}), FLOPS)
    model = AIARIMA()
    model.fit(data)
    f = model.predict(data, date(2026, 6, 1))
    assert f.lower >= values[-1]  # type: ignore[operator]


def test_short_histories_and_unfitted_models_raise() -> None:
    short = view(store(**{GPQA: (date(2024, 7, 1), [50.0] * 18)}), GPQA)
    with pytest.raises(ValueError, match="at least 24 months"):
        AIARIMA().fit(short)
    with pytest.raises(NotFittedError):
        AIETS().predict(short, date(2026, 3, 1))


# --- against the random walk -------------------------------------------------------


def test_arima_beats_the_random_walk_on_growing_counts() -> None:
    s = store(**{ARXIV: arxiv_like()})
    bt = WalkForwardBacktest(date(2023, 1, 1), date(2025, 6, 1), "quarterly", store=s)
    target = AI_TARGETS[ARXIV]
    arima = summarize(bt.run(AIARIMA(), target, horizon_months=3))
    walk = summarize(bt.run(AIRandomWalk(), target, horizon_months=3))
    assert arima["n_resolved"] == walk["n_resolved"] == 30
    assert arima["rmse"] < 0.5 * walk["rmse"]  # type: ignore[operator]
    assert arima["interval_coverage"] >= 0.6  # type: ignore[operator]
