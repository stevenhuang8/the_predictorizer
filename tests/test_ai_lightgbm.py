"""Tests for LightGBM on the AI series (AI task 15).

The main synthetic store has growth regimes: HF's monthly log growth is 1% or
6%, switching at random each year, and arXiv's growth copies HF's from six
months earlier, so HF's recent growth tells a model what arXiv will do.
Each month is known on the 1st of the next.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from eco_prediction.ai.baselines import (
    AIRandomWalk,
    BinaryData,
    BinaryQuestion,
    NoForecast,
    ThresholdBaseRate,
    evaluate_binary,
)
from eco_prediction.ai.lightgbm_model import (
    AILightGBMModel,
    AIThresholdClassifier,
    is_trending,
    pool_for,
    relative_step,
    row_offset,
)
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.targets import AI_TARGETS
from eco_prediction.backtest.data import LookaheadError, PointInTimeData, VintageStore
from eco_prediction.backtest.harness import WalkForwardBacktest, summarize
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.explainability import SHAPExplainer

ARXIV = "ARXIV_AI_SUBMISSIONS"
HF = "HF_NEW_MODELS"
GPQA = "EPOCH_GPQA_SOTA"
SWEBENCH = "EPOCH_SWEBENCH_SOTA"
ARC2 = "EPOCH_ARC2_SOTA"
START = date(2014, 1, 1)
N = 144  # through 2025-12
CUTOFF = date(2026, 1, 14)  # knows through 2025-12
SMALL_GRID = {"max_depth": [3], "learning_rate": [0.1], "min_child_samples": [5, 10]}


def monthly(start: date, values: list[float]) -> pd.DataFrame:
    months = list(iter_months(start, date(2100, 1, 1)))[: len(values)]
    return pd.DataFrame(
        [(m, next_month(m), v) for m, v in zip(months, values)],
        columns=["observed_at", "as_of", "value"],
    )


def regimes(n: int = N, seed: int = 0) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    yearly = rng.choice([0.01, 0.06], size=n // 12 + 2)
    hf_growth = np.repeat(yearly, 12)[: n + 6]
    hf = 500 * np.exp(np.cumsum(hf_growth[6:]))
    arxiv = 1000 * np.exp(np.cumsum(hf_growth[:n]))  # HF's growth, 6 months late
    noise = np.exp(rng.normal(0, 0.005, n))
    return list(arxiv * noise), list(hf)


def regime_store() -> VintageStore:
    arxiv, hf = regimes()
    return VintageStore({ARXIV: monthly(START, arxiv), HF: monthly(START, hf)})


STORE = regime_store()


def view(store: VintageStore, series_id: str, cutoff: date = CUTOFF) -> PointInTimeData:
    return PointInTimeData(store, AI_TARGETS[series_id], cutoff)


def model(horizon: int = 3, **kwargs: object) -> AILightGBMModel:
    return AILightGBMModel(horizon, param_grid=SMALL_GRID, **kwargs)  # type: ignore[arg-type]


def saturating(v0: float, rate: float, n: int) -> list[float]:
    """A running max closing `rate` of the gap to 100 every other month."""
    out, v = [], v0
    for i in range(n):
        if i % 2:
            v += rate * (100 - v)
        out.append(round(v, 1))
    return out


def sota_store() -> VintageStore:
    start = date(2023, 11, 1)  # 26 months through 2025-12
    return VintageStore(
        {
            GPQA: monthly(start, saturating(40, 0.08, 26)),
            SWEBENCH: monthly(start, saturating(20, 0.06, 26)),
            ARC2: monthly(start, saturating(5, 0.04, 26)),
        }
    )


# --- numeric -----------------------------------------------------------------------


def test_counts_are_forecast_as_log_changes() -> None:
    # Exactly 2% a month: every log-change label is 4 * log(1.02), so the
    # forecast is the last value grown 4 months (known through 2025-12,
    # forecasting 2026-04 from 2026-01), with no spread.
    values = [1000 * 1.02**i for i in range(N)]
    data = view(VintageStore({ARXIV: monthly(START, values)}), ARXIV)
    m = model()
    m.fit(data)
    f = m.predict(data, date(2026, 4, 1))
    assert f.point == pytest.approx(values[-1] * 1.02**4, rel=1e-6)
    assert f.lower == pytest.approx(f.point, rel=1e-6)
    assert f.upper == pytest.approx(f.point, rel=1e-6)
    assert data.max_as_of_seen == date(2026, 1, 1)  # reads went through the view


def test_rows_are_dated_on_the_cutoffs_day_of_the_month() -> None:
    assert row_offset(date(2025, 12, 31)) == timedelta(0)
    assert row_offset(date(2026, 1, 14)) == timedelta(days=14)
    m = model(trending=True)  # keep the levels, to see which month a row knows
    m.fit(view(STORE, ARXIV, date(2025, 12, 31)))
    X, _ = m.training_frame(view(STORE, ARXIV, date(2025, 12, 31)))
    assert (X.index.is_month_end).all()  # the economy's rule
    arxiv = regimes()[0]
    nov_2024, dec_2024 = arxiv[130], arxiv[131]
    # On Dec 31 a forecast made in January knows November.
    assert X.loc[pd.Timestamp(2024, 12, 31), "target"] == pytest.approx(nov_2024)
    m.fit(view(STORE, ARXIV))
    X, _ = m.training_frame(view(STORE, ARXIV))
    assert (X.index.day == 14).all()
    # On Jan 14 it knows December, as a live run would.
    assert X.loc[pd.Timestamp(2025, 1, 14), "target"] == pytest.approx(dec_2024)


def test_trending_features_are_left_out() -> None:
    for name in ("arxiv", "hf_log", "github_lag12", "flops_lag3", "trend_months",
                 "log_ratio_hf_arxiv"):  # fmt: skip
        assert is_trending(name)
    for name in ("arxiv_growth_3m", "flops_chg_12m", "gpqa", "gpqa_lag1",
                 "corr12_hf_arxiv", "month_3", "target"):  # fmt: skip
        assert not is_trending(name)
    assert is_trending("target_lag1", "count") and not is_trending("target", "percent")
    m = model()
    m.fit(view(STORE, ARXIV))
    assert not any(is_trending(name, "count") for name in m.feature_names)
    assert "target_growth_1m" in m.feature_names


def test_tuning_picks_from_the_grid_and_uses_cross_series_features() -> None:
    m = model()
    m.fit(view(STORE, ARXIV))
    assert m.best_params["min_child_samples"] in (5, 10)
    assert m.best_params["n_estimators"] >= 1
    assert m.cv_rmse is not None and math.isfinite(m.cv_rmse)
    assert "target_growth_3m" in m.feature_names
    assert any(name.startswith("hf_") for name in m.feature_names)
    assert m.feature_importance().index[0].startswith(("hf", "log_ratio_hf"))


def test_beats_the_random_walk_in_a_walk_forward_backtest() -> None:
    bt = WalkForwardBacktest(
        date(2022, 1, 1), date(2025, 6, 1), "quarterly", store=STORE
    )
    target = AI_TARGETS[ARXIV]
    lgbm = summarize(bt.run(model(), target, horizon_months=3))
    walk = summarize(bt.run(AIRandomWalk(), target, horizon_months=3))
    assert lgbm["n_resolved"] == walk["n_resolved"] == 42
    assert lgbm["rmse"] < 0.5 * walk["rmse"]  # type: ignore[operator]
    assert lgbm["mae"] < walk["mae"]  # type: ignore[operator]


def test_shap_explains_on_the_log_scale() -> None:
    data = view(STORE, ARXIV)
    m = model()
    m.fit(data)
    f = m.predict(data, date(2026, 4, 1))
    explanation = SHAPExplainer(m).explain(data, date(2026, 4, 1))
    assert explanation.offset == pytest.approx(math.log(regimes()[0][-1]))
    assert math.exp(explanation.prediction) == pytest.approx(f.point, rel=1e-6)
    assert len(explanation.contributions) == len(m.feature_names)


def test_benchmarks_are_too_short_alone_but_fit_pooled() -> None:
    store = sota_store()
    with pytest.raises(ValueError, match="labelled months"):
        model().fit(view(store, SWEBENCH))
    m = model(pool=True)
    data = view(store, SWEBENCH)
    m.fit(data)
    assert m.series == pool_for(SWEBENCH) == [SWEBENCH, GPQA, ARC2]
    assert {"series_swebench", "series_gpqa", "series_arc2"} <= set(m.feature_names)
    last = float(data.target_history().iloc[-1])
    f = m.predict(data, date(2026, 4, 1))
    assert last <= f.lower <= f.point <= f.upper <= 100  # type: ignore[operator]


def test_pooled_rows_are_one_per_series_and_month_in_date_order() -> None:
    m = model(pool=True)
    m.fit(view(sota_store(), GPQA))
    X, y = m.training_frame(view(sota_store(), GPQA))
    assert X.index.is_monotonic_increasing
    assert not X.reset_index().duplicated(["as_of", "series_gpqa", "series_arc2"]).any()
    # Labels are changes in log headroom: closing part of the gap is negative.
    assert (y <= 1e-12).all() and (y < 0).any()


def test_errors() -> None:
    with pytest.raises(NotFittedError):
        model().predict(view(STORE, ARXIV), date(2026, 4, 1))
    m = model()
    m.fit(view(STORE, ARXIV))
    with pytest.raises(ValueError, match="wasn't fit"):
        m.predict(view(STORE, HF), date(2026, 4, 1))
    with pytest.raises(ValueError, match="3 months ahead"):
        m.predict(view(STORE, ARXIV), date(2026, 6, 1))


# --- threshold questions ----------------------------------------------------------


def question(
    series_id: str, threshold: float, ask: date, horizon: int = 1
) -> BinaryQuestion:
    target = date(ask.year + (ask.month - 1 + horizon) // 12,
                  (ask.month - 1 + horizon) % 12 + 1, 1)  # fmt: skip
    return BinaryQuestion(
        None, f"{series_id}>={threshold}", target, ask, horizon_months=horizon
    )


def classifier(**kwargs: object) -> AIThresholdClassifier:
    return AIThresholdClassifier(param_grid=SMALL_GRID, **kwargs)  # type: ignore[arg-type]


def test_relative_steps() -> None:
    assert relative_step("count", 100, 110) == pytest.approx(math.log(1.1))
    assert relative_step("percent", 60, 70) == pytest.approx(0.25)
    assert relative_step("percent", 100, 100) == 1.0
    assert relative_step("log10", 26.0, 26.3) == pytest.approx(0.3)


def test_training_rows_use_only_outcomes_published_by_the_cutoff() -> None:
    cutoff = date(2024, 1, 14)
    X, y, published = classifier().training_frame(BinaryData(STORE), cutoff, "count")
    assert published.max() <= pd.Timestamp(cutoff)
    assert X.index.max() <= pd.Timestamp(cutoff)
    assert set(X["horizon_months"]) == {1.0, 3.0, 6.0}
    assert (
        X["step"].round(6).isin([round(math.log(m), 6) for m in (1.1, 1.25, 1.5)]).all()
    )
    assert set(y) == {0, 1}


def test_outcomes_that_all_agree_give_their_smoothed_rate() -> None:
    # 2% a month never clears +10% in 2 months, nor +25%/+50% in 7.
    values = [1000 * 1.02**i for i in range(N)]
    data = BinaryData(VintageStore({ARXIV: monthly(START, values)}))
    clf = classifier(horizons=(1,)).fit(data, CUTOFF)
    _, y, _ = clf.training_frame(data, CUTOFF, "count")
    p = clf.predict_probability(
        question(ARXIV, values[-1] * 1.1, date(2026, 1, 15)), data
    )
    assert y.sum() == 0
    assert p["yes"] == pytest.approx(1 / (len(y) + 2))
    assert set(clf.skipped) == {"log10", "percent"}  # no data for those kinds


def test_classifier_beats_the_base_rate_on_regimes() -> None:
    data = BinaryData(STORE)
    clf = classifier().fit(data, date(2020, 12, 31))
    assert "count" in clf.cv_log_loss and clf.models["count"].calibrator is not None
    arxiv = pd.Series(regimes()[0], index=pd.date_range(START, periods=N, freq="MS"))
    # Asked mid-month from 2021-01 to 2024-06, two slow years then two fast:
    # will arXiv grow 10% in 2 months?
    questions = []
    for ask in pd.date_range("2021-01-15", "2024-06-15", freq="MS") + pd.Timedelta(
        days=14
    ):
        base = arxiv[ask.to_period("M").to_timestamp() - pd.DateOffset(months=1)]
        q = question(ARXIV, float(base) * 1.1, ask.date())
        outcome = arxiv[pd.Timestamp(q.target_date)] >= float(base) * 1.1
        questions.append(
            BinaryQuestion(
                None,
                q.question_key,
                q.target_date,
                q.ask_date,
                1,
                outcome="yes" if outcome else "no",
            )
        )
    ours = evaluate_binary(clf, questions, data)
    base = evaluate_binary(ThresholdBaseRate(), questions, data)
    assert set(ours["outcome"]) == {"yes", "no"}
    assert ours["p_yes"].between(0, 1).all()
    assert ours["brier"].mean() < 0.5 * base["brier"].mean()


def test_classifier_refuses_what_it_cannot_answer() -> None:
    data = BinaryData(STORE)
    with pytest.raises(NotFittedError):
        classifier().predict_probability(question(ARXIV, 1e9, date(2024, 3, 15)), data)
    clf = classifier(horizons=(1,)).fit(data, date(2024, 1, 14))
    with pytest.raises(LookaheadError):  # asked before the model's data ends
        clf.predict_probability(question(ARXIV, 1e9, date(2023, 6, 15)), data)
    curated = BinaryQuestion(None, "pm:1", date(2024, 6, 1), date(2024, 3, 1),
                             platform_question_id=1, title="x")  # fmt: skip
    with pytest.raises(NoForecast):
        clf.predict_probability(curated, data)
    with pytest.raises(NoForecast):  # no SOTA data, so no percent model
        clf.predict_probability(question(GPQA, 90, date(2024, 3, 15)), data)
