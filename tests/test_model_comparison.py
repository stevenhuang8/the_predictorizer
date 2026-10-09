"""Tests for the model comparison report.

Runs on the synthetic unemployment store from test_lightgbm_model.py (MICH
leads unemployment by 5 months), 2020-2021, with ARIMA and ETS swapped for
cheap stand-ins so the whole pipeline runs in seconds. FOMC is left out.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from test_lightgbm_model import STORE

from eco_prediction.models.baselines import HistoricalMeanModel, RandomWalkModel
from eco_prediction.models.lightgbm_model import LightGBMModel
from eco_prediction.scripts import model_comparison as mc

SMALL_GRID: dict[str, list[Any]] = {
    "max_depth": [3],
    "learning_rate": [0.1],
    "min_child_samples": [10],
}
SETTINGS = mc.Settings(
    date(2020, 1, 1), date(2021, 12, 1), targets=("unemployment",), include_fomc=False
)


CHEAP_MODELS = {
    "lightgbm": lambda h, c: LightGBMModel(h, coverage=c, param_grid=SMALL_GRID),
    "arima": lambda h, c: RandomWalkModel(coverage=c),
    "ets": lambda h, c: HistoricalMeanModel(coverage=c),
}


@pytest.fixture
def cheap_models(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, build in CHEAP_MODELS.items():
        monkeypatch.setitem(mc.MODELS, name, build)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[mc.Runs]:
    """Every backtest, computed once for the module."""
    with pytest.MonkeyPatch.context() as patch:
        for name, build in CHEAP_MODELS.items():
            patch.setitem(mc.MODELS, name, build)
        yield mc.run_backtests(STORE, SETTINGS, tmp_path_factory.mktemp("cache"))


def test_runs_cover_every_model_horizon_and_curve_level(runs: mc.Runs) -> None:
    expected = {
        (m, "unemployment", h, c)
        for m in mc.MODELS
        for h in mc.HORIZONS
        for c in (mc.CURVE_COVERAGES if h == mc.CURVE_HORIZON else (mc.COVERAGE,))
    }
    assert set(runs.numeric) == expected
    assert all(len(r) == 24 for r in runs.numeric.values())
    # SHAP recorded for each of LightGBM's 3-month forecasts.
    assert len(runs.explanations["unemployment"]) == 24


def test_metrics_table_and_significance(runs: mc.Runs) -> None:
    table = mc.metrics_table(runs, SETTINGS)
    assert len(table) == len(mc.MODELS) * len(mc.HORIZONS)
    rw = table[table["model"] == "random_walk"]
    assert (rw["rmse_vs_random_walk"] == 1).all() and rw["dm_p"].isna().all()
    # The ARIMA stand-in is a random walk: identical errors, no test possible.
    assert table[table["model"] == "arima"]["dm_p"].isna().all()
    # With a real leading indicator LightGBM beats the random walk, significantly.
    lgbm = table[(table["model"] == "lightgbm") & (table["horizon"] == 3)].iloc[0]
    assert lgbm["rmse_vs_random_walk"] < 0.5
    assert lgbm["dm_stat"] < 0 and lgbm["dm_p"] < 0.01
    assert table["interval_score"].gt(0).all()


def test_shap_ranks_the_signal_first(runs: mc.Runs) -> None:
    shap = mc.shap_table(runs)
    top = shap[shap["target"] == "unemployment"].sort_values("rank")
    assert set(top["feature"].iloc[:2]) == {"mich", "unrate"}
    assert top["mean_abs_shap"].is_monotonic_decreasing


def test_calibration_and_by_year_tables(runs: mc.Runs) -> None:
    calibration = mc.calibration_table(runs, SETTINGS)
    assert len(calibration) == len(mc.MODELS) * len(mc.CURVE_COVERAGES)
    for _, rows in calibration.groupby("model"):
        # Wider nominal intervals never cover less.
        assert rows.sort_values("nominal")["empirical"].is_monotonic_increasing
    by_year = mc.coverage_by_year(runs, SETTINGS, 3)
    assert list(by_year.columns) == [2020, 2021]


def test_report_files_and_findings(runs: mc.Runs, tmp_path: Path) -> None:
    summary = mc.build_report(runs, SETTINGS, tmp_path)
    html = (tmp_path / "model_comparison.html").read_text()
    assert html.count("data:image/png;base64,") == 4
    assert "Success criterion" not in html  # no CPI in this run
    data = json.loads((tmp_path / "model_comparison.json").read_text())
    assert data["success_criterion"] is None
    assert len(data["metrics"]) == len(mc.MODELS) * len(mc.HORIZONS)
    assert data["settings"]["start"] == "2020-01-01"
    assert summary["findings"][1].startswith("Unemployment (%), 3-month: lowest RMSE")


def test_success_criterion() -> None:
    table = pd.DataFrame(
        [
            {"target": "cpi_yoy", "horizon": 3, "model": "lightgbm", "rmse": 1.2,
             "dm_p": 0.2},
            {"target": "cpi_yoy", "horizon": 3, "model": "random_walk", "rmse": 1.4,
             "dm_p": float("nan")},
        ]
    )  # fmt: skip
    criterion = mc.success_criterion(table)
    assert criterion is not None
    assert criterion["passed"] and not criterion["significant"]
    assert mc.success_criterion(table.assign(target="unemployment")) is None


def test_cached_runs_are_reused(
    cheap_models: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    first = mc.run_backtests(STORE, SETTINGS, cache)

    def fail(h: int, c: float) -> RandomWalkModel:
        raise AssertionError("a cached run was recomputed")

    for name in mc.MODELS:
        monkeypatch.setitem(mc.MODELS, name, fail)
    second = mc.run_backtests(STORE, SETTINGS, cache)
    for key, results in first.numeric.items():
        pd.testing.assert_frame_equal(results, second.numeric[key])
    with pytest.raises(AssertionError, match="recomputed"):
        mc.run_backtests(STORE, SETTINGS, cache, refresh=True)
