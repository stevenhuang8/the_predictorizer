"""Walk-forward comparison of every model, as an HTML report and JSON.

Usage:
    uv run python -m eco_prediction.scripts.model_comparison             # 2020-2023
    uv run python -m eco_prediction.scripts.model_comparison --refresh   # rerun backtests
    uv run python -m eco_prediction.scripts.model_comparison \\
        --start 2018-01-01 --end 2024-12-01 --out results/2018_2024

Every model in `MODELS` is backtested with `WalkForwardBacktest` on every
target and horizon (1, 3, 6 months), refit annually on all data published by
each refit (an expanding window), forecasting on the first of each month from
--start to --end. Intervals are 80%; at `CURVE_HORIZON` the models also run at
50% and 95% to draw interval calibration curves. LightGBM's 3-month forecasts
are explained with SHAP as they are made. The FOMC models (Task 21) are
backtested over the same window at a 7-day lead.

Outputs, in --out (default `results/`):
- `model_comparison.html`: tables and charts, self-contained (images inline);
- `model_comparison.json`: every number in the report;
- `cache/<start>_<end>/`: each backtest's results, reused on the next run
  unless --refresh (a full run takes about 10 minutes, a cached one seconds).

"Beats the random walk" is judged by RMSE and by a Diebold-Mariano test on
squared errors (`metrics.diebold_mariano`), since with ~48 overlapping
forecasts a lower RMSE alone can be noise.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import pickle
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eco_prediction.backtest import metrics
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.fomc import FOMCBacktest, summarize_fomc
from eco_prediction.backtest.harness import (
    Forecast,
    Forecaster,
    RetrainFrequency,
    WalkForwardBacktest,
    summarize,
)
from eco_prediction.data.fomc import RELEASE_LAGS
from eco_prediction.models.baselines import HistoricalMeanModel, RandomWalkModel
from eco_prediction.models.explainability import Explanation, SHAPExplainer
from eco_prediction.models.fomc_model import (
    AlwaysHold,
    Climatology,
    FOMCForecaster,
    FOMCModel,
    Persistence,
)
from eco_prediction.models.lightgbm_model import LightGBMModel
from eco_prediction.models.statistical import ARIMAModel, ETSModel

log = logging.getLogger("eco_prediction.model_comparison")

TARGETS = ("cpi_yoy", "unemployment")
HORIZONS = (1, 3, 6)
COVERAGE = 0.8
CURVE_COVERAGES = (0.5, 0.8, 0.95)
CURVE_HORIZON = 3
FOMC_LEAD_DAYS = 7

# Fixed order: each model keeps its color in every chart (the dataviz
# reference palette's categorical slots 1-5, light mode).
MODELS: dict[str, Callable[[int, float], Forecaster]] = {
    "lightgbm": lambda h, c: LightGBMModel(h, coverage=c),
    "random_walk": lambda h, c: RandomWalkModel(coverage=c),
    "arima": lambda h, c: ARIMAModel(coverage=c),
    "ets": lambda h, c: ETSModel(coverage=c),
    "historical_mean": lambda h, c: HistoricalMeanModel(coverage=c),
}
LABELS = {
    "lightgbm": "LightGBM",
    "random_walk": "Random walk",
    "arima": "ARIMA",
    "ets": "ETS",
    "historical_mean": "Historical mean",
    "cpi_yoy": "CPI YoY (%)",
    "unemployment": "Unemployment (%)",
}
COLORS = {
    "lightgbm": "#2a78d6",
    "random_walk": "#eb6834",
    "arima": "#1baf7a",
    "ets": "#eda100",
    "historical_mean": "#e87ba4",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

FOMC_MODELS: dict[str, Callable[[], FOMCModel]] = {
    "fomc_lightgbm": lambda: FOMCForecaster(FOMC_LEAD_DAYS),
    "persistence": lambda: Persistence(FOMC_LEAD_DAYS),
    "climatology": lambda: Climatology(FOMC_LEAD_DAYS),
    "always_hold": lambda: AlwaysHold(FOMC_LEAD_DAYS),
}


@dataclass(frozen=True)
class Settings:
    start: date
    end: date
    retrain_frequency: RetrainFrequency = "annually"
    horizons: tuple[int, ...] = HORIZONS
    targets: tuple[str, ...] = TARGETS
    include_fomc: bool = True

    @property
    def cache_name(self) -> str:
        return f"{self.start:%Y%m%d}_{self.end:%Y%m%d}_{self.retrain_frequency}"


class ExplainedLightGBM:
    """A LightGBM model that records a SHAP explanation of every forecast."""

    def __init__(self, model: LightGBMModel) -> None:
        self.model = model
        self.explainer: SHAPExplainer | None = None
        self.explanations: list[tuple[date, Explanation]] = []

    def fit(self, data: PointInTimeData) -> None:
        self.model.fit(data)
        self.explainer = SHAPExplainer(self.model)

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        assert self.explainer is not None
        forecast = self.model.predict(data, target_period)
        self.explanations.append(
            (target_period, self.explainer.explain(data, target_period))
        )
        return forecast


@dataclass
class Runs:
    """Every backtest's results, keyed (model, target, horizon, coverage)."""

    numeric: dict[tuple[str, str, int, float], pd.DataFrame] = field(
        default_factory=dict
    )
    explanations: dict[str, list[tuple[date, Explanation]]] = field(
        default_factory=dict
    )
    fomc: dict[str, pd.DataFrame] = field(default_factory=dict)


def _cached(path: Path, refresh: bool, compute: Callable[[], Any]) -> Any:
    if path.exists() and not refresh:
        with path.open("rb") as f:
            return pickle.load(f)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f)
    return value


def run_backtests(
    store: VintageStore, settings: Settings, cache_dir: Path, refresh: bool = False
) -> Runs:
    bt = WalkForwardBacktest(
        settings.start,
        settings.end,
        settings.retrain_frequency,
        store=store,
    )
    runs = Runs()
    jobs = [
        (model, target, h, c)
        for target in settings.targets
        for h in settings.horizons
        for c in (CURVE_COVERAGES if h == CURVE_HORIZON else (COVERAGE,))
        for model in MODELS
    ]
    for i, (model_name, target, h, c) in enumerate(jobs, 1):
        path = cache_dir / f"{target}_{model_name}_h{h}_c{round(c * 100)}.pkl"
        explain = model_name == "lightgbm" and h == CURVE_HORIZON and c == COVERAGE

        def compute(
            i: int = i,
            model_name: str = model_name,
            target: str = target,
            h: int = h,
            c: float = c,
            explain: bool = explain,
        ) -> dict[str, Any]:
            started = time.monotonic()
            model = MODELS[model_name](h, c)
            if explain:
                assert isinstance(model, LightGBMModel)
                wrapped = ExplainedLightGBM(model)
                results = bt.run(wrapped, target, h)
                explanations = wrapped.explanations
            else:
                results, explanations = bt.run(model, target, h), []
            log.info(
                "[%d/%d] %s %s h=%d %.0f%%: %.0fs",
                i,
                len(jobs),
                model_name,
                target,
                h,
                c * 100,
                time.monotonic() - started,
            )
            return {"results": results, "explanations": explanations}

        cached = _cached(path, refresh, compute)
        runs.numeric[(model_name, target, h, c)] = cached["results"]
        if explain:
            runs.explanations[target] = cached["explanations"]

    if settings.include_fomc:
        fomc_bt = FOMCBacktest(
            settings.start,
            settings.end,
            "annually",
            store=VintageStore.from_db(backfill=True, release_lags=RELEASE_LAGS),
        )
        for name, build in FOMC_MODELS.items():
            path = cache_dir / f"fomc_{name}_lead{FOMC_LEAD_DAYS}.pkl"

            def compute_fomc(
                build: Callable[[], FOMCModel] = build,
            ) -> pd.DataFrame:
                return fomc_bt.run(build())

            runs.fomc[name] = _cached(path, refresh, compute_fomc)
    return runs


# Tables


def _resolved(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["actual"].notna()]


def metrics_table(runs: Runs, settings: Settings) -> pd.DataFrame:
    """One row per (target, horizon, model) at 80% intervals."""
    rows = []
    for target in settings.targets:
        for h in settings.horizons:
            baseline = _resolved(runs.numeric[("random_walk", target, h, COVERAGE)])
            for model in MODELS:
                results = runs.numeric[(model, target, h, COVERAGE)]
                resolved = _resolved(results)
                summary = summarize(results)
                intervals = list(zip(resolved["lower"], resolved["upper"]))
                row: dict[str, Any] = {
                    "target": target,
                    "horizon": h,
                    "model": model,
                    "n": summary["n_resolved"],
                    "rmse": summary["rmse"],
                    "mae": summary["mae"],
                    "bias": summary["bias"],
                    "coverage": summary["interval_coverage"],
                    "interval_score": metrics.interval_score(
                        intervals, resolved["actual"], COVERAGE
                    ),
                }
                base_rmse = summarize(baseline)["rmse"]
                assert base_rmse is not None and row["rmse"] is not None
                row["rmse_vs_random_walk"] = row["rmse"] / base_rmse
                row["dm_stat"], row["dm_p"] = (
                    (math.nan, math.nan)
                    if model == "random_walk"
                    else _dm_vs(resolved, baseline, h)
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _dm_vs(
    results: pd.DataFrame, baseline: pd.DataFrame, h: int
) -> tuple[float, float]:
    joined = results.merge(
        baseline, on="forecast_date", suffixes=("", "_base"), validate="1:1"
    )
    try:
        return metrics.diebold_mariano(joined["error"], joined["error_base"], horizon=h)
    except ValueError:  # identical errors, or too few forecasts: no test
        return math.nan, math.nan


def coverage_by_year(runs: Runs, settings: Settings, h: int) -> pd.DataFrame:
    """80% interval coverage per forecast year, rows (target, model)."""
    frames = {}
    for target in settings.targets:
        for model in MODELS:
            resolved = _resolved(runs.numeric[(model, target, h, COVERAGE)])
            years = pd.to_datetime(resolved["forecast_date"]).dt.year
            frames[(target, model)] = (
                resolved["in_interval"].astype(float).groupby(years).mean()
            )
    table = pd.DataFrame(frames).T
    table.index.names = ["target", "model"]
    return table


def calibration_table(runs: Runs, settings: Settings) -> pd.DataFrame:
    """Empirical coverage at each nominal level, at CURVE_HORIZON."""
    rows = []
    for target in settings.targets:
        for model in MODELS:
            for c in CURVE_COVERAGES:
                key = (model, target, CURVE_HORIZON, c)
                if key in runs.numeric:
                    rows.append(
                        {
                            "target": target,
                            "model": model,
                            "nominal": c,
                            "empirical": summarize(runs.numeric[key])[
                                "interval_coverage"
                            ],
                        }
                    )
    return pd.DataFrame(rows)


def shap_table(runs: Runs, top: int = 10) -> pd.DataFrame:
    """Mean |SHAP contribution| per feature over LightGBM's 3-month forecasts."""
    rows = []
    for target, explanations in runs.explanations.items():
        contributions = pd.DataFrame([e.contributions for _, e in explanations])
        mean_abs = contributions.abs().mean().sort_values(ascending=False)
        for rank, (feature, value) in enumerate(mean_abs.head(top).items(), 1):
            rows.append(
                {
                    "target": target,
                    "rank": rank,
                    "feature": feature,
                    "mean_abs_shap": float(value),
                }
            )
    return pd.DataFrame(rows)


def fomc_table(runs: Runs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": name, **summarize_fomc(results)}
            for name, results in runs.fomc.items()
        ]
    )


# Charts


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _png(fig: Any) -> str:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _legend(fig: Any, ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def plot_rmse(table: pd.DataFrame, settings: Settings) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(settings.targets), figsize=(10, 3.8))
    for ax, target in zip(np.atleast_1d(axes), settings.targets):
        _style(ax)
        for model in MODELS:
            rows = table[(table["target"] == target) & (table["model"] == model)]
            ax.plot(
                rows["horizon"],
                rows["rmse"],
                color=COLORS[model],
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                solid_capstyle="round",
                label=LABELS[model],
            )
        ax.set_title(LABELS[target], fontsize=11, loc="left")
        ax.set_xticks(list(settings.horizons))
        ax.set_xlabel("horizon (months)")
        ax.set_ylabel("RMSE (percentage points)")
        ax.set_ylim(bottom=0)
    _legend(fig, np.atleast_1d(axes)[0])
    return _png(fig)


def plot_calibration(table: pd.DataFrame, settings: Settings) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(settings.targets), figsize=(10, 4.2))
    for ax, target in zip(np.atleast_1d(axes), settings.targets):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, label="perfect")
        for model in MODELS:
            rows = table[(table["target"] == target) & (table["model"] == model)]
            ax.plot(
                rows["nominal"],
                rows["empirical"],
                color=COLORS[model],
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                label=LABELS[model],
            )
        ax.set_xlim(0.4, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(
            f"{LABELS[target]}, {CURVE_HORIZON}-month", fontsize=11, loc="left"
        )
        ax.set_xlabel("nominal interval coverage")
        ax.set_ylabel("share of actuals inside")
    _legend(fig, np.atleast_1d(axes)[0])
    return _png(fig)


def plot_forecasts(runs: Runs, settings: Settings, h: int) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(settings.targets), 1, figsize=(10, 3.4 * len(settings.targets))
    )
    for ax, target in zip(np.atleast_1d(axes), settings.targets):
        _style(ax)
        base = runs.numeric[("random_walk", target, h, COVERAGE)]
        actual = base.dropna(subset=["actual"])
        ax.plot(
            pd.to_datetime(actual["target_period"]),
            actual["actual"],
            color=INK,
            linewidth=2,
            label="Actual (first release)",
        )
        for model in MODELS:
            results = runs.numeric[(model, target, h, COVERAGE)]
            ax.plot(
                pd.to_datetime(results["target_period"]),
                results["prediction"],
                color=COLORS[model],
                linewidth=1.5,
                label=LABELS[model],
            )
        ax.set_title(f"{LABELS[target]}: {h}-month forecasts", fontsize=11, loc="left")
        ax.set_ylabel("percent")
    _legend(fig, np.atleast_1d(axes)[0])
    fig.tight_layout()
    return _png(fig)


def plot_shap(table: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    targets = list(dict.fromkeys(table["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(10, 4))
    for ax, target in zip(np.atleast_1d(axes), targets):
        _style(ax)
        ax.grid(axis="y", visible=False)
        rows = table[table["target"] == target].iloc[::-1]
        ax.barh(
            rows["feature"],
            rows["mean_abs_shap"],
            height=0.6,
            color=COLORS["lightgbm"],
        )
        ax.set_title(f"{LABELS[target]}", fontsize=11, loc="left")
        ax.set_xlabel("mean |SHAP| (percentage points)")
    fig.tight_layout()
    return _png(fig)


# Report


def findings(table: pd.DataFrame, settings: Settings) -> list[str]:
    """Plain statements read off the metrics table."""
    out = []
    for target in settings.targets:
        for h in settings.horizons:
            rows = table[(table["target"] == target) & (table["horizon"] == h)]
            best = rows.sort_values("rmse").iloc[0]
            note = ""
            if best["model"] != "random_walk":
                note = (
                    f", {1 - best['rmse_vs_random_walk']:.0%} below the random walk"
                    f" (Diebold-Mariano p = {best['dm_p']:.2f})"
                )
            by_mae = rows.sort_values("mae").iloc[0]
            if by_mae["model"] != best["model"]:
                # A few huge misses (e.g. April 2020) can decide RMSE alone.
                note += (
                    f"; by MAE, {LABELS[by_mae['model']]} is lowest "
                    f"({by_mae['mae']:.2f} vs {best['mae']:.2f})"
                )
            out.append(
                f"{LABELS[target]}, {h}-month: lowest RMSE is "
                f"{LABELS[best['model']]} ({best['rmse']:.2f}){note}."
            )
    return out


def success_criterion(table: pd.DataFrame) -> dict[str, Any] | None:
    """The task's bar: LightGBM beats the random walk on 3-month CPI RMSE.

    None if the run didn't include 3-month CPI.
    """
    rows = table[(table["target"] == "cpi_yoy") & (table["horizon"] == 3)]
    if rows.empty:
        return None
    lgbm = rows[rows["model"] == "lightgbm"].iloc[0]
    rw = rows[rows["model"] == "random_walk"].iloc[0]
    return {
        "criterion": "LightGBM RMSE < random walk RMSE, CPI YoY, 3-month horizon",
        "lightgbm_rmse": float(lgbm["rmse"]),
        "random_walk_rmse": float(rw["rmse"]),
        "dm_p": float(lgbm["dm_p"]),
        "passed": bool(lgbm["rmse"] < rw["rmse"]),
        "significant": bool(lgbm["rmse"] < rw["rmse"] and lgbm["dm_p"] < 0.05),
    }


def _table_html(frame: pd.DataFrame, labels: bool = True) -> str:
    shown = frame.copy()
    if labels:
        for column in ("model", "target"):
            if column in shown:
                shown[column] = shown[column].map(lambda v: LABELS.get(v, v))
    return shown.to_html(
        index=False, float_format=lambda x: f"{x:.2f}", na_rep="–", border=0
    )


CSS = f"""
:root {{ color-scheme: light; }}
body {{ background: {SURFACE}; color: {INK}; margin: 0;
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2 {{ font-size: 19px; margin: 40px 0 8px; }}
p, li {{ color: {INK_SECONDARY}; max-width: 75ch; }}
.verdict {{ border: 1px solid {GRID}; border-radius: 8px; padding: 12px 16px;
  margin: 16px 0; background: #fff; }}
.verdict strong {{ color: {INK}; }}
table {{ border-collapse: collapse; font-size: 13px; margin: 8px 0 16px;
  font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid {GRID}; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
.scroll {{ overflow-x: auto; }}
img {{ max-width: 100%; height: auto; }}
"""


def render_html(
    settings: Settings,
    table: pd.DataFrame,
    by_year: pd.DataFrame,
    calibration: pd.DataFrame,
    shap: pd.DataFrame,
    fomc: pd.DataFrame,
    charts: dict[str, str],
    criterion: dict[str, Any] | None,
) -> str:
    def img(name: str, alt: str) -> str:
        return f'<img alt="{escape(alt)}" src="data:image/png;base64,{charts[name]}">'

    verdict_html = ""
    if criterion is not None:
        verdict = ("passed" if criterion["passed"] else "not met") + (
            " (significant)" if criterion["significant"] else ""
        )
        verdict_html = (
            "<div class='verdict'><strong>Success criterion "
            f"({escape(criterion['criterion'])}): {verdict}.</strong> "
            f"LightGBM {criterion['lightgbm_rmse']:.2f} vs random walk "
            f"{criterion['random_walk_rmse']:.2f}; Diebold-Mariano p = "
            f"{criterion['dm_p']:.2f}.</div>"
        )
    generated = datetime.now().astimezone().date()
    columns = [
        "target", "horizon", "model", "n", "rmse", "mae", "bias", "coverage",
        "interval_score", "rmse_vs_random_walk", "dm_p",
    ]  # fmt: skip
    by_year_shown = by_year.reset_index()
    by_year_shown.columns = [str(c) for c in by_year_shown.columns]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Model comparison</title><style>{CSS}</style></head><body><main>",
        "<h1>Model comparison</h1>",
        (
            f"<p>Walk-forward backtest, forecasts made monthly {settings.start} to "
            f"{settings.end}, refit {settings.retrain_frequency} on all data published "
            "by then (point-in-time vintages). Scored against first releases. "
            f"Intervals are {COVERAGE:.0%}. Generated {generated}.</p>"
        ),
        verdict_html,
        "<h2>Findings</h2><ul>",
        *(f"<li>{escape(f)}</li>" for f in findings(table, settings)),
        "</ul>",
        "<h2>Point accuracy by horizon</h2>",
        img("rmse", "RMSE by model and horizon"),
        "<div class='scroll'>",
        _table_html(table[columns]),
        (
            "</div><p>dm_p: Diebold-Mariano test of equal squared error against the "
            "random walk (small-sample corrected). interval_score: width plus 10x any "
            "miss distance, lower is better.</p>"
        ),
        f"<h2>Interval calibration ({CURVE_HORIZON}-month)</h2>",
        img("calibration", "Nominal vs empirical interval coverage"),
        "<div class='scroll'>",
        _table_html(calibration),
        "</div>",
        f"<h2>{COVERAGE:.0%} interval coverage by year ({CURVE_HORIZON}-month)</h2>",
        "<div class='scroll'>",
        _table_html(by_year_shown),
        "</div>",
        f"<h2>Forecasts vs actuals ({CURVE_HORIZON}-month)</h2>",
        img("forecasts", "Forecasts against actual values"),
    ]
    if not shap.empty:
        parts += [
            f"<h2>What drives LightGBM ({CURVE_HORIZON}-month)</h2>",
            (
                "<p>Mean absolute SHAP contribution over every forecast in the window, "
                "in percentage points of the forecast.</p>"
            ),
            img("shap", "LightGBM mean absolute SHAP by feature"),
            "<div class='scroll'>",
            _table_html(shap),
            "</div>",
        ]
    if not fomc.empty:
        parts += [
            f"<h2>FOMC decisions ({FOMC_LEAD_DAYS} days before each meeting)</h2>",
            (
                "<p>Brier score, lower is better (0 = all probability on the outcome, "
                "2 = all on a wrong one). brier_on_changes scores only the meetings "
                "that cut or hiked.</p>"
            ),
            "<div class='scroll'>",
            _table_html(fomc, labels=False),
            "</div>",
        ]
    parts.append("</main></body></html>")
    return "\n".join(parts)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def build_report(runs: Runs, settings: Settings, out_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    table = metrics_table(runs, settings)
    by_year = coverage_by_year(runs, settings, CURVE_HORIZON)
    calibration = calibration_table(runs, settings)
    shap = shap_table(runs)
    fomc = fomc_table(runs) if runs.fomc else pd.DataFrame()
    criterion = success_criterion(table)
    charts = {
        "rmse": plot_rmse(table, settings),
        "calibration": plot_calibration(calibration, settings),
        "forecasts": plot_forecasts(runs, settings, CURVE_HORIZON),
    }
    if not shap.empty:
        charts["shap"] = plot_shap(shap)

    out_dir.mkdir(parents=True, exist_ok=True)
    html = render_html(
        settings, table, by_year, calibration, shap, fomc, charts, criterion
    )
    (out_dir / "model_comparison.html").write_text(html)
    summary = {
        "settings": asdict(settings),
        "success_criterion": criterion,
        "findings": findings(table, settings),
        "metrics": table.to_dict(orient="records"),
        "coverage_by_year": [
            {str(k): v for k, v in record.items()}
            for record in by_year.reset_index().to_dict(orient="records")
        ],
        "calibration": calibration.to_dict(orient="records"),
        "shap_top_features": shap.to_dict(orient="records"),
        "fomc": fomc.to_dict(orient="records"),
    }
    (out_dir / "model_comparison.json").write_text(
        json.dumps(_json_safe(summary), indent=2)
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare every model's backtest.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2023, 12, 1))
    parser.add_argument(
        "--retrain", choices=["monthly", "quarterly", "annually"], default="annually"
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--refresh", action="store_true", help="ignore cached runs")
    parser.add_argument("--no-fomc", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr
    )

    settings = Settings(
        args.start, args.end, args.retrain, include_fomc=not args.no_fomc
    )
    store = VintageStore.from_db(backfill=True)
    runs = run_backtests(
        store, settings, args.out / "cache" / settings.cache_name, args.refresh
    )
    summary = build_report(runs, settings, args.out)
    criterion = summary["success_criterion"]
    print(f"Report: {args.out / 'model_comparison.html'}")
    if criterion is not None:
        print(
            f"Success criterion {'passed' if criterion['passed'] else 'NOT met'}: "
            f"LightGBM {criterion['lightgbm_rmse']:.2f} vs random walk "
            f"{criterion['random_walk_rmse']:.2f} (DM p = {criterion['dm_p']:.2f})"
        )
    for line in summary["findings"]:
        print(" -", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
