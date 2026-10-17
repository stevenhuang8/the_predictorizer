"""The LLM comparison report, as HTML and JSON (AI task 20).

Usage:
    uv run python -m eco_prediction.ai.llm.comparison_report          # from results/llm_backtest/
    uv run python -m eco_prediction.ai.llm.comparison_report --results path/ --out results/

    from eco_prediction.ai.llm.comparison_report import build_report
    build_report(load_results(), Path("results"))

Reads the saved runs of `ai/llm/backtest.py` (one per model; the backtest
CLI saves them) and never calls the API itself. Writes:
- `llm_comparison.html`: self-contained (charts inline), styled like the
  economy's `model_comparison.html`;
- `llm_comparison.json`: every number in the report, for the dashboard.

Sections:
1. **Binary questions**: Brier of the LLM, the Polymarket price, the
   category base rate and the ensemble, on the questions all four answered;
   skill against the market and the base rate; and a 95% interval for each
   difference from a bootstrap that resamples markets, not questions,
   because legs of one event and the three leads of one market (90, 30 and
   7 days) aren't independent. Also by category and by lead.
2. **Numeric targets**: relative RMSE (forecast / actual - 1, so series pool)
   by horizon, for the LLM, random walk, ARIMA, LightGBM and the ensemble.
3. **Cost**: calls, spend and cost per forecast per model, and cost against
   Brier when more than one model has been run (Opus against Haiku).
4. **Calibration**: a reliability diagram of the LLM's and the market's
   probabilities.
5. **Sample rationales**: the LLM's most confident hits and misses.
6. **Key phrases**: the commonest words and word pairs in the rationales.
7. **Caveats**: the training cutoff and what was dropped for it, the market
   price in the prompt, clustering, and reconstructed data.

The verdict is the PRD's criterion: the LLM beats the base rate after its
training cutoff. "Significant" means the bootstrap interval of the
difference lies below zero.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eco_prediction.ai.llm.backtest import RESULTS_DIR, LLMBacktestResult, load_results
from eco_prediction.backtest import metrics
from eco_prediction.scripts.model_comparison import (
    CSS,
    GRID,
    INK_SECONDARY,
    SURFACE,
    _json_safe,
    _legend,
    _png,
    _style,
)

# Each component keeps its color in every chart. Statistical models keep the
# colors they have in model_comparison.html; the rest take the reference
# palette's remaining slots (both sets validated with the dataviz validator:
# lightness, chroma, CVD and normal-vision separation pass; yellow, magenta
# and aqua are under 3:1 on the surface, so every chart has a legend and a
# table of the same numbers).
COLORS = {
    "llm": "#6250d6",  # violet, slot 7
    "market": "#eda100",  # yellow, slot 4
    "base_rate": "#e87ba4",  # magenta, slot 5
    "ensemble": "#008300",  # green, slot 6
    "random_walk": "#eb6834",  # orange, slot 2
    "arima": "#1baf7a",  # aqua, slot 3
    "lightgbm": "#2a78d6",  # blue, slot 1
}
LABELS = {
    "llm": "LLM",
    "market": "Polymarket price",
    "base_rate": "Base rate",
    "ensemble": "Ensemble",
    "random_walk": "Random walk",
    "arima": "ARIMA",
    "lightgbm": "LightGBM",
}
BINARY_ORDER = ("llm", "market", "base_rate", "ensemble")
NUMERIC_ORDER = ("llm", "random_walk", "arima", "lightgbm", "ensemble")
BOOTSTRAP_SAMPLES = 2000
SAMPLE_RATIONALES = 5
KEY_PHRASES = 15
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "their",
        "they",
        "them",
        "we",
        "our",
        "i",
        "you",
        "he",
        "she",
        "his",
        "her",
        "not",
        "no",
        "yes",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "than",
        "then",
        "so",
        "such",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "into",
        "over",
        "under",
        "about",
        "above",
        "below",
        "up",
        "down",
        "out",
        "more",
        "most",
        "less",
        "least",
        "very",
        "much",
        "many",
        "some",
        "any",
        "all",
        "each",
        "both",
        "few",
        "other",
        "same",
        "only",
        "own",
        "just",
        "also",
        "still",
        "even",
        "yet",
        "before",
        "after",
        "while",
        "until",
        "since",
        "because",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "done",
        "per",
        "via",
        "vs",
    ]
)


# --- tables --------------------------------------------------------------------------


def _common_binary(result: LLMBacktestResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(p_yes per component, wide; per-question metadata) on the questions every
    component answered, the ensemble included."""
    b = result.binary
    wide = b.pivot_table(
        index="question_id", columns="component", values="p_yes", aggfunc="first"
    )
    if len(result.ensemble_binary):
        wide["ensemble"] = result.ensemble_binary.set_index("question_id")["p_yes"]
    wide = wide[[c for c in BINARY_ORDER if c in wide.columns]].dropna()
    meta_cols = [
        c for c in ("outcome", "category", "market_id", "lead_days", "title") if c in b
    ]
    meta = b.groupby("question_id")[meta_cols].first().reindex(wide.index)
    return wide, meta


def _brier(p: pd.Series, outcome: pd.Series) -> pd.Series:
    return (p - (outcome == "yes").astype(float)) ** 2


def cluster_bootstrap(
    diff: pd.Series,
    clusters: pd.Series,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval of the mean of `diff`, resampling whole clusters."""
    groups = [g.to_numpy() for _, g in diff.groupby(clusters.to_numpy())]
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    sums = np.array([g.sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    picks = rng.integers(0, len(groups), size=(samples, len(groups)))
    means = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def binary_table(result: LLMBacktestResult) -> pd.DataFrame:
    """Brier per component, skill and the bootstrap interval of its difference."""
    wide, meta = _common_binary(result)
    if wide.empty:
        return pd.DataFrame()
    clusters = (
        meta["market_id"]
        if "market_id" in meta
        else pd.Series(wide.index, index=wide.index)
    )
    briers = {c: _brier(wide[c], meta["outcome"]) for c in wide.columns}
    rows = []
    for c, b in briers.items():
        row: dict[str, Any] = {"model": c, "n": len(b), "brier": b.mean()}
        for ref in ("market", "base_rate"):
            if ref in briers and c != ref:
                diff = b - briers[ref]
                low, high = cluster_bootstrap(diff, clusters)
                row[f"skill_vs_{ref}"] = metrics.brier_skill_score(
                    b.mean(), briers[ref].mean()
                )
                row[f"diff_vs_{ref}"] = diff.mean()
                row[f"ci_vs_{ref}"] = f"[{low:+.3f}, {high:+.3f}]"
        rows.append(row)
    table = pd.DataFrame(rows)
    table["n_markets"] = int(clusters.nunique())
    return table


def binary_by(result: LLMBacktestResult, column: str) -> pd.DataFrame:
    """Brier per component within each value of `column` (category, lead_days)."""
    wide, meta = _common_binary(result)
    if wide.empty or column not in meta:
        return pd.DataFrame()
    rows = []
    for value, idx in meta.groupby(column).groups.items():
        row: dict[str, Any] = {column: value, "n": len(idx)}
        for c in wide.columns:
            row[c] = _brier(wide.loc[idx, c], meta.loc[idx, "outcome"]).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(column).reset_index(drop=True)


def numeric_table(result: LLMBacktestResult) -> pd.DataFrame:
    """Relative RMSE per horizon and component, on forecasts every component made."""
    n = result.numeric
    if n.empty:
        return pd.DataFrame()
    key = ["series_id", "forecast_date", "target_period", "horizon_months"]
    wide = n.pivot_table(
        index=key, columns="component", values="prediction", aggfunc="first"
    )
    actual = n.groupby(key)["actual"].first()
    if len(result.ensemble_numeric):
        ens = result.ensemble_numeric.set_index(
            ["series_id", "forecast_date", "target_period"]
        )
        wide["ensemble"] = [ens["prediction"].get(k[:3]) for k in wide.index]
    wide = wide[[c for c in NUMERIC_ORDER if c in wide.columns]].dropna()
    rel = wide.div(actual.reindex(wide.index), axis=0) - 1
    rows = []
    for h, part in rel.groupby(level="horizon_months"):
        horizon = int(h)  # type: ignore[call-overload]  # a scalar level value
        for c in part.columns:
            rmse = float((part[c] ** 2).mean() ** 0.5)
            rw = (
                float((part["random_walk"] ** 2).mean() ** 0.5)
                if "random_walk" in part
                else None
            )
            rows.append({
                "horizon": horizon, "model": c, "n": len(part), "relative_rmse": rmse,
                "vs_random_walk": rmse / rw if rw else None,
            })  # fmt: skip
    return pd.DataFrame(rows)


COST_COLUMNS = [
    "model", "new_calls", "reused_calls", "spent_usd", "usd_per_call", "failures",
    "llm_brier", "n_binary",
]  # fmt: skip


def cost_table(results: Sequence[LLMBacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        cost = r.summary.get("cost", {})
        llm_brier = r.summary.get("binary", {}).get("llm")
        rows.append({
            "model": r.model, "new_calls": cost.get("new_calls", 0),
            "reused_calls": cost.get("reused_calls", 0), "spent_usd": cost.get("spent_usd", 0.0),
            "usd_per_call": cost.get("per_call_usd"), "failures": len(cost.get("failures", [])),
            "llm_brier": llm_brier, "n_binary": r.summary.get("binary", {}).get("n_common"),
        })  # fmt: skip
    return pd.DataFrame(rows, columns=COST_COLUMNS)


def calibration(result: LLMBacktestResult, bins: int = 10) -> dict[str, pd.DataFrame]:
    wide, meta = _common_binary(result)
    y = (meta["outcome"] == "yes").astype(float)
    return {
        c: metrics.calibration_curve(wide[c], y, n_bins=bins)
        for c in ("llm", "market")
        if c in wide.columns
    }


def sample_rationales(
    result: LLMBacktestResult, k: int = SAMPLE_RATIONALES
) -> pd.DataFrame:
    """The LLM's k most confident hits and k most confident misses."""
    b = result.binary
    if b.empty or "rationale" not in b:
        return pd.DataFrame()
    llm = b[b["component"] == "llm"].copy()
    market = b[b["component"] == "market"].set_index("question_id")["p_yes"]
    llm["market"] = llm["question_id"].map(market)
    llm["error"] = (llm["p_yes"] - (llm["outcome"] == "yes")).abs()
    llm["confidence"] = (llm["p_yes"] - 0.5).abs()
    hits = llm[llm["error"] < 0.5].nlargest(k, "confidence").assign(kind="hit")
    misses = llm[llm["error"] >= 0.5].nlargest(k, "confidence").assign(kind="miss")
    cols = ["kind", "title", "ask_date", "p_yes", "market", "outcome", "rationale"]
    both = pd.concat([hits, misses])
    return both[[c for c in cols if c in both]].reset_index(drop=True)


def key_phrases(result: LLMBacktestResult, top: int = KEY_PHRASES) -> pd.DataFrame:
    """The commonest word pairs, then the commonest words, in the rationales
    (stopwords out), `top` in all. Ranked apart: a word is always at least as
    common as any pair it's in, so one list would never show a pair."""
    b = result.binary
    texts: list[str] = (
        b.loc[b["component"] == "llm", "rationale"].dropna().tolist()
        if "rationale" in b
        else []
    )
    singles: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    for text in texts:
        words = [
            w
            for w in re.findall(r"[a-z][a-z0-9\-]+", text.lower())
            if w not in STOPWORDS
        ]
        singles.update(words)
        pairs.update(f"{a} {b_}" for a, b_ in itertools.pairwise(words))
    n_pairs = (top + 1) // 2
    ranked = [(p, c, "pair") for p, c in pairs.most_common(n_pairs) if c > 1]
    ranked += [(w, c, "word") for w, c in singles.most_common(top - len(ranked))]
    return pd.DataFrame(ranked, columns=["phrase", "count", "kind"])


def verdict(result: LLMBacktestResult) -> dict[str, Any] | None:
    """The PRD's criterion: the LLM beats the base rate after its training cutoff."""
    table = binary_table(result)
    if (
        table.empty
        or "llm" not in set(table["model"])
        or "diff_vs_base_rate" not in table
    ):
        return None
    row = table[table["model"] == "llm"].iloc[0]
    low, high = (float(x) for x in row["ci_vs_base_rate"].strip("[]").split(", "))
    return {
        "criterion": "LLM Brier below the base rate's after the training cutoff",
        "model": result.model,
        "passed": bool(row["diff_vs_base_rate"] < 0),
        "significant": bool(high < 0),
        "llm_brier": float(row["brier"]),
        "base_rate_brier": float(row["brier"] - row["diff_vs_base_rate"]),
        "ci": [low, high],
        "n": int(row["n"]),
        "n_markets": int(row["n_markets"]),
    }


# --- charts ---------------------------------------------------------------------------


def plot_brier(table: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    _style(ax)
    for i, (model, brier) in enumerate(zip(table["model"], table["brier"])):
        ax.barh(i, brier, height=0.55, color=COLORS[model], label=LABELS[model])
        ax.text(
            brier, i, f"  {brier:.3f}", va="center", fontsize=9, color=INK_SECONDARY
        )
    ax.set_yticks(range(len(table)), [LABELS[m] for m in table["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_xlim(0, max(table["brier"]) * 1.25)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def plot_calibration(curves: dict[str, pd.DataFrame]) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.6))
    _style(ax)
    ax.plot(
        [0, 1],
        [0, 1],
        color=GRID,
        linewidth=2,
        linestyle="--",
        label="Perfect calibration",
    )
    for c, curve in curves.items():
        ax.plot(curve["mean_predicted"], curve["observed_frequency"], color=COLORS[c],
                linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=LABELS[c])  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    _legend(fig, ax)
    return _png(fig)


def plot_numeric(table: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.4))
    _style(ax)
    horizons = sorted(table["horizon"].unique())
    models = [m for m in NUMERIC_ORDER if m in set(table["model"])]
    width = 0.8 / len(models)
    for j, m in enumerate(models):
        rows = table[table["model"] == m].set_index("horizon").reindex(horizons)
        x = np.arange(len(horizons)) + (j - (len(models) - 1) / 2) * width
        heights = rows["relative_rmse"] * 100
        ax.bar(x, heights, width=width * 0.9, color=COLORS[m], label=LABELS[m])
        for xi, h in zip(x, heights):  # a label even where the bar is too short to see
            if not np.isnan(h):
                ax.text(xi, h, f"{h:.1f}", ha="center", va="bottom", fontsize=8,
                        color=INK_SECONDARY)  # fmt: skip
    ax.set_xticks(
        range(len(horizons)), [f"{h} month" + ("s" if h > 1 else "") for h in horizons]
    )
    ax.set_ylabel("relative RMSE (%)")
    ax.set_ylim(bottom=0)
    _legend(fig, ax)
    return _png(fig)


def plot_cost(table: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    _style(ax)
    points = table.dropna(subset=["usd_per_call", "llm_brier"])
    ax.scatter(points["usd_per_call"], points["llm_brier"], s=80, color=COLORS["llm"],
               edgecolors=SURFACE, linewidths=2, zorder=3)  # fmt: skip
    for _, row in points.iterrows():
        ax.annotate(f"  {row['model']}", (row["usd_per_call"], row["llm_brier"]),
                    fontsize=9, color=INK_SECONDARY, va="center")  # fmt: skip
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    low, high = points["usd_per_call"].min(), points["usd_per_call"].max()
    ax.set_xlim(low / 2, high * 4)  # room for the labels on the right
    ax.set_xlabel("cost per forecast (USD, log scale)")
    ax.set_ylabel("LLM Brier score")
    return _png(fig)


def plot_phrases(phrases: pd.DataFrame) -> str:
    """The word pairs (the table also lists single words); words if no pairs."""
    import matplotlib.pyplot as plt

    pairs = phrases[phrases["kind"] == "pair"]
    phrases = pairs if len(pairs) else phrases
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(phrases) + 0.8))
    _style(ax)
    ax.barh(range(len(phrases)), phrases["count"], height=0.6, color=COLORS["llm"])
    ax.set_yticks(range(len(phrases)), phrases["phrase"])
    ax.invert_yaxis()
    ax.set_xlabel("occurrences in the rationales")
    ax.grid(axis="y", visible=False)
    return _png(fig)


# --- the report ------------------------------------------------------------------------


def _html_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    if "model" in shown:
        shown["model"] = shown["model"].map(lambda v: LABELS.get(v, v))
    shown.columns = [LABELS.get(str(c), str(c)) for c in shown.columns]
    html = shown.to_html(index=False, float_format=lambda x: f"{x:.3f}", na_rep="–",
                         border=0, escape=True)  # fmt: skip
    return f"<div class='scroll'>{html}</div>"


def _img(b64: str, alt: str) -> str:
    return f'<img alt="{escape(alt)}" src="data:image/png;base64,{b64}">'


def render_html(sections: dict[str, Any], charts: dict[str, str]) -> str:
    primary = sections["primary"]
    generated = datetime.now().astimezone().date()
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>LLM comparison</title><style>{CSS}</style></head><body><main>",
        "<h1>LLM comparison</h1>",
    ]
    if primary is None:
        parts += [
            (
                "<p>No LLM backtest results yet. Run "
                "<code>uv run python -m eco_prediction.ai.llm.backtest --budget N</code>, "
                "then build this report again.</p>"
            ),
            "</main></body></html>",
        ]
        return "\n".join(parts)

    parts.append(
        f"<p>{escape(primary.model)}, forecasts made from {primary.cutoff} (its training "
        f"data cutoff) on. Generated {generated}.</p>"
    )
    v = sections["verdict"]
    if v is not None:
        state = ("passed" if v["passed"] else "not met") + (
            " (significant)" if v["significant"] else ""
        )
        parts.append(
            f"<div class='verdict'><strong>Success criterion ({escape(v['criterion'])}): "
            f"{state}.</strong> LLM {v['llm_brier']:.3f} vs base rate "
            f"{v['base_rate_brier']:.3f} on {v['n']} questions from {v['n_markets']} markets; "
            f"95% interval of the difference [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}].</div>"
        )

    parts.append("<h2>Binary questions</h2>")
    if sections["binary"].empty:
        parts.append("<p>No binary questions were forecast.</p>")
    else:
        parts += [
            ("<p>Brier score on the curated Polymarket questions every model answered. "
            "Skill is 1 - Brier / the reference's (0 matches it, above 0 beats it). "
            "Intervals resample markets, since a market's leads and an event's legs "
            "aren't independent.</p>"),
            _img(charts["brier"], "Brier score by model"),
            _html_table(sections["binary"]),
            "<h3>By category</h3>", _html_table(sections["by_category"]),
            "<h3>By lead (days before the market closes)</h3>", _html_table(sections["by_lead"]),
        ]  # fmt: skip

    parts.append("<h2>Numeric targets</h2>")
    if sections["numeric"].empty:
        parts.append("<p>No numeric target has resolved in the window yet.</p>")
    else:
        parts += [
            (
                "<p>Relative RMSE: the root mean square of forecast / actual - 1, pooled over "
                "the series. vs random walk below 1 beats it. Very few forecasts have resolved "
                "after the cutoff, so treat these as indicative.</p>"
            ),
            _img(charts["numeric"], "Relative RMSE by horizon"),
            _html_table(sections["numeric"]),
        ]

    parts += ["<h2>Cost</h2>", _html_table(sections["cost"])]
    if "cost" in charts:
        parts.append(_img(charts["cost"], "Cost per forecast against Brier"))
    else:
        parts.append(
            "<p>One model run so far; a cost-against-accuracy chart needs two.</p>"
        )

    if "calibration" in charts:
        parts += [
            "<h2>Calibration</h2>",
            (
                "<p>Questions binned by predicted probability: a calibrated forecaster's points "
                "lie on the diagonal.</p>"
            ),
            _img(charts["calibration"], "Reliability diagram"),
            *(
                f"<h3>{LABELS[c]}</h3>" + _html_table(t)
                for c, t in sections["calibration"].items()
            ),
        ]
    if not sections["rationales"].empty:
        parts += [
            "<h2>Sample rationales</h2>",
            "<p>The LLM's most confident hits and misses.</p>",
            _html_table(sections["rationales"]),
        ]
    if not sections["phrases"].empty:
        parts += [
            "<h2>Key phrases in the rationales</h2>",
            "<p>The commonest word pairs (charted) and single words.</p>",
            _img(charts["phrases"], "Commonest phrases"),
            _html_table(sections["phrases"]),
        ]
    parts += [
        "<h2>Caveats</h2><ul>",
        *(f"<li>{escape(c)}</li>" for c in sections["caveats"]),
        "</ul></main></body></html>",
    ]
    return "\n".join(parts)


def caveats(result: LLMBacktestResult) -> list[str]:
    return [
        (
            f"Only forecasts made on or after {result.cutoff}, {result.model}'s training data "
            f"cutoff, are scored; {result.dropped_before_cutoff} earlier questions were dropped. "
            "The model could have read how earlier questions came out."
        ),
        (
            "The prompt shows the market's price, so the LLM can lean on it: the LLM-vs-market "
            "comparison measures what it adds to the price, not independent skill."
        ),
        (
            "Polymarket questions cluster (legs of one event, three leads per market); the "
            "intervals resample markets for that reason."
        ),
        (
            "Numeric actuals and features over the window are backfilled values, reconstructed "
            "with some hindsight; live pulls replace them going forward."
        ),
        "Metaculus isn't compared: its terms forbid evaluating AI models without permission.",
    ]


def build_report(results: Sequence[LLMBacktestResult], out_dir: Path) -> dict[str, Any]:
    """Write llm_comparison.html and .json to `out_dir`; returns the JSON summary.

    The primary run (the sections other than cost) is the first one whose
    model is Opus, else the first.
    """
    import matplotlib

    matplotlib.use("Agg")
    primary = next(
        (r for r in results if "opus" in r.model), results[0] if results else None
    )
    sections: dict[str, Any] = {"primary": primary, "cost": cost_table(results)}
    charts: dict[str, str] = {}
    if primary is not None:
        sections.update({
            "verdict": verdict(primary),
            "binary": binary_table(primary),
            "by_category": binary_by(primary, "category"),
            "by_lead": binary_by(primary, "lead_days"),
            "numeric": numeric_table(primary),
            "calibration": calibration(primary) if len(primary.binary) else {},
            "rationales": sample_rationales(primary),
            "phrases": key_phrases(primary),
            "caveats": caveats(primary),
        })  # fmt: skip
        if not sections["binary"].empty:
            charts["brier"] = plot_brier(sections["binary"])
        if not sections["numeric"].empty:
            charts["numeric"] = plot_numeric(sections["numeric"])
        if sections["calibration"]:
            charts["calibration"] = plot_calibration(sections["calibration"])
        if not sections["phrases"].empty:
            charts["phrases"] = plot_phrases(sections["phrases"])
    if len(sections["cost"].dropna(subset=["usd_per_call", "llm_brier"])) >= 2:
        charts["cost"] = plot_cost(sections["cost"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llm_comparison.html").write_text(render_html(sections, charts))
    summary: dict[str, Any] = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": [r.model for r in results],
        "primary_model": primary.model if primary else None,
        "training_cutoff": primary.cutoff if primary else None,
        "cost": sections["cost"].to_dict(orient="records"),
    }
    if primary is not None:
        summary.update({
            "verdict": sections["verdict"],
            "binary": sections["binary"].to_dict(orient="records"),
            "binary_by_category": sections["by_category"].to_dict(orient="records"),
            "binary_by_lead": sections["by_lead"].to_dict(orient="records"),
            "numeric": sections["numeric"].to_dict(orient="records"),
            "calibration": {c: t.to_dict(orient="records") for c, t in sections["calibration"].items()},
            "sample_rationales": sections["rationales"].to_dict(orient="records"),
            "key_phrases": sections["phrases"].to_dict(orient="records"),
            "caveats": sections["caveats"],
        })  # fmt: skip
    (out_dir / "llm_comparison.json").write_text(
        json.dumps(_json_safe(summary), indent=2)
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the LLM comparison report.")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR,
                        help=f"saved backtest runs (default {RESULTS_DIR})")  # fmt: skip
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    results = load_results(args.results)
    summary = build_report(results, args.out)
    print(f"wrote {args.out / 'llm_comparison.html'} ({len(results)} run(s))")
    if summary.get("verdict"):
        print(json.dumps(_json_safe(summary["verdict"]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
