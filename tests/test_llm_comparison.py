"""Tests for the LLM comparison report (AI task 20), on synthetic backtest
results: no database, no API."""

from __future__ import annotations

import base64
import json
import re
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from eco_prediction.ai.llm.backtest import (
    BacktestBudget,
    LLMBacktestResult,
    load_results,
    save_result,
    summarize_backtest,
)
from eco_prediction.ai.llm.comparison_report import (
    binary_by,
    binary_table,
    build_report,
    cluster_bootstrap,
    key_phrases,
    main,
    numeric_table,
    sample_rationales,
    verdict,
)
from eco_prediction.ai.llm.ensemble import LLMEnsembleForecaster

CUTOFF = date(2026, 7, 1)
SERIES = ("ARXIV_AI_SUBMISSIONS", "HF_NEW_MODELS", "GITHUB_AI_ACTIVITY")


def synthetic_result(
    model: str = "claude-opus-5-5",
    markets: int = 40,
    llm_skill: float = 0.8,
    seed: int = 0,
    per_call: float = 0.07,
) -> LLMBacktestResult:
    """`markets` markets x 3 leads. The market price and LLM lean towards the
    outcome by `llm_skill`; the base rate is a flat 0.3."""
    rng = np.random.default_rng(seed)
    rows = []
    qid = 0
    for m in range(markets):
        outcome = "yes" if rng.random() < 0.3 else "no"
        y = float(outcome == "yes")
        close = CUTOFF + timedelta(days=100 + m)
        for lead in (90, 30, 7):
            qid += 1
            ask = close - timedelta(days=lead)
            base = {
                "question_id": qid, "ask_date": ask, "outcome": outcome,
                "resolved_at": datetime.combine(close, datetime.min.time(), tzinfo=UTC),
                "category": ("release", "benchmark", "ranking")[m % 3],
                "title": f"Will lab {m} ship model {m} by {close}?", "market_id": m,
                "lead_days": lead,
            }  # fmt: skip
            llm_p = float(
                np.clip(
                    y * llm_skill + (1 - llm_skill) * 0.5 + rng.normal(0, 0.05),
                    0.01,
                    0.99,
                )
            )
            market_p = float(np.clip(y * 0.6 + 0.2 + rng.normal(0, 0.1), 0.01, 0.99))
            rows += [
                {**base, "component": "market", "p_yes": market_p},
                {**base, "component": "base_rate", "p_yes": 0.3},
                {**base, "component": "llm", "p_yes": llm_p, "cost_usd": per_call, "call_id": qid,
                 "rationale": f"Lab {m} release cadence and benchmark evidence suggest "
                              f"{'likely' if llm_p > 0.5 else 'unlikely'} shipping."},
            ]  # fmt: skip
    binary = pd.DataFrame(rows)
    numeric_rows = []
    for s in SERIES:
        for fd, target in (
            (date(2026, 7, 1), date(2026, 8, 1)),
            (date(2026, 8, 1), date(2026, 9, 1)),
        ):
            actual = 10_000.0
            for comp, pred in (("llm", 10_200.0), ("random_walk", 9_500.0),
                               ("arima", 10_400.0), ("lightgbm", 9_900.0)):  # fmt: skip
                numeric_rows.append({
                    "series_id": s, "forecast_date": fd, "target_period": target,
                    "data_as_of": fd - timedelta(days=1), "horizon_months": 1,
                    "component": comp, "prediction": pred, "lower": pred * 0.9,
                    "upper": pred * 1.1, "actual": actual, "actual_as_of": target + timedelta(days=31),
                })  # fmt: skip
    numeric = pd.DataFrame(numeric_rows)
    ens = LLMEnsembleForecaster("learned")
    ens_b, ens_n = ens.walk_forward_binary(binary), ens.walk_forward_numeric(numeric)
    budget = BacktestBudget(100.0, spent_usd=per_call * qid, calls=qid)
    summary = summarize_backtest(binary, numeric, ens_b, ens_n, budget, model=model,
                                 cutoff=CUTOFF, dropped=12)  # fmt: skip
    return LLMBacktestResult(
        model, CUTOFF, binary, numeric, ens_b, ens_n, budget, 12, summary
    )


RESULT = synthetic_result()


# --- tables -------------------------------------------------------------------------


def test_binary_table_scores_every_component_on_common_questions() -> None:
    table = binary_table(RESULT).set_index("model")
    assert list(table.index) == ["llm", "market", "base_rate", "ensemble"]
    assert (table["n"] == 120).all() and (table["n_markets"] == 40).all()
    assert (
        table.at["llm", "brier"]
        < table.at["market", "brier"]
        < table.at["base_rate", "brier"]
    )
    # Skill against the base rate is positive, and the interval excludes zero.
    assert table.at["llm", "skill_vs_base_rate"] > 0
    low, high = map(float, table.at["llm", "ci_vs_base_rate"].strip("[]").split(", "))
    assert low < table.at["llm", "diff_vs_base_rate"] < high < 0
    assert pd.isna(table.at["market", "diff_vs_market"])  # no self-comparison


def test_the_bootstrap_resamples_markets_not_questions() -> None:
    # Two markets, each with many identical questions: with only two clusters
    # the interval spans both markets' values, however many questions there are.
    diff = pd.Series([1.0] * 50 + [-1.0] * 50)
    clusters = pd.Series([0] * 50 + [1] * 50)
    low, high = cluster_bootstrap(diff, clusters)
    assert low == pytest.approx(-1.0) and high == pytest.approx(1.0)
    # Treating every question as independent would wrongly be far tighter.
    low_i, high_i = cluster_bootstrap(diff, pd.Series(range(100)))
    assert high_i - low_i < 0.5
    assert all(np.isnan(cluster_bootstrap(diff, pd.Series([0] * 100))))


def test_breakdowns_by_category_and_lead() -> None:
    by_cat = binary_by(RESULT, "category")
    assert list(by_cat["category"]) == ["benchmark", "ranking", "release"]
    assert by_cat["n"].sum() == 120
    by_lead = binary_by(RESULT, "lead_days")
    assert list(by_lead["lead_days"]) == [7, 30, 90] and (by_lead["n"] == 40).all()
    assert binary_by(RESULT, "nonexistent").empty


def test_numeric_table_is_relative_and_against_the_random_walk() -> None:
    table = numeric_table(RESULT).set_index("model")
    assert table.at["llm", "relative_rmse"] == pytest.approx(0.02)
    assert table.at["random_walk", "relative_rmse"] == pytest.approx(0.05)
    assert table.at["llm", "vs_random_walk"] == pytest.approx(0.4)
    assert (table["n"] == 6).all() and "ensemble" in table.index


def test_rationales_and_phrases() -> None:
    samples = sample_rationales(RESULT, k=3)
    assert set(samples["kind"]) <= {"hit", "miss"}
    hits = samples[samples["kind"] == "hit"]
    assert len(hits) == 3 and hits["rationale"].str.contains("Lab").all()
    assert hits["p_yes"].sub(0.5).abs().is_monotonic_decreasing  # most confident first
    assert {"title", "p_yes", "market", "outcome"} <= set(samples.columns)
    phrases = key_phrases(RESULT, top=5)
    by_kind = phrases.groupby("kind")["phrase"].apply(set)
    assert "release cadence" in by_kind["pair"] and "release" in by_kind["word"]
    assert len(phrases) == 5 and "the" not in set(phrases["phrase"])


def test_the_verdict_is_the_prd_criterion() -> None:
    v = verdict(RESULT)
    assert v is not None and v["passed"] and v["significant"]
    assert v["llm_brier"] < v["base_rate_brier"] and v["n_markets"] == 40
    weak = verdict(synthetic_result(llm_skill=0.0, seed=1))  # always ~0.5
    assert weak is not None and not weak["passed"]


# --- the report ------------------------------------------------------------------------


class Structure(HTMLParser):
    """Collects headings and images, and checks tags are balanced."""

    VOID: ClassVar[frozenset[str]] = frozenset(
        {"meta", "img", "br", "hr", "link", "input"}
    )

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.headings: list[str] = []
        self.images: list[str] = []
        self._in_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            self.images.append(dict(attrs)["src"] or "")
        if tag not in self.VOID:
            self.stack.append(tag)
        self._in_heading = tag in ("h1", "h2", "h3")

    def handle_endtag(self, tag: str) -> None:
        assert self.stack and self.stack[-1] == tag, (
            f"unbalanced </{tag}> after {self.stack[-3:]}"
        )
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self._in_heading and data.strip():
            self.headings.append(data.strip())
            self._in_heading = False


def test_the_report_has_every_section_valid_html_and_charts(tmp_path: Path) -> None:
    haiku = synthetic_result("claude-haiku-5-5", llm_skill=0.6, seed=2, per_call=0.002)
    summary = build_report([haiku, RESULT], tmp_path)
    html = (tmp_path / "llm_comparison.html").read_text()
    parser = Structure()
    parser.feed(html)
    assert parser.stack == []  # every tag closed
    for heading in ("LLM comparison", "Binary questions", "By category", "Numeric targets",
                    "Cost", "Calibration", "Sample rationales", "Key phrases in the rationales",
                    "Caveats"):  # fmt: skip
        assert heading in parser.headings, heading
    # Charts: Brier, numeric, cost (two models), calibration, phrases; all real PNGs.
    assert len(parser.images) == 5
    for src in parser.images:
        assert base64.b64decode(src.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
    assert "Success criterion" in html and "(significant)" in html
    assert "claude-opus-5-5, forecasts made from 2026-07-01" in html  # Opus is primary
    assert "12 earlier questions were dropped" in html
    assert re.search(r"Polymarket price", html)

    on_disk = json.loads((tmp_path / "llm_comparison.json").read_text())
    assert on_disk["primary_model"] == "claude-opus-5-5" == summary["primary_model"]
    assert on_disk["models"] == ["claude-haiku-5-5", "claude-opus-5-5"]
    assert {r["model"] for r in on_disk["binary"]} == {
        "llm",
        "market",
        "base_rate",
        "ensemble",
    }
    assert set(on_disk["calibration"]) == {"llm", "market"}
    assert (
        on_disk["verdict"]["passed"] is True and len(on_disk["sample_rationales"]) > 0
    )
    costs = {r["model"]: r for r in on_disk["cost"]}
    assert costs["claude-haiku-5-5"]["usd_per_call"] == pytest.approx(0.002)
    assert on_disk["training_cutoff"] == "2026-07-01"


def test_with_no_results_the_report_says_how_to_get_some(tmp_path: Path) -> None:
    summary = build_report([], tmp_path)
    html = (tmp_path / "llm_comparison.html").read_text()
    assert "No LLM backtest results yet" in html and "--budget N" in html
    assert summary["primary_model"] is None and summary["cost"] == []


def test_runs_are_saved_loaded_and_reported_from_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    save_result(RESULT, tmp_path / "runs")
    loaded = load_results(tmp_path / "runs")
    assert [r.model for r in loaded] == ["claude-opus-5-5"]
    assert loaded[0].binary.equals(RESULT.binary)
    assert load_results(tmp_path / "missing") == []
    assert (
        main(["--results", str(tmp_path / "runs"), "--out", str(tmp_path / "out")]) == 0
    )
    assert (tmp_path / "out" / "llm_comparison.html").exists()
    assert '"passed": true' in capsys.readouterr().out
