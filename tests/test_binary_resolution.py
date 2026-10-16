"""Tests for binary resolution, Brier scoring and AI resolving (AI task 11)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from typing import Any

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import ingest_arxiv
from eco_prediction.ai.monthly_ingest import ingest_backfill, ingest_live
from eco_prediction.ai.months import MonthlyCount
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.ai.resolutions import (
    parse_threshold_key,
    resolve_and_score_ai,
)
from eco_prediction.ai.store import (
    AIObservation,
    ensure_source,
    insert_ai_observations,
    upsert_ai_series,
)
from eco_prediction.backtest import metrics
from eco_prediction.backtest.harness import Forecast
from eco_prediction.db.forecasts import (
    get_or_create_model_version,
    get_or_create_question,
    save_binary_forecast,
    save_forecast,
    save_probability_forecast,
)
from eco_prediction.db.migrate import migrate
from eco_prediction.db.resolutions import (
    resolve_binary_question,
    save_resolution,
    scored_forecasts,
    unscored_forecasts,
)

ARXIV = ingest_arxiv.SERIES
NOV = date(2026, 11, 1)
DEC = date(2026, 12, 1)


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def rows(conn: Connection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# --- metrics ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "actual", "expected"),
    [
        (0.0, "yes", 1.0),
        (0.0, "no", 0.0),
        (0.5, "yes", 0.25),
        (0.5, "no", 0.25),
        (1.0, "yes", 0.0),
        (1.0, "no", 1.0),
        (0.8, "yes", 0.04),
        (0.8, "no", 0.64),
    ],
)
def test_brier_score_binary(p: float, actual: str, expected: float) -> None:
    assert metrics.brier_score_binary(p, actual) == pytest.approx(expected)


def test_binary_brier_is_half_the_two_category_score() -> None:
    for p in (0.1, 0.35, 0.9):
        two = metrics.brier_score({"yes": p, "no": 1 - p}, "yes")
        assert two == pytest.approx(2 * metrics.brier_score_binary(p, "yes"))


@pytest.mark.parametrize(("p", "actual"), [(1.2, "yes"), (-0.1, "no"), (0.5, "hold")])
def test_brier_score_binary_rejects_bad_input(p: float, actual: str) -> None:
    with pytest.raises(ValueError):
        metrics.brier_score_binary(p, actual)


def test_mean_brier_and_log_loss() -> None:
    # (0.1^2 + 0.2^2 + 0.6^2) / 3, by hand.
    assert metrics.mean_brier_score_binary(
        [0.9, 0.2, 0.6], ["yes", "no", "no"]
    ) == pytest.approx(0.41 / 3)
    assert metrics.mean_brier_score_binary([0.9, 0.2], [1, 0]) == pytest.approx(0.025)
    assert metrics.log_loss_binary([0.5, 0.5], ["yes", "no"]) == pytest.approx(
        math.log(2)
    )
    assert math.isfinite(metrics.log_loss_binary([0.0], ["yes"]))  # clipped


def test_expected_calibration_error() -> None:
    assert metrics.expected_calibration_error(
        [0.25] * 4, ["yes", "no", "no", "no"]
    ) == pytest.approx(0.0)
    assert metrics.expected_calibration_error(
        [0.9] * 10, ["yes"] * 5 + ["no"] * 5
    ) == pytest.approx(0.4)


def test_brier_decomposition_adds_up() -> None:
    # Two bins: 0.2 (1 yes in 5) and 0.8 (4 in 5): perfectly calibrated.
    probs = [0.2] * 5 + [0.8] * 5
    hits = [1, 0, 0, 0, 0] + [1, 1, 1, 1, 0]
    d = metrics.brier_decomposition(probs, hits)
    assert d["reliability"] == pytest.approx(0.0)
    assert d["resolution"] == pytest.approx(0.09)
    assert d["uncertainty"] == pytest.approx(0.25)
    assert d["brier"] == pytest.approx(0.16)
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"]
    )


def test_brier_skill_score() -> None:
    assert metrics.brier_skill_score(0.16, 0.25) == pytest.approx(0.36)
    with pytest.raises(ValueError):
        metrics.brier_skill_score(0.1, 0.0)


# --- storing resolutions ----------------------------------------------------


def binary_question(conn: Connection, key: str = "ARXIV_AI_SUBMISSIONS>=33000") -> int:
    qid, _ = get_or_create_question(conn, "ai_binary_event", NOV, 1, question_key=key)
    return qid


def test_resolve_binary_question(migrated: Connection) -> None:
    qid = binary_question(migrated)
    resolve_binary_question(migrated, qid, "yes", date(2026, 12, 3))
    assert rows(
        migrated,
        "SELECT actual_outcome, actual_value, actual_as_of, resolved_at IS NOT NULL"
        " FROM resolutions WHERE question_id = %s",
        qid,
    ) == [("yes", None, date(2026, 12, 3), True)]
    with pytest.raises(psycopg2.errors.UniqueViolation):
        resolve_binary_question(migrated, qid, "no", date(2026, 12, 4))


def test_outcomes_must_fit_the_question_type(migrated: Connection) -> None:
    binary = binary_question(migrated)
    numeric, _ = get_or_create_question(
        migrated, "arxiv_ai_submissions", NOV, 1, question_key="ARXIV_AI_SUBMISSIONS"
    )
    fomc, _ = get_or_create_question(migrated, "fomc_decision", date(2026, 12, 9), 2)
    on = date(2026, 12, 10)
    for call in [
        lambda: resolve_binary_question(migrated, numeric, "yes", on),
        lambda: resolve_binary_question(migrated, fomc, "yes", on),
        lambda: resolve_binary_question(migrated, binary, "hold", on),  # type: ignore[arg-type]
        lambda: save_resolution(migrated, binary, actual_as_of=on, value=1.0),
        lambda: save_resolution(migrated, numeric, actual_as_of=on, outcome="yes"),
        lambda: save_resolution(migrated, fomc, actual_as_of=on, outcome="no"),
        lambda: save_resolution(migrated, 999_999, actual_as_of=on, value=1.0),
    ]:
        with pytest.raises(ValueError):
            call()
    assert rows(migrated, "SELECT count(*) FROM resolutions") == [(0,)]


def test_economy_resolutions_still_work(migrated: Connection) -> None:
    fomc, _ = get_or_create_question(migrated, "fomc_decision", date(2026, 12, 9), 2)
    cpi, _ = get_or_create_question(migrated, "cpi_yoy", NOV, 1)
    save_resolution(migrated, fomc, actual_as_of=date(2026, 12, 10), outcome="hold")
    save_resolution(migrated, cpi, actual_as_of=date(2026, 12, 10), value=2.9)
    assert rows(
        migrated,
        "SELECT actual_outcome, actual_value::float8 FROM resolutions"
        " ORDER BY question_id",
    ) == [("hold", None), (None, 2.9)]


def test_save_binary_forecast(migrated: Connection) -> None:
    qid = binary_question(migrated)
    mv = get_or_create_model_version(migrated, "base_rate", "v1")
    save_binary_forecast(migrated, qid, mv, date(2026, 10, 15), 0.35)
    assert rows(migrated, "SELECT probabilities FROM forecasts") == [
        ({"yes": 0.35, "no": 0.65},)
    ]
    with pytest.raises(ValueError):
        save_binary_forecast(migrated, qid, mv, date(2026, 10, 15), 1.5)


# --- resolving AI questions -------------------------------------------------


def store_arxiv(conn: Connection, *obs: tuple[date, date, float, bool]) -> None:
    source = ensure_source(conn, *ARXIV.source)
    pk = upsert_ai_series(conn, source, ARXIV.info)
    insert_ai_observations(conn, pk, [AIObservation(*o) for o in obs])


def resolutions(conn: Connection) -> dict[str, tuple[Any, ...]]:
    return {
        key: (outcome, None if value is None else float(value), as_of)
        for key, outcome, value, as_of in rows(
            conn,
            """
            SELECT q.question_key, r.actual_outcome, r.actual_value, r.actual_as_of
            FROM resolutions r JOIN questions q ON q.id = r.question_id
            """,
        )
    }


def test_numeric_and_threshold_questions_resolve_to_the_first_live_value(
    migrated: Connection,
) -> None:
    get_or_create_question(
        migrated, "arxiv_ai_submissions", NOV, 1, question_key="ARXIV_AI_SUBMISSIONS"
    )
    binary_question(migrated, "ARXIV_AI_SUBMISSIONS>=33000")
    binary_question(migrated, "ARXIV_AI_SUBMISSIONS>=32000")  # equal counts as yes
    store_arxiv(
        migrated,
        (NOV, DEC, 99_999, True),  # a backfill: never resolves a question
        (NOV, date(2026, 12, 2), 32_000, False),
        (NOV, date(2026, 12, 9), 34_000, False),  # a later revision: ignored
    )

    early = resolve_and_score_ai(migrated, date(2026, 12, 1))
    assert (early.resolved, early.still_open) == (0, 3)

    result = resolve_and_score_ai(migrated, date(2026, 12, 15))
    assert (result.resolved, result.still_open) == (3, 0)
    assert resolutions(migrated) == {
        "ARXIV_AI_SUBMISSIONS": (None, 32_000.0, date(2026, 12, 2)),
        "ARXIV_AI_SUBMISSIONS>=33000": ("no", None, date(2026, 12, 2)),
        "ARXIV_AI_SUBMISSIONS>=32000": ("yes", None, date(2026, 12, 2)),
    }


class FakeCounter:
    def __init__(self, counts: dict[date, float]) -> None:
        self.counts = counts

    def count_months(self, months: Sequence[date]) -> Iterable[MonthlyCount]:
        return [MonthlyCount(m, self.counts[m]) for m in months]

    def close(self) -> None:
        pass


def test_a_backfilled_month_resolves_after_a_live_pull_confirms_it(
    migrated: Connection,
) -> None:
    # Backfilled after it ended, then re-counted unchanged by a live pull.
    get_or_create_question(
        migrated, "arxiv_ai_submissions", NOV, 1, question_key="ARXIV_AI_SUBMISSIONS"
    )
    counts = {date(2026, 9, 1): 29_000, date(2026, 10, 1): 30_000, NOV: 31_000}
    ingest_backfill(migrated, ARXIV, FakeCounter(counts), NOV, NOV)
    assert resolve_and_score_ai(migrated, date(2026, 12, 5)).resolved == 0
    ingest_live(migrated, ARXIV, FakeCounter(counts), date(2026, 12, 6))
    assert resolve_and_score_ai(migrated, date(2026, 12, 6)).resolved == 1
    assert resolutions(migrated)["ARXIV_AI_SUBMISSIONS"] == (
        None,
        31_000.0,
        date(2026, 12, 6),
    )


def market(conn: Connection, external_id: str, **fields: Any) -> int:
    pk, _, _ = upsert_platform_question(
        conn,
        PlatformQuestion(
            platform="polymarket",
            external_id=external_id,
            title=f"Market {external_id}?",
            closes_at=datetime(2026, 12, 31, tzinfo=UTC),
            **fields,
        ),
    )
    qid, _ = get_or_create_question(
        conn,
        "ai_binary_event",
        date(2026, 12, 31),
        lead_days=30,
        question_key=f"polymarket:{external_id}",
        platform_question_id=pk,
    )
    return qid


def test_curated_questions_resolve_with_their_market(migrated: Connection) -> None:
    resolved_on = datetime(2026, 12, 20, 15, tzinfo=UTC)
    market(migrated, "1", resolved_at=resolved_on, resolved_outcome="yes")
    market(migrated, "2", resolved_at=resolved_on, resolved_outcome="no")
    market(migrated, "3", metadata={"uma_resolution_status": "resolved"})  # 50/50
    market(migrated, "4")  # still open

    assert resolve_and_score_ai(migrated, date(2026, 12, 19)).resolved == 0
    result = resolve_and_score_ai(migrated, date(2026, 12, 21))
    assert (result.resolved, result.still_open) == (2, 2)
    assert resolutions(migrated) == {
        "polymarket:1": ("yes", None, date(2026, 12, 20)),
        "polymarket:2": ("no", None, date(2026, 12, 20)),
    }


def test_an_unknown_binary_key_raises(migrated: Connection) -> None:
    with pytest.raises(ValueError, match="not a threshold question key"):
        parse_threshold_key("swe_bench_verified")
    assert parse_threshold_key("EPOCH_SWEBENCH_SOTA>=87.6") == (
        "EPOCH_SWEBENCH_SOTA",
        87.6,
    )


# --- scoring ----------------------------------------------------------------


def test_ai_forecasts_are_scored_by_question_type(migrated: Connection) -> None:
    numeric, _ = get_or_create_question(
        migrated, "arxiv_ai_submissions", NOV, 1, question_key="ARXIV_AI_SUBMISSIONS"
    )
    binary = binary_question(migrated)
    store_arxiv(migrated, (NOV, date(2026, 12, 2), 34_000, False))
    mv = get_or_create_model_version(migrated, "random_walk", "v1")
    when = date(2026, 10, 15)
    save_forecast(migrated, numeric, mv, when, Forecast(33_000, 31_000, 35_000))
    save_binary_forecast(migrated, binary, mv, when, 0.8)

    result = resolve_and_score_ai(migrated, date(2026, 12, 15))
    assert result.scored == 2
    scored = scored_forecasts(migrated, domain="ai").set_index("question_type")
    assert scored.loc["numeric_interval", "error"] == pytest.approx(-1_000)
    assert bool(scored.loc["numeric_interval", "in_interval"]) is True
    assert scored.loc["binary", "brier_score"] == pytest.approx(0.04)  # (0.8 - 1)^2
    assert scored.loc["binary", "question_key"] == "ARXIV_AI_SUBMISSIONS>=33000"
    # The economy's readers don't see them.
    assert scored_forecasts(migrated).empty


def test_the_economy_scorer_skips_ai_forecasts(migrated: Connection) -> None:
    binary = binary_question(migrated)
    fomc, _ = get_or_create_question(migrated, "fomc_decision", date(2026, 12, 9), 2)
    mv = get_or_create_model_version(migrated, "m", "v1")
    save_binary_forecast(migrated, binary, mv, date(2026, 10, 15), 0.8)
    save_probability_forecast(
        migrated, fomc, mv, date(2026, 10, 15), {"cut": 0.1, "hold": 0.8, "hike": 0.1}
    )
    resolve_binary_question(migrated, binary, "yes", date(2026, 12, 2))
    save_resolution(migrated, fomc, actual_as_of=date(2026, 12, 10), outcome="hold")

    economy = unscored_forecasts(migrated)
    assert [f.question_type for f in economy] == ["probability"]
    assert [f.question_type for f in unscored_forecasts(migrated, domain="ai")] == [
        "binary"
    ]
