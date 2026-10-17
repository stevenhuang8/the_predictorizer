"""Tests for the AI forecast and resolution jobs (AI task 21).

Each test gets a throwaway, migrated database (skipped if Postgres isn't
reachable), seeded with arXiv history through 2026-09 and two Polymarket
markets. The API is never called.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.baselines import AILinearTrend, AIRandomWalk
from eco_prediction.ai.ingest_arxiv import SERIES as ARXIV_SERIES
from eco_prediction.ai.jobs import ai_forecast_job as job
from eco_prediction.ai.jobs import ai_resolution_job as resolution
from eco_prediction.ai.lightgbm_model import AILightGBMModel
from eco_prediction.ai.llm.claude_forecaster import MODEL, ClaudeForecaster
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.ai.store import (
    AIObservation,
    insert_ai_observations,
    upsert_ai_series,
)
from eco_prediction.data.ingest import ensure_source
from eco_prediction.db.migrate import migrate
from eco_prediction.scheduler.forecast_job import LOCK_KEY as ECONOMY_LOCK_KEY

ARXIV = "ARXIV_AI_SUBMISSIONS"
TODAY = date(2026, 10, 16)
SMALL_GRID = {"max_depth": [3], "learning_rate": [0.1], "min_child_samples": [10]}
FAST_NUMERIC = (
    job.NumericSpec("ai_random_walk", lambda s, h: AIRandomWalk()),
    job.NumericSpec("ai_linear_trend", lambda s, h: AILinearTrend()),
    job.NumericSpec(
        "ai_lightgbm",
        lambda s, h: AILightGBMModel(h, param_grid=SMALL_GRID),
        explain=True,
    ),
)


def utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


@pytest.fixture
def db(conn: Connection) -> Connection:
    migrate(conn)
    source = ensure_source(conn, *ARXIV_SERIES.source)
    series = upsert_ai_series(conn, source, ARXIV_SERIES.info)
    months = list(iter_months(date(2021, 1, 1), date(2026, 9, 1)))
    insert_ai_observations(conn, series, [
        AIObservation(m, next_month(m), 4000 * 1.015**i, True) for i, m in enumerate(months)
    ])  # fmt: skip
    # "live": asked 90 days before close = yesterday. "late": asked 14 days ago.
    for ext, ask in (
        ("live", TODAY - timedelta(days=1)),
        ("late", TODAY - timedelta(days=14)),
    ):
        close = ask + timedelta(days=90)
        pq, _, _ = upsert_platform_question(conn, PlatformQuestion(
            "polymarket", ext, f"Will OpenAI release GPT-7 by {close}?",
            opened_at=utc(date(2026, 6, 1)), closes_at=utc(close),
            metadata={"closed_at": None},  # as the live ingest stores an open market
        ))  # fmt: skip
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO platform_probabilities VALUES (%s, %s, 0.3, false)",
                (pq, ask - timedelta(days=5)),
            )
    conn.commit()
    return conn


def run(db: Connection, **kwargs: Any) -> job.AIJobResult:
    kwargs.setdefault("numeric_models", FAST_NUMERIC)
    kwargs.setdefault("classifier", False)
    return job.run_ai_forecast_job(db, TODAY, **kwargs)


def rows(db: Connection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    with db.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# --- the forecast job ----------------------------------------------------------------


def test_the_job_creates_questions_and_stores_every_models_forecasts(
    db: Connection,
) -> None:
    result = run(db)
    assert result.failures == []
    # 7 series x 3 horizons numeric; arXiv's threshold sets (only it has a base
    # month); and the two curated questions due by today (the 90-day leads).
    kinds = dict(
        rows(db, "SELECT question_type::text, count(*) FROM questions GROUP BY 1")
    )
    assert kinds["numeric_interval"] == 21
    assert rows(
        db, "SELECT count(*) FROM questions WHERE platform_question_id IS NOT NULL"
    ) == [(2,)]
    stored = dict(rows(db, """
        SELECT m.model_type, count(*) FROM forecasts f
        JOIN model_versions m ON m.id = f.model_version_id GROUP BY 1"""))  # fmt: skip
    # arXiv numeric: 3 horizons x 3 models. Thresholds: 9 questions x base rate.
    # Curated: 2 questions x (market price + category base rate).
    assert stored == {"ai_random_walk": 3, "ai_linear_trend": 3, "ai_lightgbm": 3,
                      "ai_base_rate": 9 + 2, "ai_market_price": 2}  # fmt: skip
    assert result.forecasts_inserted == sum(stored.values())
    # The other six series have no data: unavailable, not failures.
    assert result.unavailable == 6 * 3 * 3
    # LightGBM stored SHAP values and its feature row.
    shap = rows(db, """SELECT count(*) FROM forecasts f JOIN model_versions m
        ON m.id = f.model_version_id WHERE m.model_type = 'ai_lightgbm'
        AND f.shap_values IS NOT NULL AND f.feature_snapshot_id IS NOT NULL""")  # fmt: skip
    assert shap == [(3,)]


def test_binary_forecasts_are_dated_at_their_ask_dates(db: Connection) -> None:
    run(db)
    dated = rows(db, """
        SELECT p.external_id, f.forecast_date, f.is_backtest, (f.probabilities->>'yes')::float8
        FROM forecasts f JOIN questions q ON q.id = f.question_id
        JOIN platform_questions p ON p.id = q.platform_question_id
        JOIN model_versions m ON m.id = f.model_version_id
        WHERE m.model_type = 'ai_market_price' ORDER BY 1""")  # fmt: skip
    assert dated == [
        ("late", TODAY - timedelta(days=14), True, 0.3),  # caught up: a backtest
        ("live", TODAY - timedelta(days=1), False, 0.3),
    ]
    numeric_dates = rows(
        db,
        """SELECT DISTINCT forecast_date, is_backtest FROM forecasts f
        JOIN questions q ON q.id = f.question_id WHERE q.question_type = 'numeric_interval'""",
    )
    assert numeric_dates == [(TODAY, False)]


def test_runs_are_idempotent(db: Connection) -> None:
    first = run(db)
    again = run(db)
    assert again.forecasts_inserted == 0 and again.questions_created == 0
    assert again.forecasts_skipped == first.forecasts_inserted
    assert rows(db, "SELECT count(*) FROM forecasts") == [(first.forecasts_inserted,)]


def test_a_market_resolved_before_its_ask_date_isnt_forecast(db: Connection) -> None:
    with db.cursor() as cur:
        cur.execute(
            "UPDATE platform_questions SET resolved_at = %s, resolved_outcome = 'yes'"
            " WHERE external_id = 'late'",
            (utc(TODAY - timedelta(days=20)),),
        )
    db.commit()
    run(db)
    due = {d.question.question_key for d in job.due_binary_questions(db, TODAY)}
    assert "polymarket:live" in due and "polymarket:late" not in due


def test_an_unexpected_error_is_a_failure_and_doesnt_stop_the_run(
    db: Connection,
) -> None:
    def broken(series: str, horizon: int) -> Any:
        raise RuntimeError("boom")

    result = run(
        db, numeric_models=(*FAST_NUMERIC[:1], job.NumericSpec("broken", broken))
    )
    assert len(result.failures) == 21 and "boom" in result.failures[0]
    assert rows(
        db,
        """SELECT count(*) FROM forecasts f JOIN model_versions m
        ON m.id = f.model_version_id WHERE m.model_type = 'ai_random_walk'""",
    ) == [(3,)]


# --- the LLM (opt-in) ---------------------------------------------------------------------


class FakeClaude:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        numeric = "point_forecast" in json.dumps(kwargs["output_config"])
        text = json.dumps(
            {"rationale": "trend", "point_forecast": 5000, "lower_80": 4500, "upper_80": 5500}
            if numeric else {"rationale": "unlikely", "probability": 0.2}
        )  # fmt: skip
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
            model=MODEL, stop_details=None,
            usage=SimpleNamespace(input_tokens=4000, output_tokens=1000, iterations=None,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )  # fmt: skip


def test_the_llm_runs_only_when_asked_and_within_its_budget(db: Connection) -> None:
    fake = FakeClaude()
    forecaster = ClaudeForecaster(conn=db, client=fake)  # type: ignore[arg-type]
    # $0.036 a call: a $0.10 budget allows three calls, then stops the LLM.
    result = run(
        db, numeric_models=FAST_NUMERIC[:1], llm=forecaster, llm_budget_usd=0.10
    )
    assert len(fake.requests) == 3
    assert result.llm_spent_usd == pytest.approx(0.108)
    assert any(f.startswith("llm budget") for f in result.failures)
    llm = rows(
        db,
        """SELECT count(*), count(f.rationale), bool_and(c.forecast_id = f.id)
        FROM forecasts f JOIN model_versions m ON m.id = f.model_version_id
        JOIN llm_calls c ON c.forecast_id = f.id WHERE m.model_type = 'claude_llm'""",
    )
    assert llm == [(3, 3, True)]
    # The cheap models still forecast everything despite the LLM stopping early.
    assert rows(
        db,
        """SELECT count(*) FROM forecasts f JOIN model_versions m
        ON m.id = f.model_version_id WHERE m.model_type = 'ai_market_price'""",
    ) == [(2,)]
    # Next run, with budget: the stored LLM forecasts are skipped, the rest made.
    again = run(db, numeric_models=FAST_NUMERIC[:1], llm=forecaster, llm_budget_usd=5)
    assert again.failures == [] and len(fake.requests) > 3
    assert (
        rows(
            db,
            """SELECT count(*) FROM forecasts f JOIN model_versions m
        ON m.id = f.model_version_id WHERE m.model_type = 'claude_llm'""",
        )[0][0]
        > 3
    )


def test_without_llm_nothing_is_called() -> None:
    with pytest.raises(SystemExit):
        job.main(["--llm", "claude-haiku-5-5"])  # a budget is required
    with pytest.raises(SystemExit):
        job.main(["--llm", "claude-haiku-5-5", "--llm-budget", "1", "--dry-run"])


# --- main: locks and dry runs ---------------------------------------------------------------


@contextmanager
def using(conn: Connection) -> Iterator[Connection]:
    yield conn


def main(db: Connection, *argv: str) -> int:
    return job.main(
        ["--no-ingest", "--log-file", "/dev/null", *argv],
        connect=lambda: using(db), refresh=list, clock=lambda: TODAY,
    )  # fmt: skip


@pytest.fixture
def other(test_dsn: str) -> Iterator[Connection]:
    c = psycopg2.connect(test_dsn)
    yield c
    c.close()


def test_the_ai_lock_blocks_a_second_run_and_the_economy_lock_doesnt(
    db: Connection, other: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job, "NUMERIC_MODELS", FAST_NUMERIC[:1])
    with other.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s)", (job.LOCK_KEY,))
    assert main(db) == 0
    assert rows(db, "SELECT count(*) FROM questions") == [(0,)]  # it did nothing
    with other.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s)", (job.LOCK_KEY,))
        cur.execute(
            "SELECT pg_advisory_lock(%s)", (ECONOMY_LOCK_KEY,)
        )  # an economy run
    assert main(db, "--no-resolve") in (0, 1)
    assert rows(db, "SELECT count(*) FROM questions")[0][0] > 0  # it ran anyway


def test_a_dry_run_changes_nothing(
    db: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job, "NUMERIC_MODELS", FAST_NUMERIC[:2])
    before = rows(
        db, "SELECT (SELECT count(*) FROM questions), (SELECT count(*) FROM forecasts)"
    )
    main(db, "--dry-run")
    after = rows(
        db, "SELECT (SELECT count(*) FROM questions), (SELECT count(*) FROM forecasts)"
    )
    assert before == after == [(0, 0)]
    main(db)  # a real run then stores them
    assert rows(db, "SELECT count(*) FROM forecasts")[0][0] > 0


def test_once_per_month_reuses_the_months_forecast_date(
    db: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job, "NUMERIC_MODELS", FAST_NUMERIC[:1])
    job.run_ai_forecast_job(db, date(2026, 10, 2), numeric_models=FAST_NUMERIC[:1],
                            classifier=False)  # fmt: skip
    assert job.month_forecast_date(db, TODAY) == date(2026, 10, 2)
    main(db, "--once-per-month", "--no-resolve")
    numeric_dates = rows(
        db,
        """SELECT DISTINCT forecast_date FROM forecasts f
        JOIN questions q ON q.id = f.question_id WHERE q.question_type = 'numeric_interval'""",
    )
    assert numeric_dates == [(date(2026, 10, 2),)]


# --- the resolution job ---------------------------------------------------------------------


def test_resolution_resolves_scores_and_summarizes(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    run(db, numeric_models=FAST_NUMERIC[:2])
    # November's first live arXiv value arrives in December; the late market resolves no.
    series_pk = rows(db, "SELECT id FROM series WHERE series_id = %s", ARXIV)[0][0]
    insert_ai_observations(db, series_pk, [
        AIObservation(date(2026, 11, 1), date(2026, 12, 2), 5200.0, False)
    ])  # fmt: skip
    with db.cursor() as cur:
        cur.execute("UPDATE platform_questions SET resolved_at = %s, resolved_outcome = 'no'"
                    " WHERE external_id = 'late'", (utc(date(2026, 12, 5)),))  # fmt: skip
    db.commit()
    args = ["--date", "2026-12-10", "--log-file", "/dev/null"]
    hooks: dict[str, Any] = {"connect": lambda: using(db), "clock": lambda: TODAY}

    assert resolution.main([*args, "--dry-run"], **hooks) == 0
    assert rows(db, "SELECT count(*) FROM resolutions") == [(0,)]  # rolled back

    assert resolution.main(args, **hooks) == 0
    out = capsys.readouterr().out
    resolved = rows(
        db,
        """SELECT q.question_key, r.actual_value, r.actual_outcome
        FROM resolutions r JOIN questions q ON q.id = r.question_id ORDER BY 1""",
    )
    keys = {k for k, _, _ in resolved}
    assert ARXIV in keys and "polymarket:late" in keys
    assert "polymarket:live" not in keys  # still open
    summary = resolution.ai_score_summary(
        resolution.scored_ai_forecasts(db, include_backtest=True)
    )
    numeric = summary["numeric"].set_index("model_type")
    assert numeric.at["ai_random_walk", "n"] == 1
    assert numeric.at["ai_random_walk", "relative_rmse"] > 0
    binary = summary["binary"].set_index("model_type")
    # The market said 0.3 and the outcome was no: Brier 0.09.
    assert binary.at["ai_market_price", "brier"] == pytest.approx(0.09)
    assert binary.at["ai_market_price", "calibration_gap"] == pytest.approx(0.3)
    # By default only live forecasts are summarized: the numeric ones (made
    # today) but not the late market's (a catch-up, so a backtest).
    assert "ai_random_walk" in out and "ai_market_price" not in out
    assert resolution.main([*args, "--include-backtest"], **hooks) == 0
    assert "ai_market_price" in capsys.readouterr().out


def test_the_cli_passes_options_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from eco_prediction import cli

    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(
        job, "main", lambda argv: seen.setdefault("forecast", argv) and 0
    )
    monkeypatch.setattr(
        resolution, "main", lambda argv: seen.setdefault("resolve", argv) and 0
    )
    assert cli.main(["ai-forecast", "--dry-run", "--no-ingest"]) == 0
    assert cli.main(["ai-resolve", "--include-backtest"]) == 0
    assert seen == {
        "forecast": ["--dry-run", "--no-ingest"],
        "resolve": ["--include-backtest"],
    }


def test_running_max_forecasts_store_as_plain_floats(db: Connection) -> None:
    """Regression: the bounded baselines returned numpy floats, which psycopg2
    wrote as `np.float64(...)` (schema "np" does not exist)."""
    from eco_prediction.ai.ingest_epoch import SERIES as EPOCH_SERIES

    gpqa = EPOCH_SERIES["gpqa"].series
    source = ensure_source(db, *gpqa.source)
    pk = upsert_ai_series(db, source, gpqa.info)
    months = list(iter_months(date(2024, 1, 1), date(2026, 9, 1)))
    insert_ai_observations(db, pk, [
        AIObservation(m, next_month(m), min(95.0, 50 + i * 1.5), True) for i, m in enumerate(months)
    ])  # fmt: skip
    db.commit()
    result = run(db, numeric_models=FAST_NUMERIC[:2])
    assert result.failures == []
    stored = rows(
        db,
        """SELECT m.model_type, f.prediction::float8 FROM forecasts f
        JOIN model_versions m ON m.id = f.model_version_id JOIN questions q ON q.id = f.question_id
        WHERE q.question_key = 'EPOCH_GPQA_SOTA' ORDER BY 1""",
    )
    assert len(stored) == 6 and all(95.0 <= p <= 100 for _, p in stored)
