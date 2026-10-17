"""Tests for `eco-forecast ... --domain ai` (AI task 23).

Each test gets a throwaway, migrated database (skipped if Postgres isn't
reachable) with arXiv history, AI questions from the forecast job, and an
economy question; `cli.main` is pointed at it. The API is never called.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta

import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction import cli
from eco_prediction.ai import cli_ai
from eco_prediction.ai.baselines import AIRandomWalk
from eco_prediction.ai.ingest_arxiv import SERIES as ARXIV_SERIES
from eco_prediction.ai.jobs import ai_forecast_job as job
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.ai.resolutions import resolve_and_score_ai
from eco_prediction.ai.store import (
    AIObservation,
    insert_ai_observations,
    upsert_ai_series,
)
from eco_prediction.data.ingest import ensure_source
from eco_prediction.db import connection as db_connection
from eco_prediction.db.forecasts import get_or_create_question
from eco_prediction.db.migrate import migrate

TODAY = date(2026, 10, 16)


def utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


@pytest.fixture
def db(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> Connection:
    migrate(conn)
    source = ensure_source(conn, *ARXIV_SERIES.source)
    series = upsert_ai_series(conn, source, ARXIV_SERIES.info)
    months = list(iter_months(date(2021, 1, 1), date(2026, 9, 1)))
    insert_ai_observations(conn, series, [
        AIObservation(m, next_month(m), 4000 * 1.015**i, True) for i, m in enumerate(months)
    ])  # fmt: skip
    ask = TODAY - timedelta(days=1)
    pq, _, _ = upsert_platform_question(conn, PlatformQuestion(
        "polymarket", "m1", "Will OpenAI release GPT-7 by January 13?",
        opened_at=utc(date(2026, 6, 1)), closes_at=utc(ask + timedelta(days=90)),
        metadata={"closed_at": None},
    ))  # fmt: skip
    with conn.cursor() as cur:
        cur.execute("INSERT INTO platform_probabilities VALUES (%s, %s, 0.35, false)",
                    (pq, ask - timedelta(days=2)))  # fmt: skip
    get_or_create_question(conn, "cpi_yoy", date(2026, 11, 1), 1)  # an economy question
    conn.commit()
    job.run_ai_forecast_job(
        conn, TODAY, numeric_models=(job.NumericSpec("ai_random_walk", lambda s, h: AIRandomWalk()),),
        classifier=False,
    )  # fmt: skip

    @contextmanager
    def using() -> Iterator[Connection]:
        yield conn

    monkeypatch.setattr(db_connection, "connection", using)
    monkeypatch.setattr(cli, "today", lambda: TODAY)
    return conn


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    assert cli.main(list(argv)) == 0
    return capsys.readouterr().out


def question_id(db: Connection, where: str) -> int:
    with db.cursor() as cur:
        cur.execute(f"SELECT min(id) FROM questions WHERE {where}")
        return int(cur.fetchone()[0])  # type: ignore[index]


# --- help and routing ---------------------------------------------------------------------


def test_help_lists_the_ai_commands_and_domain(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    top = capsys.readouterr().out
    for command in ("ai-forecast", "ai-resolve", "review-postmortems", "--domain ai"):
        assert command in top
    with pytest.raises(SystemExit):
        cli.main(["forecast", "--help"])
    sub = capsys.readouterr().out
    assert "{economy,ai}" in sub and "arxiv" in sub and "binary" in sub


def test_the_cli_target_list_matches_the_ai_module() -> None:
    assert tuple(cli.AI_TARGET_NAMES) == tuple(cli_ai.AI_TARGET_NAMES)
    assert cli_ai.AI_TARGET_NAMES["arxiv"] == "ARXIV_AI_SUBMISSIONS"
    assert tuple(cli.AI_KINDS) == cli_ai.KINDS


def test_backtest_routes_by_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    from eco_prediction.ai.llm import comparison_report
    from eco_prediction.scripts import model_comparison

    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        comparison_report, "main", lambda a: seen.append(("ai", a)) or 0
    )
    monkeypatch.setattr(
        model_comparison, "main", lambda a: seen.append(("economy", a)) or 0
    )
    assert cli.main(["backtest", "--domain", "ai", "--out", "x"]) == 0
    assert cli.main(["backtest", "--domain=ai"]) == 0
    assert cli.main(["backtest", "--refresh"]) == 0
    assert seen == [("ai", ["--out", "x"]), ("ai", []), ("economy", ["--refresh"])]
    with pytest.raises(SystemExit):
        cli.main(["backtest", "--domain", "space"])


# --- forecast --------------------------------------------------------------------------------


def test_forecast_an_ai_series(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(capsys, "forecast", "--domain", "ai", "--target", "arxiv", "--horizon", "3",
              "--model", "ai_random_walk")  # fmt: skip
    assert "January 2027, 3 month(s) ahead" in out
    assert "Data through 2026-10-15" in out and "(September 2026)" in out
    assert "ai_random_walk" in out and "ai_lightgbm" not in out
    every = run(
        capsys, "forecast", "--domain", "ai", "--target", "arxiv", "--horizon", "1"
    )
    for model in (
        "ai_random_walk",
        "ai_linear_trend",
        "ai_arima",
        "ai_ets",
        "ai_lightgbm",
    ):
        assert model in every
    assert "drivers:" in every  # LightGBM's SHAP


def test_a_series_without_data_explains_each_model(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(
        capsys, "forecast", "--domain", "ai", "--target", "gpqa", "--horizon", "1"
    )
    assert "no values known" in out and "no EPOCH_GPQA_SOTA data known" in out


def test_forecast_a_binary_question(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    curated = question_id(db, "platform_question_id IS NOT NULL")
    out = run(capsys, "forecast", "--domain", "ai", "--target", "binary",
              "--question-id", str(curated))  # fmt: skip
    assert "Will OpenAI release GPT-7" in out
    assert "ai_market_price  0.350" in out and "ai_base_rate" in out
    threshold = question_id(db, "question_key LIKE 'ARXIV%>=%'")
    out = run(capsys, "forecast", "--domain", "ai", "--target", "binary",
              "--question-id", str(threshold), "--model", "ai_base_rate")  # fmt: skip
    assert "ai_base_rate" in out and "ai_threshold_classifier" not in out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--target", "binary"], "needs --question-id"),
        (["--target", "cpi"], "isn't an AI target"),
        (["--target", "arxiv", "--model", "nope"], "unknown model 'nope'"),
    ],
)
def test_ai_forecast_errors(db: Connection, argv: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        cli.main(["forecast", "--domain", "ai", *argv])


def test_a_numeric_question_id_points_to_the_numeric_form(db: Connection) -> None:
    numeric = question_id(db, "question_type = 'numeric_interval' AND domain = 'ai'")
    with pytest.raises(SystemExit, match="is numeric; use --target arxiv --horizon 1"):
        cli.main(
            [
                "forecast",
                "--domain",
                "ai",
                "--target",
                "binary",
                "--question-id",
                str(numeric),
            ]
        )


# --- inspect and scores -----------------------------------------------------------------------


def test_inspect_lists_ai_questions_with_filters(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(capsys, "inspect", "--domain", "ai", "--limit", "3")
    assert "ARXIV_AI_SUBMISSIONS" in out and "cpi_yoy" not in out
    assert "3 of " in out and "numeric 21" in out and "narrow with" in out
    curated = run(capsys, "inspect", "--domain", "ai", "--kind", "curated")
    assert "polymarket:m1" in curated and "ARXIV" not in curated
    # --open drops resolved questions: resolve the market, and it's gone.
    assert "polymarket:m1" in run(
        capsys, "inspect", "--domain", "ai", "--kind", "curated", "--open"
    )
    with db.cursor() as cur:
        cur.execute("UPDATE platform_questions SET resolved_at = %s, resolved_outcome = 'no'",
                    (utc(TODAY),))  # fmt: skip
    resolve_and_score_ai(db, TODAY + timedelta(days=1))
    db.commit()
    gone = run(capsys, "inspect", "--domain", "ai", "--kind", "curated", "--open")
    assert gone.strip() == "No AI questions match."


def test_inspect_one_ai_question_shows_its_key_and_text(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    curated = question_id(db, "platform_question_id IS NOT NULL")
    out = run(
        capsys, "inspect", "--question-id", str(curated)
    )  # works without --domain
    assert "Key: polymarket:m1" in out and "Text: Will OpenAI release GPT-7" in out
    assert "ai_market_price" in out


def test_ai_scores(db: Connection, capsys: pytest.CaptureFixture[str]) -> None:
    assert "No scored AI forecasts yet." in run(capsys, "scores", "--domain", "ai")
    series = ARXIV_SERIES.info.series_id
    with db.cursor() as cur:
        cur.execute("SELECT id FROM series WHERE series_id = %s", (series,))
        pk = cur.fetchone()[0]  # type: ignore[index]
    insert_ai_observations(
        db, pk, [AIObservation(date(2026, 11, 1), date(2026, 12, 2), 6000.0, False)]
    )
    resolve_and_score_ai(db, date(2026, 12, 10))
    db.commit()
    out = run(capsys, "scores", "--domain", "ai")
    assert (
        "numeric" in out and "ARXIV_AI_SUBMISSIONS" in out and "ai_random_walk" in out
    )


# --- the economy is unchanged -------------------------------------------------------------------


def test_economy_commands_are_unchanged_without_domain(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(capsys, "inspect")
    assert "cpi_yoy" in out and "ARXIV" not in out and "polymarket" not in out
    assert "No scored forecasts yet." in run(capsys, "scores")
    with pytest.raises(SystemExit, match="add --domain ai for AI targets"):
        cli.main(["forecast", "--target", "arxiv"])
    args, _ = cli.build_parser().parse_known_args(["forecast", "--target", "cpi"])
    assert args.domain == "economy" and args.horizon == 3


def test_ai_flags_are_ignored_by_economy_listing(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    # --kind/--open/--limit only shape the AI listing; the economy list is the same.
    assert run(capsys, "inspect") == run(capsys, "inspect", "--limit", "1")
