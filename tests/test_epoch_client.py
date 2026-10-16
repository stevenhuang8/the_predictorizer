"""Tests for the Epoch AI client and its step-series ingestion.

Client tests serve small in-memory copies of Epoch's files offline; ingestion
tests use a throwaway database (skipped if Postgres isn't reachable).
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from datetime import date
from typing import Any

import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import ingest_epoch
from eco_prediction.ai.ingest_epoch import SERIES, EpochCounter
from eco_prediction.ai.monthly_ingest import ingest_backfill, ingest_live
from eco_prediction.ai.months import MonthlyCount
from eco_prediction.ai.sources.base_client import SourceAPIError
from eco_prediction.ai.sources.epoch_client import (
    ARC_AGI_2,
    BENCHMARK_ZIP,
    GPQA_DIAMOND,
    MODELS_CSV,
    EpochClient,
    Result,
    running_max,
)
from eco_prediction.data.fred_client import RateLimiter
from eco_prediction.db.migrate import migrate

JUN, JUL, AUG, SEP, OCT = (date(2026, m, 1) for m in (6, 7, 8, 9, 10))
TODAY = date(2026, 10, 14)

BENCH_HEADER = "Model version,mean_score,Best score (across scorers),Release date,id\n"
GPQA = BENCH_HEADER + (
    "model-a,0.70,0.72,2026-06-10,1\n"
    "model-b,0.80,0.81,2026-08-31,2\n"  # last day of August counts for August
    "model-c,0.75,0.76,2026-09-02,3\n"  # not a record
    "model-d,0.90,0.95,,4\n"  # no release date: skipped
    "model-e,,,2026-09-20,5\n"  # no score: skipped
)
ARC2 = "Model version,Score,Release date\nold-model,0.0,2026-06-01\nnew-model,0.25,2026-08-15\n"
MODELS = (
    "Model,Publication date,Training compute (FLOP),Confidence\n"
    "Small,2026-06-05,1e24,Confident\n"
    "Big,2026-08-20,3.16e26,Speculative\n"
    "Undisclosed,2026-09-01,,Unknown\n"
)


def zip_of(**files: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = content.decode("latin-1")[:500]


class FakeSession:
    """Serves Epoch's files by name and records what was fetched."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.fetched: list[str] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        name = url.rsplit("/", 1)[-1]
        self.fetched.append(name)
        if name not in self.files:
            return FakeResponse(b"not found", 404)
        return FakeResponse(self.files[name])

    def close(self) -> None:
        self.closed = True


def client_with(
    benchmarks: dict[str, str] | None = None, models: str = MODELS
) -> tuple[EpochClient, FakeSession]:
    files = {
        BENCHMARK_ZIP: zip_of(
            **(
                benchmarks
                if benchmarks is not None
                else {
                    "gpqa_diamond.csv": GPQA,
                    "arc_agi_2_external.csv": ARC2,
                }
            )
        ),
        MODELS_CSV: models.encode(),
    }
    session = FakeSession(files)
    client = EpochClient(
        session=session,  # type: ignore[arg-type]
        rate_limiter=RateLimiter(100, 1.0),
        sleep=lambda s: None,
    )
    return client, session


# --- parsing ----------------------------------------------------------------


def test_benchmark_scores_use_the_headline_column_as_percent() -> None:
    client, _ = client_with()

    results = client.benchmark_results(GPQA_DIAMOND)

    assert [(r.model, r.released, round(r.value, 6)) for r in results] == [
        ("model-a", date(2026, 6, 10), 72.0),
        ("model-b", date(2026, 8, 31), 81.0),
        ("model-c", date(2026, 9, 2), 76.0),
    ]


def test_the_zip_is_downloaded_once_for_every_benchmark() -> None:
    client, session = client_with()

    client.benchmark_results(GPQA_DIAMOND)
    arc = client.benchmark_results(ARC_AGI_2)
    client.training_runs()
    client.training_runs()

    assert [r.value for r in arc] == [0.0, 25.0]
    assert session.fetched == [BENCHMARK_ZIP, MODELS_CSV]


def test_training_compute_is_log10_flop_and_blank_estimates_are_skipped() -> None:
    client, _ = client_with()

    runs = client.training_runs()

    assert [(r.model, round(r.value, 4)) for r in runs] == [
        ("Small", 24.0),
        ("Big", round(math.log10(3.16e26), 4)),
    ]


@pytest.mark.parametrize(
    ("benchmarks", "message"),
    [
        ({"other.csv": GPQA}, "can't read gpqa_diamond.csv"),
        (
            {"gpqa_diamond.csv": "Model version,Release date\nx,2026-01-01\n"},
            "missing column(s) ['Best score (across scorers)']",
        ),
        (
            {"gpqa_diamond.csv": BENCH_HEADER + "x,0.5,0.5,June 2026,1\n"},
            "bad row for 'x'",
        ),
        (
            {"gpqa_diamond.csv": BENCH_HEADER + "x,0.5,n/a,2026-06-01,1\n"},
            "bad row for 'x'",
        ),
        ({"gpqa_diamond.csv": BENCH_HEADER}, "no usable rows"),
    ],
)
def test_format_changes_raise(benchmarks: dict[str, str], message: str) -> None:
    client, _ = client_with(benchmarks)

    with pytest.raises(SourceAPIError, match=re.escape(message)):
        client.benchmark_results(GPQA_DIAMOND)


def test_a_file_that_is_not_a_zip_raises() -> None:
    client, session = client_with()
    session.files[BENCHMARK_ZIP] = b"<html>maintenance</html>"

    with pytest.raises(SourceAPIError, match="can't read"):
        client.benchmark_results(GPQA_DIAMOND)


def test_zero_training_compute_raises() -> None:
    client, _ = client_with(models=MODELS + "Broken,2026-09-01,0,Likely\n")

    with pytest.raises(SourceAPIError, match="training compute 0"):
        client.training_runs()


def test_missing_file_raises_with_the_status() -> None:
    client, session = client_with()
    del session.files[MODELS_CSV]

    with pytest.raises(SourceAPIError) as exc:
        client.training_runs()
    assert exc.value.status_code == 404


# --- running max --------------------------------------------------------------


def test_running_max_steps_only_at_records() -> None:
    results = [
        Result(date(2026, 9, 2), 76.0, "c"),  # unsorted input
        Result(date(2026, 6, 10), 72.0, "a"),
        Result(date(2026, 8, 31), 81.0, "b"),
    ]

    assert running_max(results, [JUN, JUL, AUG, SEP, OCT]) == [
        MonthlyCount(JUN, 72.0),
        MonthlyCount(JUL, 72.0),
        MonthlyCount(AUG, 81.0),
        MonthlyCount(SEP, 81.0),  # c is below the record
        MonthlyCount(OCT, 81.0),
    ]


def test_running_max_leaves_out_months_before_the_first_release() -> None:
    results = [Result(date(2026, 8, 1), 10.0, "a")]

    assert running_max(results, [JUN, JUL, AUG]) == [MonthlyCount(AUG, 10.0)]


def test_running_max_includes_models_released_before_the_first_month() -> None:
    results = [Result(date(2020, 1, 1), 50.0, "old"), Result(SEP, 40.0, "new")]

    assert running_max(results, [SEP]) == [MonthlyCount(SEP, 50.0)]


def test_running_max_rounds_to_4_decimals() -> None:
    results = [Result(JUN, 96.40151515151515, "a")]

    assert running_max(results, [JUN]) == [MonthlyCount(JUN, 96.4015)]


# --- ingestion ----------------------------------------------------------------


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def stored(conn: Connection, series_id: str) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.observed_at, o.as_of, o.value::float8, o.reconstructed
            FROM observations o JOIN series s ON s.id = o.series_id
            WHERE s.series_id = %s
            ORDER BY o.observed_at, o.as_of
            """,
            (series_id,),
        )
        return cur.fetchall()


def test_backfill_then_a_late_evaluation_adds_live_vintages(
    migrated: Connection,
) -> None:
    gpqa = SERIES["gpqa"]
    client, _ = client_with()
    result = ingest_backfill(
        migrated, gpqa.series, EpochCounter(client, gpqa), JUN, SEP
    )
    migrated.commit()
    assert (result.fetched, result.inserted) == (4, 4)

    # Epoch later evaluates a July model that beats everything so far.
    late = GPQA + "model-f,0.85,0.88,2026-07-04,6\n"
    client, _ = client_with({"gpqa_diamond.csv": late})
    live = ingest_live(migrated, gpqa.series, EpochCounter(client, gpqa), TODAY)
    # July to September change; June is unchanged and only confirmed live.
    assert (live.fetched, live.inserted) == (4, 4)

    assert stored(migrated, "EPOCH_GPQA_SOTA") == [
        (JUN, JUL, 72.0, True),
        (JUN, TODAY, 72.0, False),
        (JUL, AUG, 72.0, True),
        (JUL, TODAY, 88.0, False),
        (AUG, SEP, 81.0, True),
        (AUG, TODAY, 88.0, False),
        (SEP, OCT, 81.0, True),
        (SEP, TODAY, 88.0, False),
    ]
    with migrated.cursor() as cur:
        cur.execute(
            """
            SELECT s.domain, s.units, src.name
            FROM series s JOIN sources src ON src.id = s.source_id
            WHERE s.series_id = 'EPOCH_GPQA_SOTA'
            """
        )
        assert cur.fetchone() == ("ai", "Percent", "Epoch AI")


def test_a_zero_score_is_stored_but_zero_compute_is_not(
    migrated: Connection,
) -> None:
    arc = SERIES["arc_agi_2"]
    client, _ = client_with()
    ingest_backfill(migrated, arc.series, EpochCounter(client, arc), JUN, JUL)
    assert stored(migrated, "EPOCH_ARC2_SOTA") == [
        (JUN, JUL, 0.0, True),
        (JUL, AUG, 0.0, True),
    ]

    compute = SERIES["compute"]
    with pytest.raises(SourceAPIError, match="refusing to store"):
        ingest_backfill(
            migrated,
            compute.series,
            FakeCounter({JUN: 0}),  # type: ignore[arg-type]
            JUN,
            JUN,
        )


class FakeCounter:
    def __init__(self, values: dict[date, float]) -> None:
        self.values = values

    def count_months(self, months: list[date]) -> list[MonthlyCount]:
        return [MonthlyCount(m, self.values[m]) for m in months]

    def close(self) -> None:
        pass


def test_all_four_series_share_one_source(migrated: Connection) -> None:
    client, _ = client_with(
        {
            "gpqa_diamond.csv": GPQA,
            "swe_bench_verified.csv": GPQA,
            "arc_agi_2_external.csv": ARC2,
        }
    )
    for s in SERIES.values():
        ingest_live(migrated, s.series, EpochCounter(client, s), TODAY)

    with migrated.cursor() as cur:
        cur.execute(
            """
            SELECT s.series_id, src.name, s.domain
            FROM series s JOIN sources src ON src.id = s.source_id
            WHERE src.name = 'Epoch AI'
            ORDER BY s.series_id
            """
        )
        assert cur.fetchall() == [
            ("EPOCH_ARC2_SOTA", "Epoch AI", "ai"),
            ("EPOCH_GPQA_SOTA", "Epoch AI", "ai"),
            ("EPOCH_LARGEST_FLOPS", "Epoch AI", "ai"),
            ("EPOCH_SWEBENCH_SOTA", "Epoch AI", "ai"),
        ]
    assert stored(migrated, "EPOCH_LARGEST_FLOPS")[-1] == (
        SEP,
        TODAY,
        round(math.log10(3.16e26), 4),
        False,
    )


# --- command line -------------------------------------------------------------


def test_cli_runs_one_series_and_closes_the_client(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, session = client_with()

    code = ingest_epoch.main(
        ["compute", "--dry-run", "--months", "2"],
        client_factory=lambda: client,
        today=lambda: TODAY,
    )

    assert code == 0
    assert session.closed
    assert session.fetched == [MODELS_CSV]
    assert capsys.readouterr().out.splitlines() == [
        "2026-08    26.50  as_of=2026-10-14  live",
        "2026-09    26.50  as_of=2026-10-14  live",
    ]


def test_cli_dry_run_of_all_series_labels_each(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, session = client_with(
        {
            "gpqa_diamond.csv": GPQA,
            "swe_bench_verified.csv": GPQA,
            "arc_agi_2_external.csv": ARC2,
        }
    )

    code = ingest_epoch.main(
        ["--dry-run", "--months", "1"],
        client_factory=lambda: client,
        today=lambda: TODAY,
    )

    assert code == 0
    assert session.fetched == [BENCHMARK_ZIP, MODELS_CSV]
    assert capsys.readouterr().out.splitlines() == [
        "== EPOCH_GPQA_SOTA",
        "2026-09    81.00  as_of=2026-10-14  live",
        "== EPOCH_SWEBENCH_SOTA",
        "2026-09    81.00  as_of=2026-10-14  live",
        "== EPOCH_ARC2_SOTA",
        "2026-09    25.00  as_of=2026-10-14  live",
        "== EPOCH_LARGEST_FLOPS",
        "2026-09    26.50  as_of=2026-10-14  live",
    ]


def test_cli_reports_a_download_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, session = client_with()
    del session.files[MODELS_CSV]

    code = ingest_epoch.main(
        ["compute", "--dry-run"], client_factory=lambda: client, today=lambda: TODAY
    )

    assert code == 1
    assert "notable_ai_models.csv failed (404)" in capsys.readouterr().err
