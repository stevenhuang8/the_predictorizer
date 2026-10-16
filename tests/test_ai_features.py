"""Tests for the AI feature pipeline (AI task 12).

The store is built by hand so every feature can be checked:
- ARXIV_AI_SUBMISSIONS 2024-01..2025-12 = 100, 110, ..., 330 (10 more each
  month), each month known on the 1st of the next (a reconstructed backfill),
  plus a live revision of 2025-11 to 999 on 2025-12-25.
- HF_NEW_MODELS = exactly twice arXiv, so their log changes correlate fully.
- EPOCH_GPQA_SOTA 2025-01..12: 50 x3, 60 x4, 72 x5 (jumps in April and August).
- EPOCH_SWEBENCH_SOTA 2024-12..2025-12: 40 x7, 45 x3, 50 x3 (two increases).
- EPOCH_ARC2_SOTA 2025-10..12: flat at 10.
GitHub, stars and compute have no data.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.features import (
    AI_FEATURE_SERIES,
    AIFeatureEngineer,
    build_ai_features,
)
from eco_prediction.ai.ingest_arxiv import SERIES as ARXIV
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.store import (
    AIObservation,
    ensure_source,
    insert_ai_observations,
    upsert_ai_series,
)
from eco_prediction.backtest.data import (
    TARGETS,
    LookaheadError,
    PointInTimeData,
    VintageStore,
)
from eco_prediction.db.forecasts import save_feature_snapshot
from eco_prediction.db.migrate import migrate

AS_OF = date(2026, 1, 15)


def monthly(start: date, values: list[float]) -> list[tuple[date, date, float]]:
    months = list(iter_months(start, date(2100, 1, 1)))[: len(values)]
    return [(m, next_month(m), v) for m, v in zip(months, values)]


def frame(rows: list[tuple[date, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])


ARXIV_VALUES = [100.0 + 10 * i for i in range(24)]  # 2024-01 .. 2025-12


def build_store() -> VintageStore:
    arxiv = monthly(date(2024, 1, 1), ARXIV_VALUES)
    arxiv.append((date(2025, 11, 1), date(2025, 12, 25), 999.0))  # live revision
    hf = monthly(date(2024, 1, 1), [2 * v for v in ARXIV_VALUES])
    gpqa = monthly(date(2025, 1, 1), [50.0] * 3 + [60.0] * 4 + [72.0] * 5)
    swe = monthly(date(2024, 12, 1), [40.0] * 7 + [45.0] * 3 + [50.0] * 3)
    arc2 = monthly(date(2025, 10, 1), [10.0] * 3)
    return VintageStore(
        {
            "ARXIV_AI_SUBMISSIONS": frame(arxiv),
            "HF_NEW_MODELS": frame(hf),
            "EPOCH_GPQA_SOTA": frame(gpqa),
            "EPOCH_SWEBENCH_SOTA": frame(swe),
            "EPOCH_ARC2_SOTA": frame(arc2),
        }
    )


@pytest.fixture
def features() -> dict[str, float | None]:
    return AIFeatureEngineer().build_features_as_of(build_store(), AS_OF)


# --- counts -----------------------------------------------------------------


def test_count_level_log_and_lags(features: dict[str, Any]) -> None:
    # Latest month 2025-12 = 330, but 2025-11 was revised to 999 on 2025-12-25.
    assert features["arxiv"] == 330
    assert features["arxiv_log"] == pytest.approx(math.log(330))
    assert features["arxiv_lag1"] == 999
    assert features["arxiv_lag3"] == 300
    assert features["arxiv_lag6"] == 270
    assert features["arxiv_lag12"] == 210
    assert features["arxiv_months_behind"] == 1


def test_count_growth_rates(features: dict[str, Any]) -> None:
    assert features["arxiv_growth_1m"] == pytest.approx((330 / 999 - 1) * 100)
    assert features["arxiv_growth_3m"] == pytest.approx(10.0)
    assert features["arxiv_growth_12m"] == pytest.approx((330 / 210 - 1) * 100)


def test_features_are_point_in_time() -> None:
    fe = AIFeatureEngineer()
    # On 2025-12-20, 2025-12 isn't known and 2025-11's revision isn't either.
    early = fe.build_features_as_of(build_store(), date(2025, 12, 20))
    assert early["arxiv"] == 320
    assert early["arxiv_lag1"] == 310
    assert early["arxiv_growth_1m"] == pytest.approx((320 / 310 - 1) * 100)


def test_stale_and_missing_series_give_none(features: dict[str, Any]) -> None:
    assert features["github"] is None and features["github_growth_12m"] is None
    assert features["flops"] is None and features["flops_months_since_change"] is None
    # Five months after the latest month is more than 3: stale.
    late = AIFeatureEngineer().build_features_as_of(build_store(), date(2026, 6, 1))
    assert late["arxiv"] is None and late["arxiv_log"] is None
    assert late["month_6"] == 1.0  # calendar features never go missing


def test_every_feature_name_is_always_present() -> None:
    empty = AIFeatureEngineer().build_features({}, AS_OF)
    full = AIFeatureEngineer().build_features_as_of(build_store(), AS_OF)
    assert empty.keys() == full.keys()
    assert all(
        v is None
        for k, v in empty.items()
        if not k.startswith(("month_", "quarter_", "trend"))
    )


# --- step series --------------------------------------------------------------


def test_step_series_features(features: dict[str, Any]) -> None:
    assert features["gpqa"] == 72
    assert features["gpqa_lag3"] == 72
    assert features["gpqa_lag6"] == 60
    assert features["gpqa_lag12"] is None  # 2024-12 is before the series
    assert features["gpqa_chg_1m"] == 0
    assert features["gpqa_chg_3m"] == 0
    assert features["gpqa_chg_12m"] is None
    assert features["gpqa_months_since_change"] == 4  # 2025-08 -> 2025-12
    assert features["gpqa_last_jump"] == 12
    assert features["gpqa_headroom"] == 28


def test_a_series_that_never_changed(features: dict[str, Any]) -> None:
    assert features["arc2_months_since_change"] == 2  # since its first month
    assert features["arc2_last_jump"] is None


def test_only_percent_series_have_headroom(features: dict[str, Any]) -> None:
    assert "flops_headroom" not in features
    assert {"gpqa_headroom", "swebench_headroom", "arc2_headroom"} <= features.keys()


# --- cross-series and calendar ------------------------------------------------


def test_cross_series_signals(features: dict[str, Any]) -> None:
    # HF is exactly 2x arXiv in every month up to 2025-10, but the 2025-11
    # revision of arXiv (999) breaks the pattern in the last two changes.
    assert -1 <= features["corr12_hf_arxiv"] < 1
    assert features["log_ratio_hf_arxiv"] == pytest.approx(math.log(2))
    assert features["corr12_github_arxiv"] is None
    assert features["log_ratio_github_arxiv"] is None
    # GPQA has no 2024-12 value, so only SWE-bench's two increases count.
    assert features["sota_increases_12m"] == 2


def test_perfectly_correlated_series() -> None:
    early = AIFeatureEngineer().build_features_as_of(build_store(), date(2025, 11, 5))
    assert early["corr12_hf_arxiv"] == pytest.approx(1.0)


def test_calendar(features: dict[str, Any]) -> None:
    assert features["month_1"] == 1.0 and features["month_2"] == 0.0
    assert features["quarter_1"] == 1.0 and features["quarter_4"] == 0.0
    assert features["trend_months"] == 192  # 2010-01 -> 2026-01


# --- targets, matrices and look-ahead ----------------------------------------


def test_target_aliases() -> None:
    fe = AIFeatureEngineer()
    f = fe.build_features_as_of(build_store(), AS_OF, "ARXIV_AI_SUBMISSIONS")
    assert f["target"] == f["arxiv"] == 330
    assert f["target_growth_12m"] == f["arxiv_growth_12m"]
    assert "target_corr12_hf_arxiv" not in f  # cross features aren't the series'
    swe = fe.build_features_as_of(build_store(), AS_OF, "EPOCH_SWEBENCH_SOTA")
    assert swe["target_months_since_change"] == 2  # 2025-10 -> 2025-12
    assert swe["target_headroom"] == 50
    with pytest.raises(ValueError, match="no AI features"):
        fe.build_features_as_of(build_store(), AS_OF, "CPIAUCSL")


def test_build_matrix() -> None:
    dates = [date(2025, 12, 20), AS_OF]
    X = AIFeatureEngineer().build_matrix(build_store(), dates)
    assert list(X.index) == [pd.Timestamp(d) for d in dates]
    assert X.loc[pd.Timestamp(AS_OF), "arxiv"] == 330
    assert (
        X.loc[pd.Timestamp(AS_OF), "github"] != X.loc[pd.Timestamp(AS_OF), "github"]
    )  # NaN


def test_a_view_refuses_dates_after_its_cutoff() -> None:
    view = PointInTimeData(build_store(), TARGETS["cpi_yoy"], cutoff=date(2025, 12, 31))
    fe = AIFeatureEngineer()
    assert fe.build_features_as_of(view, date(2025, 12, 31))["arxiv"] == 999
    with pytest.raises(LookaheadError):
        fe.build_features_as_of(view, AS_OF)


def test_ai_series_list() -> None:
    assert set(AI_FEATURE_SERIES) == {
        "ARXIV_AI_SUBMISSIONS", "HF_NEW_MODELS", "GITHUB_AI_ACTIVITY",
        "GITHUB_AI_STARS", "EPOCH_GPQA_SOTA", "EPOCH_SWEBENCH_SOTA",
        "EPOCH_ARC2_SOTA", "EPOCH_LARGEST_FLOPS",
    }  # fmt: skip


# --- database ---------------------------------------------------------------


def test_build_from_the_database_and_save_a_snapshot(conn: Connection) -> None:
    migrate(conn)
    source = ensure_source(conn, *ARXIV.source)
    pk = upsert_ai_series(conn, source, ARXIV.info)
    insert_ai_observations(
        conn,
        pk,
        [
            AIObservation(m, a, v, reconstructed=True)
            for m, a, v in monthly(date(2024, 1, 1), ARXIV_VALUES)
        ],
    )
    features = build_ai_features(conn, AS_OF, "ARXIV_AI_SUBMISSIONS")
    assert features["target"] == 330
    assert features["arxiv_growth_3m"] == pytest.approx((330 / 300 - 1) * 100)

    snapshot_id = save_feature_snapshot(conn, AS_OF, features)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT snapshot_date, features FROM feature_snapshots WHERE id = %s",
            (snapshot_id,),
        )
        snapshot_date, stored = cur.fetchone()  # type: ignore[misc]
    assert snapshot_date == AS_OF
    assert stored["arxiv"] == 330
    assert stored["github"] is None  # stored as JSON null
    assert stored.keys() == features.keys()
