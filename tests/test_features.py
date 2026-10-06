"""Tests for the feature pipeline, on synthetic series with hand-computed values."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from eco_prediction.backtest.data import (
    TARGETS,
    LookaheadError,
    PointInTimeData,
    VintageStore,
)
from eco_prediction.features.engineering import (
    FeatureEngineer,
    drop_sparse,
    monthly_mean,
    to_monthly,
)

AS_OF = date(2020, 3, 31)


def monthly(values: list[float], start: str) -> pd.Series:
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float
    )


def build(series_data: dict[str, pd.Series], as_of: date = AS_OF) -> dict:
    return FeatureEngineer().build_features(series_data, as_of)


def test_price_changes_match_hand_calculation() -> None:
    # PPI grows 1% a month, Jan 2018 to Feb 2020.
    ppi = monthly([100 * 1.01**i for i in range(26)], "2018-01-01")
    f = build({"PPIACO": ppi})
    assert f["ppi_mom"] == pytest.approx(1.0)
    assert f["ppi_3m"] == pytest.approx(3.0301)
    assert f["ppi_yoy"] == pytest.approx((1.01**12 - 1) * 100)


def test_short_history_gives_none_not_an_error() -> None:
    f = build({"PPIACO": monthly([100.0, 101.0, 102.0], "2019-12-01")})
    assert f["ppi_mom"] == pytest.approx(100 * (102 / 101 - 1))
    assert f["ppi_3m"] is None
    assert f["ppi_yoy"] is None


def test_monthly_mean_counts_only_months_ended_by_as_of() -> None:
    days = pd.date_range("2020-01-01", "2020-02-14", freq="D")
    oil = pd.Series([10.0 if d.month == 1 else 20.0 for d in days], index=days)
    mid_feb = monthly_mean(oil, date(2020, 2, 14))
    assert mid_feb.index.tolist() == [pd.Timestamp("2020-01-01")]
    assert mid_feb.iloc[0] == 10.0
    assert monthly_mean(oil, date(2020, 1, 31)).iloc[-1] == 10.0
    assert monthly_mean(oil, date(2020, 1, 30)).empty


def test_oil_and_gas_changes_use_monthly_averages() -> None:
    days = pd.bdate_range("2019-01-01", "2020-03-31")
    # Each month's price is constant except one spike day, which the mean absorbs.
    level = {
        m: 50.0 + i
        for i, m in enumerate(pd.period_range("2019-01", "2020-03", freq="M"))
    }
    oil = pd.Series([level[d.to_period("M")] for d in days], index=days)
    oil.loc["2020-03-02"] += 22 * 10  # March has 22 business days: mean rises by 10
    f = build({"DCOILWTICO": oil})
    mar, feb, dec, mar_prev = 64.0 + 10, 63.0, 61.0, 52.0
    assert f["oil_mom"] == pytest.approx((mar / feb - 1) * 100)
    assert f["oil_3m"] == pytest.approx((mar / dec - 1) * 100)
    assert f["oil_yoy"] == pytest.approx((mar / mar_prev - 1) * 100)


def test_target_features_describe_the_latest_published_period() -> None:
    # Known at the end of March: CPI and unemployment through February.
    cpi = monthly([100 + i for i in range(26)], "2018-01-01")  # Jan 2018..Feb 2020
    unrate = monthly([4.0, 4.2, 3.9, 3.6, 3.5], "2019-10-01")  # Oct 2019..Feb 2020
    f = build({"CPIAUCSL": cpi, "UNRATE": unrate})
    assert f["cpi_yoy"] == pytest.approx((125 / 113 - 1) * 100)  # Feb 2020 vs Feb 2019
    assert f["cpi_yoy_lag1"] == pytest.approx((124 / 112 - 1) * 100)
    assert f["cpi_yoy_lag3"] == pytest.approx((122 / 110 - 1) * 100)
    assert f["cpi_mom"] == pytest.approx((125 / 124 - 1) * 100)
    assert f["unrate"] == 3.5
    assert f["unrate_diff1"] == pytest.approx(-0.1)
    assert f["unrate_ma3"] == pytest.approx((3.9 + 3.6 + 3.5) / 3)
    assert [f["unrate_lag1"], f["unrate_lag2"], f["unrate_lag3"]] == [3.6, 3.9, 4.2]


@pytest.mark.parametrize(
    ("last", "expected"), [("2019-12-01", 3.5), ("2019-11-01", None)]
)
def test_series_older_than_max_stale_months_gives_none(
    last: str, expected: float | None
) -> None:
    unrate = monthly([3.5] * 12, "2019-01-01")
    f = build({"UNRATE": unrate[unrate.index <= pd.Timestamp(last)]})
    assert f["unrate"] == expected


def test_one_missing_month_is_forward_filled_two_are_not() -> None:
    cpi = monthly([100 + i for i in range(26)], "2018-01-01")
    one_gap = cpi.drop(pd.Timestamp("2020-01-01"))
    assert build({"CPIAUCSL": one_gap})["cpi_mom"] == pytest.approx(
        (125 / 123 - 1) * 100
    )
    two_gaps = one_gap.drop(pd.Timestamp("2019-12-01"))
    assert build({"CPIAUCSL": two_gaps})["cpi_mom"] is None
    filled = to_monthly(two_gaps, AS_OF)
    assert filled["2019-12-01"] == 122 and pd.isna(filled["2020-01-01"])


def test_value_published_as_missing_is_forward_filled() -> None:
    cpi = monthly([100 + i for i in range(26)], "2018-01-01")
    cpi["2020-01-01"] = float("nan")
    assert build({"CPIAUCSL": cpi})["cpi_mom"] == pytest.approx((125 / 123 - 1) * 100)


def test_missing_series_give_none_and_every_name_is_present() -> None:
    full = build(
        {
            "PPIACO": monthly([100 * 1.01**i for i in range(26)], "2018-01-01"),
            "UNRATE": monthly([3.5] * 26, "2018-01-01"),
        }
    )
    empty = build({})
    assert list(empty) == list(full)
    calendar = {k for k in empty if k.startswith("month_") or k == "trend_months"}
    assert all(empty[k] is None for k in empty if k not in calendar)


def test_claims_level_four_week_average_and_yoy() -> None:
    weeks = pd.date_range("2019-01-05", "2020-03-28", freq="W-SAT")
    recent = [250_000.0, 300_000.0, 350_000.0, 400_000.0]
    claims = pd.Series([200_000.0] * (len(weeks) - 4) + recent, index=weeks)
    f = build({"ICSA": claims})
    assert f["claims"] == 400_000
    assert f["claims_4wk"] == 325_000
    assert f["claims_4wk_yoy"] == pytest.approx(62.5)

    gappy = build({"ICSA": claims.drop(weeks[-2])})
    assert gappy["claims"] == 400_000
    assert gappy["claims_4wk"] is None and gappy["claims_4wk_yoy"] is None


def test_spreads_use_the_latest_daily_value_unless_stale() -> None:
    days = pd.bdate_range("2020-01-01", "2020-03-27")
    spread = pd.Series(range(len(days)), index=days, dtype=float)
    assert build({"T10Y2Y": spread})["t10y2y"] == len(days) - 1
    assert build({"T10Y2Y": spread}, date(2020, 4, 10))["t10y2y"] == len(days) - 1
    assert build({"T10Y2Y": spread}, date(2020, 4, 11))["t10y2y"] is None


def test_calendar_features() -> None:
    f = build({}, date(1991, 3, 31))
    assert [f[f"month_{m}"] for m in range(1, 13)] == [0, 0, 1] + [0] * 9
    assert f["trend_months"] == 14


def test_periods_after_as_of_are_ignored() -> None:
    cpi = monthly([100 + i for i in range(30)], "2018-01-01")  # runs to Jun 2020
    days = pd.bdate_range("2019-01-01", "2020-06-30")
    oil = pd.Series(range(len(days)), index=days, dtype=float)
    data = {"CPIAUCSL": cpi, "DCOILWTICO": oil, "T10Y2Y": oil}
    cut = {k: v[v.index <= pd.Timestamp(AS_OF)] for k, v in data.items()}
    assert build(data) == build(cut)


def revised_unrate() -> pd.DataFrame:
    """Period M first printed on the 15th of M+1, revised +0.5 on the 15th of M+2."""
    rows = []
    for i, period in enumerate(pd.date_range("2019-01-01", "2020-06-01", freq="MS")):
        first = 4.0 + 0.1 * i
        rows.append((period, period + pd.DateOffset(months=1, days=14), first))
        rows.append((period, period + pd.DateOffset(months=2, days=14), first + 0.5))
    return pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])


def test_matrix_rows_use_the_vintages_known_on_each_date() -> None:
    store = VintageStore({"UNRATE": revised_unrate()})
    matrix = FeatureEngineer().build_matrix(
        store, [date(2020, 4, 30), date(2020, 3, 31)]
    )
    assert matrix.index.tolist() == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-04-30"),
    ]
    # Mar 31: Feb 2020 (i=13) first print; Jan revised. Apr 30: Mar first, Feb revised.
    assert matrix["unrate"].tolist() == pytest.approx([5.3, 5.4])
    assert matrix["unrate_lag1"].tolist() == pytest.approx([5.2 + 0.5, 5.3 + 0.5])
    assert matrix["oil_yoy"].isna().all()


def test_view_source_refuses_dates_after_its_cutoff() -> None:
    store = VintageStore({"UNRATE": revised_unrate()})
    view = PointInTimeData(store, TARGETS["unemployment"], cutoff=date(2020, 3, 31))
    fe = FeatureEngineer()
    assert fe.build_features_as_of(view, date(2020, 3, 31))["unrate"] == pytest.approx(
        5.3
    )
    with pytest.raises(LookaheadError):
        fe.build_matrix(view, [date(2020, 3, 31), date(2020, 4, 1)])


def test_rows_are_built_once_per_store(monkeypatch: pytest.MonkeyPatch) -> None:
    fe = FeatureEngineer()
    calls: list[date] = []
    real = fe.series_as_of

    def counting(source: VintageStore, as_of: date) -> dict[str, pd.Series]:
        calls.append(as_of)
        return real(source, as_of)

    monkeypatch.setattr(fe, "series_as_of", counting)
    store = VintageStore({"UNRATE": revised_unrate()})
    first = fe.build_features_as_of(store, AS_OF)
    first["unrate"] = -1.0  # callers get a copy
    assert fe.build_features_as_of(store, AS_OF)["unrate"] == pytest.approx(5.3)
    assert calls == [AS_OF]
    fe.build_features_as_of(VintageStore({"UNRATE": revised_unrate()}), AS_OF)
    assert calls == [AS_OF, AS_OF]


def test_drop_sparse() -> None:
    matrix = pd.DataFrame(
        {
            "full": [1.0, 2.0, 3.0, 4.0],
            "half": [1.0, None, 3.0, None],
            "sparse": [None, None, None, 1.0],
        }
    )
    kept, dropped = drop_sparse(matrix, max_missing=0.5)
    assert kept.columns.tolist() == ["full", "half"]
    assert dropped == ["sparse"]
    with pytest.raises(ValueError):
        drop_sparse(matrix, max_missing=1.5)
