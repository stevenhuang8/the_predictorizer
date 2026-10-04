"""Tests for the walk-forward harness and point-in-time data views.

The synthetic series are published like FRED's monthly releases: period M's
first print appears on the 15th of M+1 and is revised by +0.5 on the 15th of
M+2. Leakage would show up as a model seeing a period, or a revision, before
its publish date.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import date

import pandas as pd
import pytest

from eco_prediction.backtest.data import (
    TARGETS,
    PointInTimeData,
    VintageStore,
)
from eco_prediction.backtest.harness import (
    Forecast,
    WalkForwardBacktest,
    add_months,
    months_between,
    summarize,
)
from eco_prediction.db import connection as db
from eco_prediction.db.migrate import migrate

REVISION = 0.5


def first_print(period: date) -> float:
    """Unemployment-like level that rises 0.1 a month from 4.0 in Jan 2018."""
    return round(4.0 + 0.1 * months_between(date(2018, 1, 1), period), 6)


def published(period: date, months_later: int) -> date:
    return add_months(period, months_later).replace(day=15)


def vintages(
    first: date = date(2018, 1, 1), last: date = date(2023, 12, 1)
) -> pd.DataFrame:
    rows = []
    period = first
    while period <= last:
        rows.append((period, published(period, 1), first_print(period)))
        rows.append((period, published(period, 2), first_print(period) + REVISION))
        period = add_months(period, 1)
    return pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])


def store() -> VintageStore:
    frame = vintages()
    return VintageStore({"UNRATE": frame, "CPIAUCSL": frame})


class LastValue:
    """Random walk on the target; records what it was shown."""

    def __init__(self, width: float = 1.0) -> None:
        self.width = width
        self.fits: list[PointInTimeData] = []
        self.fit_histories: list[pd.Series] = []
        self.seen: list[tuple[date, pd.Series]] = []

    def fit(self, data: PointInTimeData) -> None:
        self.fits.append(data)
        self.fit_histories.append(data.target_history())

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        history = data.target_history()
        self.seen.append((data.cutoff, history))
        last = float(history.iloc[-1])
        return Forecast(last, last - self.width, last + self.width)


def backtest(**kwargs: object) -> WalkForwardBacktest:
    params: dict[str, object] = {
        "start_date": date(2020, 1, 1),
        "end_date": date(2020, 12, 1),
        "store": store(),
        **kwargs,
    }
    return WalkForwardBacktest(**params)  # type: ignore[arg-type]


def test_forecast_dates_are_month_starts_in_range() -> None:
    bt = backtest(start_date=date(2020, 1, 15), end_date=date(2020, 4, 1))
    assert bt.get_forecast_dates() == [
        date(2020, 2, 1),
        date(2020, 3, 1),
        date(2020, 4, 1),
    ]


def test_models_never_see_data_published_on_or_after_forecast_date() -> None:
    model = LastValue()
    results = backtest().run(model, "unemployment", horizon_months=1)

    assert (results["max_as_of_seen"] < results["forecast_date"]).all()
    for cutoff, history in model.seen:
        # Period M is published on the 15th of M+1; on cutoff day the latest
        # visible period is the one released on or before it.
        last_period = history.index[-1].date()
        assert published(last_period, 1) <= cutoff
        assert published(add_months(last_period, 1), 1) > cutoff


def test_views_return_each_period_as_first_printed_until_revised() -> None:
    view = PointInTimeData(store(), TARGETS["unemployment"], date(2020, 3, 20))
    history = view.target_history()

    # Feb 2020 was first printed on 03-15; Jan 2020's revision came out the same day.
    assert history.index[-1] == pd.Timestamp("2020-02-01")
    assert history.iloc[-1] == pytest.approx(first_print(date(2020, 2, 1)))
    assert history.iloc[-2] == pytest.approx(first_print(date(2020, 1, 1)) + REVISION)
    assert view.max_as_of_seen == date(2020, 3, 15)


def test_revision_published_on_forecast_date_is_not_seen() -> None:
    # Forecast on 2020-03-15: data as of 03-14, so Feb isn't out and Jan is unrevised.
    view = PointInTimeData(store(), TARGETS["unemployment"], date(2020, 3, 14))
    history = view.target_history()
    assert history.index[-1] == pd.Timestamp("2020-01-01")
    assert history.iloc[-1] == pytest.approx(first_print(date(2020, 1, 1)))


def test_retrains_on_schedule_with_data_known_at_the_time() -> None:
    model = LastValue()
    results = backtest(retrain_frequency="quarterly").run(model, "unemployment", 1)

    cutoffs = [view.cutoff for view in model.fits]
    assert cutoffs == [
        date(2019, 12, 31),
        date(2020, 3, 31),
        date(2020, 6, 30),
        date(2020, 9, 30),
    ]
    assert results["trained_as_of"].tolist()[:4] == [date(2019, 12, 31)] * 3 + [
        date(2020, 3, 31)
    ]
    for view, history in zip(model.fits, model.fit_histories, strict=True):
        assert view.max_as_of_seen is not None and view.max_as_of_seen <= view.cutoff
        assert published(history.index[-1].date(), 1) <= view.cutoff


def test_expanding_window_trains_on_all_history_or_from_train_start() -> None:
    model = LastValue()
    backtest(retrain_frequency="annually").run(model, "unemployment", 1)
    assert model.fit_histories[0].index[0] == pd.Timestamp("2018-01-01")

    model = LastValue()
    backtest(retrain_frequency="annually", train_start=date(2019, 1, 1)).run(
        model, "unemployment", 1
    )
    assert model.fit_histories[0].index[0] == pd.Timestamp("2019-01-01")


def test_rolling_window_trains_on_last_n_months_only() -> None:
    model = LastValue()
    backtest(window="rolling", window_months=12).run(model, "unemployment", 1)

    first, second = model.fit_histories[:2]
    # Fit on 2019-12-31: Jan-Dec 2019 periods allowed, Nov 2019 the latest published.
    assert (first.index[0], first.index[-1]) == (
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2019-11-01"),
    )
    assert second.index[0] == pd.Timestamp("2019-04-01")


def test_scores_against_first_release_by_default() -> None:
    results = backtest().run(LastValue(), "unemployment", horizon_months=3)
    row = results.iloc[0]

    # 2020-01-01: latest known is Nov 2019's first print; target is Apr 2020.
    assert row["target_period"] == date(2020, 4, 1)
    assert row["prediction"] == pytest.approx(first_print(date(2019, 11, 1)))
    assert row["actual"] == pytest.approx(first_print(date(2020, 4, 1)))
    assert row["actual_as_of"] == date(2020, 5, 15)
    assert row["error"] == pytest.approx(-0.5)  # five months of +0.1
    assert row["squared_error"] == pytest.approx(0.25)
    assert bool(row["in_interval"]) is True


def test_latest_resolution_uses_revised_value() -> None:
    results = backtest(resolve_with="latest").run(LastValue(), "unemployment", 3)
    row = results.iloc[0]
    assert row["actual"] == pytest.approx(first_print(date(2020, 4, 1)) + REVISION)
    assert row["actual_as_of"] == date(2024, 2, 15)  # latest vintage in the store


def test_cpi_yoy_resolves_against_year_earlier_value_known_at_release() -> None:
    bt = backtest()
    actual, as_of = bt.get_actual_value(TARGETS["cpi_yoy"], date(2020, 4, 1))

    # At Apr 2020's first release, Apr 2019 had long been revised.
    then = first_print(date(2020, 4, 1))
    year_before = first_print(date(2019, 4, 1)) + REVISION
    assert as_of == date(2020, 5, 15)
    assert actual == pytest.approx((then / year_before - 1) * 100)


def test_unpublished_targets_are_left_unscored() -> None:
    bt = backtest(start_date=date(2023, 9, 1), end_date=date(2023, 12, 1))
    results = bt.run(LastValue(), "unemployment", horizon_months=1)

    # Targets Oct 2023 - Jan 2024; Dec 2023 is the last period in the data.
    assert results["actual"].isna().tolist() == [False, False, False, True]
    assert results.iloc[-1][["error", "squared_error"]].isna().all()
    summary = summarize(results)
    assert (summary["n_forecasts"], summary["n_resolved"]) == (4, 3)


def test_summary_metrics() -> None:
    results = backtest().run(LastValue(width=0.25), "unemployment", horizon_months=1)
    summary = summarize(results)

    # Every forecast is 3 months stale on a +0.1/month trend: error -0.3.
    assert summary["n_resolved"] == 12
    assert summary["rmse"] == pytest.approx(0.3)
    assert summary["mae"] == pytest.approx(0.3)
    assert summary["bias"] == pytest.approx(-0.3)
    assert summary["interval_coverage"] == 0.0


def test_point_only_forecasts_have_no_coverage() -> None:
    class PointOnly(LastValue):
        def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
            return Forecast(super().predict(data, target_period).point)

    results = backtest().run(PointOnly(), "unemployment", 1)
    assert results["in_interval"].isna().all()
    assert summarize(results)["interval_coverage"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"end_date": date(2019, 1, 1)},
        {"retrain_frequency": "weekly"},
        {"window": "rolling"},
        {"window_months": 12},
        {"resolve_with": "final"},
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        backtest(**kwargs)


def test_horizon_must_be_positive() -> None:
    with pytest.raises(ValueError):
        backtest().run(LastValue(), "unemployment", 0)


def test_missing_first_print_is_not_a_release() -> None:
    frame = pd.DataFrame(
        [
            (date(2020, 1, 1), date(2020, 2, 7), math.nan),  # reported missing
            (date(2020, 1, 1), date(2020, 3, 6), 3.6),
        ],
        columns=["observed_at", "as_of", "value"],
    )
    s = VintageStore({"UNRATE": frame})
    assert s.first_release("UNRATE", date(2020, 1, 1)) == date(2020, 3, 6)


@pytest.fixture
def pool(test_dsn: str) -> Iterator[None]:
    db.init_pool(test_dsn, maxconn=2)
    with db.connection() as conn:
        migrate(conn)
    try:
        yield
    finally:
        db.close_pool()


def test_store_loads_vintages_from_database(pool: None) -> None:
    frame = vintages(date(2019, 1, 1), date(2019, 6, 1))
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO sources (name) VALUES ('FRED') RETURNING id")
        source = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO series (source_id, series_id) VALUES (%s, 'UNRATE') RETURNING id",
            (source,),
        )
        series = cur.fetchone()[0]  # type: ignore[index]
        cur.executemany(
            "INSERT INTO observations (series_id, observed_at, as_of, value) VALUES (%s, %s, %s, %s)",
            [(series, *row) for row in frame.itertuples(index=False)],
        )

    s = VintageStore.from_db()
    view = PointInTimeData(s, TARGETS["unemployment"], date(2019, 4, 1))
    history = view.target_history()
    assert history.index[-1] == pd.Timestamp("2019-02-01")
    assert history.iloc[-1] == pytest.approx(first_print(date(2019, 2, 1)))
    assert view.max_as_of_seen == date(2019, 3, 15)
    assert s.first_release("UNRATE", date(2019, 6, 1)) == date(2019, 7, 15)
    assert s.first_release("UNRATE", date(2019, 7, 1)) is None
