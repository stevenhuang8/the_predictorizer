"""Tests for the FOMC calendar, decision labels, forecaster and backtest.

The synthetic store uses the real meeting calendar from 2009. Decisions are
random, the target moves 25 bp the day after each non-hold decision, and the
3-month bill trades at the target plus 0.2 x the next decision (-1, 0, 1), so
markets anticipate the Fed and a model that reads the bill spread can beat
every baseline.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eco_prediction.backtest import metrics
from eco_prediction.backtest.data import VintageStore
from eco_prediction.backtest.fomc import FOMCBacktest, summarize_fomc
from eco_prediction.data.fomc import (
    FOMC_MEETINGS,
    OUTCOMES,
    decision,
    fed_funds_target,
    meetings_between,
)
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.fomc_model import (
    AlwaysHold,
    Climatology,
    FOMCForecaster,
    FOMCModel,
    Persistence,
    forecast_cutoff,
    rate_features,
)

SMALL_GRID: dict[str, list[Any]] = {
    "max_depth": [2],
    "min_data_in_leaf": [5],
    "learning_rate": [0.1],
}
CODE = {"cut": -1, "hold": 0, "hike": 1}


def frame(rows: list[tuple[date, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])


def daily(values: pd.Series, lag_days: int = 1) -> pd.DataFrame:
    """A daily series where each day is published `lag_days` later."""
    days = pd.DatetimeIndex(values.index)
    return frame(
        [
            (d.date(), (d + timedelta(days=lag_days)).date(), float(v))
            for d, v in zip(days, values.to_numpy())
        ]
    )


def synthetic(end: date = date(2024, 12, 31)) -> tuple[VintageStore, dict[date, str]]:
    rng = np.random.default_rng(21)
    meetings = meetings_between(date(2009, 1, 1), end)
    outcomes = {m: str(rng.choice(OUTCOMES, p=[0.2, 0.55, 0.25])) for m in meetings}
    days = pd.date_range("2008-12-16", end, freq="D")
    upper = pd.Series(2.0, index=days)
    for m, outcome in outcomes.items():
        upper[upper.index > pd.Timestamp(m)] += 0.25 * CODE[outcome]
    # The next meeting's decision, for every day before it.
    upcoming = pd.Series(
        [
            CODE[outcomes[next(m for m in meetings if m >= d.date())]]
            if d.date() <= meetings[-1]
            else 0
            for d in days
        ],
        index=days,
    )
    tbill = upper - 0.125 + 0.2 * upcoming + rng.normal(0, 0.03, len(days))
    weekdays = tbill[days.dayofweek < 5]
    store = VintageStore({"DFEDTARU": daily(upper), "DTB3": daily(weekdays)})
    return store, outcomes


STORE, OUTCOMES_BY_MEETING = synthetic()


# Calendar and labels


def test_calendar_has_eight_scheduled_meetings_a_year() -> None:
    per_year = Counter(m.year for m in FOMC_MEETINGS)
    assert set(per_year) == set(range(1994, 2028))
    assert all(n == 8 for y, n in per_year.items() if y != 2020)
    assert per_year[2020] == 7  # March 17-18 was cancelled
    assert list(FOMC_MEETINGS) == sorted(set(FOMC_MEETINGS))
    for excluded in (date(2020, 3, 15), date(2003, 9, 15), date(2025, 8, 22)):
        assert excluded not in FOMC_MEETINGS
    assert date(2015, 12, 16) in FOMC_MEETINGS  # first hike after the ZLB


def test_target_splices_the_rate_and_the_range_midpoint() -> None:
    store = VintageStore(
        {
            "DFEDTAR": frame(
                [
                    (date(2008, 12, 14), date(2008, 12, 15), 1.0),
                    (date(2008, 12, 15), date(2008, 12, 15), 1.0),
                ]
            ),
            "DFEDTARU": frame([(date(2008, 12, 16), date(2008, 12, 17), 0.25)]),
        }
    )
    target = fed_funds_target(store, date(2008, 12, 31))
    assert target.tolist() == [1.0, 1.0, 0.125]
    assert decision(store, date(2008, 12, 15), date(2008, 12, 31)) == "cut"


def test_decision_labels_and_when_they_are_known() -> None:
    for meeting, outcome in list(OUTCOMES_BY_MEETING.items())[:20]:
        assert decision(STORE, meeting, date(2024, 12, 31)) == outcome
    meeting = date(2015, 12, 16)
    # The day after the meeting is published two days after it.
    assert decision(STORE, meeting, meeting + timedelta(days=1)) is None
    assert decision(STORE, meeting, meeting + timedelta(days=2)) is not None


def test_rate_features() -> None:
    meeting = date(2016, 3, 16)
    as_of = forecast_cutoff(meeting, 7)
    features = rate_features(STORE, meeting, as_of)
    target = fed_funds_target(STORE, as_of).iloc[-1]
    assert features["fed_target"] == pytest.approx(target)
    previous = [m for m in OUTCOMES_BY_MEETING if m < meeting]
    codes = [CODE[OUTCOMES_BY_MEETING[m]] for m in previous]
    assert features["last_decision"] == codes[-1]
    assert features["last_decision2"] == codes[-2]
    since = next(i for i, c in enumerate(reversed(codes)) if c != 0)
    assert features["meetings_since_change"] == since
    # The bill spread carries the upcoming decision (0.2 x code, plus noise).
    expected = 0.2 * CODE[OUTCOMES_BY_MEETING[meeting]]
    assert features["tbill_spread"] == pytest.approx(expected, abs=0.12)
    # No 2-year series in the store: its features are NaN, not errors.
    assert np.isnan(features["t2y_spread"])


def test_store_until_hides_later_vintages() -> None:
    cutoff = date(2016, 6, 30)
    clipped = STORE.until(cutoff)
    assert clipped.history("DTB3")["as_of"].max() <= pd.Timestamp(cutoff)
    assert clipped.latest_vintage("DFEDTARU") == cutoff
    assert clipped.history("UNRATE").empty


# Baselines


def test_baselines() -> None:
    cutoff = date(2016, 12, 31)
    known = [o for m, o in OUTCOMES_BY_MEETING.items() if m <= date(2016, 12, 28)]

    hold = AlwaysHold()
    hold.fit(STORE, cutoff)
    assert hold.predict_proba(STORE, date(2017, 2, 1)) == {
        "cut": 0.0,
        "hold": 1.0,
        "hike": 0.0,
    }

    clim = Climatology(train_start=date(2009, 1, 1))
    clim.fit(STORE, cutoff)
    p = clim.predict_proba(STORE, date(2017, 2, 1))
    assert p["hike"] == pytest.approx((known.count("hike") + 1) / (len(known) + 3))
    assert sum(p.values()) == pytest.approx(1)

    pers = Persistence(train_start=date(2009, 1, 1))
    pers.fit(STORE, cutoff)
    after_hike = [b for a, b in pairwise(known) if a == "hike"]
    assert pers.transitions is not None
    assert pers.transitions["hike"]["hold"] == pytest.approx(
        (after_hike.count("hold") + 1) / (len(after_hike) + 3)
    )
    last = known[-1]
    assert pers.predict_proba(STORE, date(2017, 2, 1)) == pers.transitions[last]

    with pytest.raises(NotFittedError):
        Climatology().predict_proba(STORE, date(2017, 2, 1))


# Forecaster


def model(calibrate: bool = False) -> FOMCForecaster:
    return FOMCForecaster(
        train_start=date(2009, 1, 1), param_grid=SMALL_GRID, calibrate=calibrate
    )


def test_training_frame_lines_up_meetings_and_labels() -> None:
    cutoff = date(2016, 12, 31)
    X, y = model().training_frame(STORE, cutoff)
    expected = [m for m in OUTCOMES_BY_MEETING if m <= date(2016, 12, 28)]
    assert [d.date() for d in X.index] == expected
    assert [OUTCOMES[i] for i in y] == [OUTCOMES_BY_MEETING[m] for m in expected]
    assert "trend_months" not in X and "month_1" not in X
    assert "tbill_spread" in X


def test_forecaster_learns_the_market_signal() -> None:
    m = model()
    m.fit(STORE, date(2018, 12, 31))
    assert m.feature_importance().index[0] == "tbill_spread"
    for meeting in meetings_between(date(2019, 1, 1), date(2019, 12, 31)):
        probs = m.predict_proba(STORE, meeting)
        assert list(probs) == list(OUTCOMES)
        assert sum(probs.values()) == pytest.approx(1)
        assert max(probs, key=probs.get) == OUTCOMES_BY_MEETING[meeting]  # type: ignore[arg-type]


def test_calibrated_forecaster_returns_valid_probabilities() -> None:
    m = model(calibrate=True)
    m.fit(STORE, date(2018, 12, 31))
    assert m.calibrator is not None
    probs = m.predict_proba(STORE, date(2019, 3, 20))
    assert list(probs) == list(OUTCOMES)
    assert sum(probs.values()) == pytest.approx(1)
    assert all(0 <= p <= 1 for p in probs.values())


def test_forecaster_errors() -> None:
    with pytest.raises(ValueError, match="lead_days"):
        FOMCForecaster(lead_days=-1)
    with pytest.raises(NotFittedError):
        model().predict_proba(STORE, date(2019, 3, 20))
    with pytest.raises(ValueError, match="at least 40"):
        model().fit(STORE, date(2011, 12, 31))


# Backtest


class Spy:
    """Records the stores it is given and checks nothing beyond the cutoff."""

    lead_days = 7

    def __init__(self) -> None:
        self.fits: list[date] = []

    def fit(self, store: VintageStore, cutoff: date) -> None:
        assert store.history("DFEDTARU")["as_of"].max() <= pd.Timestamp(cutoff)
        self.fits.append(cutoff)

    def predict_proba(self, store: VintageStore, meeting: date) -> dict[str, float]:
        cutoff = forecast_cutoff(meeting, self.lead_days)
        assert store.history("DTB3")["as_of"].max() <= pd.Timestamp(cutoff)
        return {"cut": 0.1, "hold": 0.8, "hike": 0.1}


def test_backtest_refits_and_never_passes_later_data() -> None:
    spy = Spy()
    bt = FOMCBacktest(date(2017, 1, 1), date(2018, 12, 31), "annually", store=STORE)
    results = bt.run(spy)
    assert len(results) == 16
    assert spy.fits == [date(2017, 1, 24), date(2018, 1, 23)]  # 7 days + 1 before
    expected = [OUTCOMES_BY_MEETING[m] for m in results["meeting"]]
    assert list(results["outcome"]) == expected
    briers = [
        metrics.brier_score({"cut": 0.1, "hold": 0.8, "hike": 0.1}, o) for o in expected
    ]
    assert results["brier"].tolist() == pytest.approx(briers)


def test_backtest_leaves_future_meetings_unresolved() -> None:
    store, _ = synthetic(end=date(2019, 6, 30))
    bt = FOMCBacktest(date(2019, 1, 1), date(2019, 12, 31), "annually", store=store)
    results = bt.run(Spy())
    assert results["outcome"].isna().sum() == 4  # July, Sept, Oct, Dec
    assert summarize_fomc(results)["n_resolved"] == 4


def test_summarize_fomc_hand_computed() -> None:
    results = pd.DataFrame(
        {
            "p_cut": [0.1, 0.6],
            "p_hold": [0.8, 0.3],
            "p_hike": [0.1, 0.1],
            "outcome": ["hold", "hold"],
            "brier": [0.06, 0.86],
        }
    )
    summary = summarize_fomc(results)
    assert summary["brier"] == pytest.approx(0.46)
    assert summary["accuracy"] == 0.5
    assert summary["n_changes"] == 0 and summary["brier_on_changes"] is None


def test_forecaster_beats_baselines_in_walk_forward() -> None:
    bt = FOMCBacktest(date(2019, 1, 1), date(2024, 12, 31), "annually", store=STORE)
    models: dict[str, FOMCModel] = {
        "hold": AlwaysHold(),
        "climatology": Climatology(train_start=date(2009, 1, 1)),
        "persistence": Persistence(train_start=date(2009, 1, 1)),
        "lgbm": model(),
    }
    briers = {}
    for name, m in models.items():
        brier = summarize_fomc(bt.run(m))["brier"]
        assert brier is not None
        briers[name] = brier
    best_baseline = min(v for k, v in briers.items() if k != "lgbm")
    assert briers["lgbm"] < 0.5 * best_baseline
