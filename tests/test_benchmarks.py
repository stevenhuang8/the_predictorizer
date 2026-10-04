"""Tests for the SPF and Cleveland Fed benchmarks, using canned files (no network)."""

from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from eco_prediction.backtest.data import TARGETS, PointInTimeData, Target, VintageStore
from eco_prediction.backtest.harness import WalkForwardBacktest, add_months
from eco_prediction.models.benchmarks import (
    NOWCAST_URL,
    SPF,
    SPF_MEDIAN_URLS,
    SPF_RELEASE_DATES_URL,
    BenchmarkUnavailable,
    CachedDownloader,
    ClevelandNowcast,
    ExpertBenchmark,
    SPFForecaster,
    parse_nowcasts,
    parse_spf_release_dates,
)

RELEASE_TEXT = """\
Deadline and Release Dates for the Survey of Professional Forecasters

Survey         True Deadline Date       News Release Date

2019 Q4             11/12/19            11/15/19
2020 Q1             2/11/20             2/14/20
     Q2             5/12/20             5/15/20
     Q3             8/11/20**           8/14/20**

**The survey was delayed.
"""


def spf_xlsx(variable: str, rows: list[tuple[int, int, list[float | None]]]) -> bytes:
    frame = pd.DataFrame(
        [
            {
                "YEAR": y,
                "QUARTER": q,
                **{f"{variable}{k + 1}": v for k, v in enumerate(vals)},
            }
            for y, q, vals in rows
        ]
    )
    buffer = io.BytesIO()
    frame.to_excel(buffer, sheet_name="Median_Level", index=False)
    return buffer.getvalue()


UNEMP = spf_xlsx(
    "UNEMP",
    [
        (2019, 4, [3.6, 3.6, 3.6, 3.7, 3.7, 3.8]),
        (2020, 1, [3.5, 3.5, 3.6, 3.6, 3.7, 3.7]),
        (2020, 2, [3.8, 15.0, 11.0, 9.0, 8.0, 7.0]),
        (2020, 3, [13.0, 9.0, 8.0, 7.0, 6.5, None]),
    ],
)
# Annualized q/q % changes of quarterly-average CPI.
CPI = spf_xlsx(
    "CPI",
    [
        (2019, 4, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        (2020, 1, [2.4, 4.0, 4.0, 4.0, 4.0, 4.0]),
        (2020, 2, [1.0, -3.0, 2.0, 2.0, 2.0, 2.0]),
        (2020, 3, [-3.5, 4.0, 2.0, 2.0, 2.0, 2.0]),
    ],
)


class CannedDownloader(CachedDownloader):
    """Serves fixed bytes per URL and counts requests."""

    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__(Path("/nonexistent"))
        self.files = files
        self.requests: list[str] = []

    def get(self, url: str) -> bytes:
        self.requests.append(url)
        if url not in self.files:
            raise BenchmarkUnavailable(f"no canned file for {url}")
        return self.files[url]


def spf() -> SPF:
    return SPF(
        CannedDownloader(
            {
                SPF_RELEASE_DATES_URL: RELEASE_TEXT.encode(),
                SPF_MEDIAN_URLS["UNEMP"]: UNEMP,
                SPF_MEDIAN_URLS["CPI"]: CPI,
            }
        )
    )


# --- CachedDownloader -------------------------------------------------------


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False
        self.content = b"v1"

    def get(self, url: str, **_: Any) -> FakeResponse:
        self.calls += 1
        if self.fail:
            raise requests.ConnectionError("network down")
        return FakeResponse(self.content)


def downloader(
    tmp_path: Path, session: FakeSession, now: list[float]
) -> CachedDownloader:
    return CachedDownloader(
        tmp_path,
        timedelta(hours=1),
        session=session,  # type: ignore[arg-type]
        clock=lambda: now[0],
    )


def test_downloader_serves_fresh_cache_without_a_request(tmp_path: Path) -> None:
    session, now = FakeSession(), [0.0]
    dl = downloader(tmp_path, session, now)
    assert dl.get("https://x/a.json") == b"v1"
    now[0] = dl.path_for("https://x/a.json").stat().st_mtime + 60
    session.content = b"v2"
    assert dl.get("https://x/a.json") == b"v1"
    assert session.calls == 1


def test_downloader_refreshes_stale_cache(tmp_path: Path) -> None:
    session, now = FakeSession(), [0.0]
    dl = downloader(tmp_path, session, now)
    dl.get("https://x/a.json")
    now[0] = dl.path_for("https://x/a.json").stat().st_mtime + 7200
    session.content = b"v2"
    assert dl.get("https://x/a.json") == b"v2"
    assert dl.path_for("https://x/a.json").read_bytes() == b"v2"


def test_downloader_falls_back_to_stale_cache_when_offline(tmp_path: Path) -> None:
    session, now = FakeSession(), [0.0]
    dl = downloader(tmp_path, session, now)
    dl.get("https://x/a.json")
    now[0] = dl.path_for("https://x/a.json").stat().st_mtime + 7200
    session.fail = True
    assert dl.get("https://x/a.json") == b"v1"


def test_downloader_without_cache_raises_when_offline(tmp_path: Path) -> None:
    session = FakeSession()
    session.fail = True
    with pytest.raises(BenchmarkUnavailable, match="network down"):
        downloader(tmp_path, session, [0.0]).get("https://x/a.json")


def test_cache_files_are_distinct_per_url(tmp_path: Path) -> None:
    dl = CachedDownloader(tmp_path)
    assert dl.path_for("https://a/f.xlsx") != dl.path_for("https://b/f.xlsx")
    assert dl.path_for("https://a/f.xlsx").name.endswith("_f.xlsx")


# --- SPF ---------------------------------------------------------------------


def test_release_dates_carry_the_year_and_ignore_footnotes() -> None:
    dates = parse_spf_release_dates(RELEASE_TEXT)
    assert dates == {
        pd.Period("2019Q4"): date(2019, 11, 15),
        pd.Period("2020Q1"): date(2020, 2, 14),
        pd.Period("2020Q2"): date(2020, 5, 15),
        pd.Period("2020Q3"): date(2020, 8, 14),
    }


def test_two_digit_years_before_2000() -> None:
    dates = parse_spf_release_dates("1991 Q1             2/16/91             2/21/91\n")
    assert dates == {pd.Period("1991Q1"): date(1991, 2, 21)}


def test_survey_is_visible_only_from_its_release_date() -> None:
    s = spf()
    assert s.latest_survey("UNEMP", date(2020, 5, 14)) == pd.Period("2020Q1")
    assert s.latest_survey("UNEMP", date(2020, 5, 15)) == pd.Period("2020Q2")
    assert s.latest_survey("UNEMP", date(2019, 11, 1)) is None


def test_surveys_without_known_dates_are_assumed_mid_quarter() -> None:
    assert spf().released_on(pd.Period("1985Q2")) == date(1985, 6, 15)


def test_unemployment_maps_target_month_to_its_quarter() -> None:
    s = spf()
    # 2020Q2 survey: column 2 is Q2, column 3 is Q3.
    assert s.unemployment(date(2020, 6, 1), date(2020, 6, 30)) == 15.0
    assert s.unemployment(date(2020, 7, 1), date(2020, 6, 30)) == 11.0
    # Before that survey's release the Q1 survey applies.
    assert s.unemployment(date(2020, 6, 1), date(2020, 5, 1)) == 3.6
    # Beyond the survey's last forecast quarter (2021Q2 for the Q2 survey).
    assert s.unemployment(date(2021, 7, 1), date(2020, 6, 30)) is None


def monthly_cpi(levels: dict[str, float]) -> pd.Series:
    """Monthly CPI with each quarter's three months at the given average."""
    values = {}
    for q, avg in levels.items():
        start = pd.Period(q).start_time
        for m in range(3):
            values[start + pd.DateOffset(months=m)] = avg
    return pd.Series(values, dtype="float64").sort_index()


def test_cpi_yoy_chains_spf_rates_onto_known_quarters() -> None:
    # As of 2020-06-30: the 2020Q2 survey is out; CPI is known through 2020Q1.
    cpi = monthly_cpi(
        {"2019Q2": 100.0, "2019Q3": 100.5, "2019Q4": 101.0, "2020Q1": 101.5}
    )
    value = spf().cpi_yoy(date(2020, 8, 1), date(2020, 6, 30), cpi)

    q2 = 101.5 * (1 - 0.03) ** 0.25  # survey quarter: -3.0% annualized
    q3 = q2 * 1.02**0.25  # next quarter: +2.0%
    assert value == pytest.approx((q3 / 100.5 - 1) * 100)


def test_cpi_yoy_prefers_actuals_over_the_survey() -> None:
    cpi = monthly_cpi(
        {
            "2019Q2": 100.0,
            "2019Q3": 100.5,
            "2019Q4": 101.0,
            "2020Q1": 101.5,
            "2020Q2": 99.0,
        }
    )
    value = spf().cpi_yoy(date(2020, 8, 1), date(2020, 6, 30), cpi)
    q3 = 99.0 * 1.02**0.25
    assert value == pytest.approx((q3 / 100.5 - 1) * 100)


def test_cpi_yoy_needs_the_year_earlier_quarter() -> None:
    cpi = monthly_cpi({"2020Q1": 101.5})
    assert spf().cpi_yoy(date(2020, 8, 1), date(2020, 6, 30), cpi) is None


def store_through(last: date) -> VintageStore:
    rows, period = [], date(2017, 1, 1)
    while period <= last:
        rows.append((period, add_months(period, 1).replace(day=12), 4.0))
        period = add_months(period, 1)
    frame = pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])
    return VintageStore({"UNRATE": frame, "CPIAUCSL": frame})


def test_spf_forecaster_in_harness() -> None:
    bt = WalkForwardBacktest(
        date(2020, 6, 1),
        date(2020, 9, 1),
        "monthly",
        store=store_through(date(2020, 12, 1)),
    )
    results = bt.run(SPFForecaster(spf()), "unemployment", horizon_months=1)
    # Forecast on 06-01 uses the Q2 survey (out 05-15); 09-01 the Q3 one (08-14).
    assert results["prediction"].tolist() == [11.0, 11.0, 11.0, 8.0]


def test_spf_forecaster_rejects_targets_it_does_not_cover() -> None:
    other = Target("fomc_decision", "DFEDTARU", lambda s: s, "")
    data = PointInTimeData(store_through(date(2020, 6, 1)), other, date(2020, 6, 30))
    with pytest.raises(BenchmarkUnavailable, match="no forecast for fomc_decision"):
        SPFForecaster(spf()).predict(data, date(2020, 7, 1))

    data = PointInTimeData(
        store_through(date(2020, 6, 1)), TARGETS["unemployment"], date(2020, 6, 30)
    )
    with pytest.raises(BenchmarkUnavailable, match="covers 2021Q3"):
        SPFForecaster(spf()).predict(data, date(2021, 7, 1))


def test_spf_files_are_fetched_once() -> None:
    s = spf()
    for d in (date(2020, 6, 30), date(2020, 9, 30)):
        s.unemployment(date(2020, 12, 1), d)
    assert isinstance(s.downloader, CannedDownloader)
    assert sorted(s.downloader.requests) == sorted(
        [SPF_RELEASE_DATES_URL, SPF_MEDIAN_URLS["UNEMP"]]
    )


# --- Cleveland Fed nowcast ----------------------------------------------------


def nowcast_entry(target: str, points: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "chart": {"subcaption": target},
        "categories": [
            {
                "category": [{"label": label} for label, _ in points]
                + [{"label": "PCE", "vline": "true"}]
            }
        ],
        "dataset": [
            {"seriesname": "CPI Inflation", "data": [{"value": v} for _, v in points]},
            {
                "seriesname": "Core CPI Inflation",
                "data": [{"value": "9.9"} for _ in points],
            },
        ],
    }


NOWCAST_JSON = json.dumps(
    [
        nowcast_entry("2023-12", [("12/01", "3.0"), ("12/04", ""), ("01/10", "3.3")]),
        nowcast_entry("2024-1", [("01/02", "3.1"), ("01/10", "3.2"), ("02/12", "3.0")]),
    ]
).encode()


def test_nowcasts_parse_dates_across_the_year_end() -> None:
    frame = parse_nowcasts(NOWCAST_JSON)
    assert list(frame.itertuples(index=False, name=None)) == [
        (date(2023, 12, 1), date(2023, 12, 1), 3.0),
        (date(2023, 12, 1), date(2024, 1, 10), 3.3),  # empty 12/04 skipped
        (date(2024, 1, 1), date(2024, 1, 2), 3.1),
        (date(2024, 1, 1), date(2024, 1, 10), 3.2),
        (date(2024, 1, 1), date(2024, 2, 12), 3.0),
    ]


def test_nowcast_as_of_a_date() -> None:
    nowcast = ClevelandNowcast(CannedDownloader({NOWCAST_URL: NOWCAST_JSON}))
    assert nowcast.nowcast(date(2024, 1, 9), date(2023, 12, 1)) == 3.0
    assert nowcast.nowcast(date(2024, 1, 10), date(2023, 12, 1)) == 3.3
    assert nowcast.nowcast(date(2024, 1, 9)) == 3.1  # latest month nowcast then
    assert nowcast.nowcast(date(2023, 11, 30)) is None


def test_expert_benchmark_shares_one_downloader() -> None:
    files = {
        NOWCAST_URL: NOWCAST_JSON,
        SPF_RELEASE_DATES_URL: RELEASE_TEXT.encode(),
        SPF_MEDIAN_URLS["UNEMP"]: UNEMP,
    }
    bench = ExpertBenchmark(CannedDownloader(files))
    assert bench.get_cpi_nowcast(date(2024, 2, 12), date(2024, 1, 1)) == 3.0
    assert bench.get_unemployment_spf(date(2020, 6, 30), date(2020, 6, 1)) == 15.0
