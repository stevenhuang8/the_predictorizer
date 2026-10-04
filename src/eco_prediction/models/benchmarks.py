"""Professional forecasts to benchmark against, read point-in-time.

Usage:
    from eco_prediction.models.benchmarks import ExpertBenchmark, SPFForecaster

    bench = ExpertBenchmark()
    bench.get_unemployment_spf(date(2020, 3, 31), date(2020, 6, 1))  # SPF median
    bench.get_cpi_nowcast(date(2024, 3, 11))  # Cleveland Fed CPI YoY nowcast

    bt.run(SPFForecaster(), "unemployment", horizon_months=3)  # in the harness

Sources:
- Survey of Professional Forecasters (Philadelphia Fed), quarterly median
  forecasts. Each survey is visible from its published news release date
  (the 15th of the survey quarter's third month for surveys before 1990Q2,
  whose dates aren't known). Unemployment is the quarterly average rate; CPI is
  the annualized quarter-over-quarter change of the quarterly average CPI. For
  CPI YoY the harness target is approximated by the YoY change of the target
  month's quarterly average, chaining SPF's rates onto CPI known at the time.
- Cleveland Fed inflation nowcast: daily CPI YoY nowcasts since 2013-08, for
  the current and previous month only, so it serves as a nowcast benchmark
  rather than for the harness's 1-6 month horizons.

CME FedWatch has no free historical archive and is not implemented.

Downloads are cached on disk (`data/cache/benchmarks/` by default) and
refreshed after `max_age`; if a refresh fails, the stale copy is used.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import warnings
from collections.abc import Callable
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import requests

from eco_prediction.backtest.data import PointInTimeData
from eco_prediction.backtest.harness import Forecast

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache/benchmarks")

_SPF_BASE = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters"
)
SPF_MEDIAN_URLS = {
    "UNEMP": f"{_SPF_BASE}/data-files/files/median_unemp_level.xlsx",
    "CPI": f"{_SPF_BASE}/data-files/files/median_cpi_level.xlsx",
}
SPF_RELEASE_DATES_URL = f"{_SPF_BASE}/spf-release-dates.txt"
NOWCAST_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/"
    "inflationnowcasting/nowcast_year.json"
)

SPFVariable = Literal["UNEMP", "CPI"]

# "2019 Q1   3/12/19**   3/22/19**" or "     Q2   5/18/91   5/24/91"
_RELEASE_RE = re.compile(
    r"^\s*(?:(\d{4})\s+)?Q([1-4])\s+\S+\s+(\d{1,2})/(\d{1,2})/(\d{2})\**\s*$"
)


class BenchmarkUnavailable(Exception):
    """The benchmark has no forecast for this target, or couldn't be fetched."""


class CachedDownloader:
    """GET with an on-disk cache that falls back to stale copies on failure."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        max_age: timedelta = timedelta(days=1),
        *,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_age = max_age
        self.session = session or requests.Session()
        self.timeout = timeout
        self._clock = clock

    def path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:8]
        return self.cache_dir / f"{digest}_{url.rsplit('/', 1)[-1]}"

    def get(self, url: str) -> bytes:
        path = self.path_for(url)
        if path.exists():
            age = self._clock() - path.stat().st_mtime
            if age < self.max_age.total_seconds():
                return path.read_bytes()
        try:
            resp = self.session.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "eco-prediction/0.1 (research)"},
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            if path.exists():
                log.warning("Refreshing %s failed (%s); using cached copy", url, exc)
                return path.read_bytes()
            raise BenchmarkUnavailable(f"Couldn't download {url}: {exc}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
        return resp.content


def quarter(d: date) -> pd.Period:
    return pd.Period(year=d.year, quarter=(d.month - 1) // 3 + 1, freq="Q")


def parse_spf_release_dates(text: str) -> dict[pd.Period, date]:
    """Survey quarter -> news release date, from the Philadelphia Fed's text file."""
    dates: dict[pd.Period, date] = {}
    year: int | None = None
    for line in text.splitlines():
        match = _RELEASE_RE.match(line)
        if not match:
            continue
        if match[1]:
            year = int(match[1])
        if year is None:
            continue
        yy = int(match[5])
        released = date(
            1900 + yy if yy >= 68 else 2000 + yy, int(match[3]), int(match[4])
        )
        dates[pd.Period(year=year, quarter=int(match[2]), freq="Q")] = released
    return dates


def parse_spf_medians(content: bytes, variable: SPFVariable) -> pd.DataFrame:
    """Median forecasts indexed by survey quarter.

    Column k (1-6) is the forecast for survey quarter + (k - 2): 1 is the
    previous quarter (the jump-off estimate), 2 the survey quarter, 3-6 the
    following four quarters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # unparsable sheet header
        raw = pd.read_excel(io.BytesIO(content), sheet_name=0)
    index = pd.PeriodIndex(
        [
            pd.Period(year=int(y), quarter=int(q), freq="Q")
            for y, q in zip(raw["YEAR"], raw["QUARTER"], strict=True)
        ],
        name="survey",
    )
    columns = {
        k: pd.to_numeric(raw[f"{variable}{k}"], errors="coerce") for k in range(1, 7)
    }
    return pd.DataFrame({k: col.to_numpy() for k, col in columns.items()}, index=index)


class SPF:
    """Survey of Professional Forecasters medians, as released at the time."""

    def __init__(self, downloader: CachedDownloader | None = None) -> None:
        self.downloader = downloader or CachedDownloader()
        self._medians: dict[str, pd.DataFrame] = {}

    @cached_property
    def release_dates(self) -> dict[pd.Period, date]:
        text = self.downloader.get(SPF_RELEASE_DATES_URL).decode("utf-8", "replace")
        return parse_spf_release_dates(text)

    def medians(self, variable: SPFVariable) -> pd.DataFrame:
        if variable not in self._medians:
            content = self.downloader.get(SPF_MEDIAN_URLS[variable])
            self._medians[variable] = parse_spf_medians(content, variable)
        return self._medians[variable]

    def released_on(self, survey: pd.Period) -> date:
        known = self.release_dates.get(survey)
        if known is not None:
            return known
        # Before 1990Q2 the dates aren't known; surveys came out mid-quarter.
        middle = survey.start_time.date().replace(day=15)
        return (pd.Timestamp(middle) + pd.DateOffset(months=2)).date()

    def latest_survey(self, variable: SPFVariable, as_of: date) -> pd.Period | None:
        """The most recent survey released on or before `as_of`."""
        released = [
            s for s in self.medians(variable).index if self.released_on(s) <= as_of
        ]
        return max(released) if released else None

    def forecasts(self, variable: SPFVariable, as_of: date) -> dict[pd.Period, float]:
        """The latest released survey's medians, keyed by the quarter forecast."""
        survey = self.latest_survey(variable, as_of)
        if survey is None:
            return {}
        frame = self.medians(variable)
        row = frame[frame.index == survey].iloc[0]
        return {
            survey + (k - 2): float(row[k]) for k in range(1, 7) if pd.notna(row[k])
        }

    def unemployment(self, target_month: date, as_of: date) -> float | None:
        """Median forecast of the target month's quarterly average rate."""
        return self.forecasts("UNEMP", as_of).get(quarter(target_month))

    def cpi_yoy(
        self, target_month: date, as_of: date, cpi_monthly: pd.Series
    ) -> float | None:
        """YoY % change of the target month's quarterly average CPI.

        `cpi_monthly` is CPI as known on `as_of`. Quarters it fully covers use
        those actual values; later quarters are chained on with the SPF's
        annualized quarterly rates.
        """
        rates = self.forecasts("CPI", as_of)
        if not rates:
            return None
        known = cpi_monthly.dropna()
        by_quarter = known.groupby(pd.DatetimeIndex(known.index).to_period("Q"))
        complete = by_quarter.mean()[by_quarter.count() == 3]
        levels: dict[pd.Period, float] = {
            cast(pd.Period, q): float(v) for q, v in complete.items()
        }
        target = quarter(target_month)
        q = min(rates)
        while q <= target:
            if q not in levels:
                if q not in rates or q - 1 not in levels:
                    return None
                levels[q] = levels[q - 1] * (1 + rates[q] / 100) ** 0.25
            q += 1
        now, prior = levels.get(target), levels.get(target - 4)
        if now is None or prior is None:
            return None
        return (now / prior - 1) * 100


class SPFForecaster:
    """The SPF median as a harness `Forecaster` (point forecasts only)."""

    def __init__(self, spf: SPF | None = None) -> None:
        self.spf = spf or SPF()

    def fit(self, data: PointInTimeData) -> None:
        pass

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        name = data.target.name
        if name == "unemployment":
            value = self.spf.unemployment(target_period, data.cutoff)
        elif name == "cpi_yoy":
            value = self.spf.cpi_yoy(
                target_period, data.cutoff, data.series(data.target.series_id)
            )
        else:
            raise BenchmarkUnavailable(f"SPF has no forecast for {name}")
        if value is None:
            raise BenchmarkUnavailable(
                f"No SPF survey released by {data.cutoff} covers {quarter(target_period)}"
            )
        return Forecast(value)


def parse_nowcasts(content: bytes) -> pd.DataFrame:
    """Daily CPI YoY nowcasts: target_month, nowcast_date, cpi_yoy."""
    rows = []
    for entry in json.loads(content):
        year, month = (int(p) for p in entry["chart"]["subcaption"].split("-"))
        target = date(year, month, 1)
        labels = [
            c["label"] for c in entry["categories"][0]["category"] if not c.get("vline")
        ]
        series = _dataset(entry, "CPI Inflation")
        if series is None:
            continue
        for label, point in zip(labels, series, strict=False):
            if not point.get("value"):
                continue
            m, d = (int(p) for p in label.split("/"))
            # Labels run from the target month into the next two; only the
            # December -> January turn crosses a year.
            nowcast_date = date(year if m >= month else year + 1, m, d)
            rows.append((target, nowcast_date, float(point["value"])))
    frame = pd.DataFrame(rows, columns=["target_month", "nowcast_date", "cpi_yoy"])
    return frame.sort_values(["target_month", "nowcast_date"], ignore_index=True)


def _dataset(entry: dict[str, Any], name: str) -> list[dict[str, Any]] | None:
    for ds in entry.get("dataset", []):
        if ds.get("seriesname") == name:
            return list(ds.get("data", []))
    return None


class ClevelandNowcast:
    """Cleveland Fed daily CPI YoY nowcasts, read as of a date."""

    def __init__(self, downloader: CachedDownloader | None = None) -> None:
        self.downloader = downloader or CachedDownloader()

    @cached_property
    def history(self) -> pd.DataFrame:
        return parse_nowcasts(self.downloader.get(NOWCAST_URL))

    def nowcast(self, as_of: date, target_month: date | None = None) -> float | None:
        """Latest nowcast made on or before `as_of`.

        For `target_month` (first of month) if given, else for the latest month
        being nowcast on that date.
        """
        known = self.history[self.history["nowcast_date"] <= as_of]
        if target_month is not None:
            known = known[known["target_month"] == target_month]
        else:
            known = known[known["target_month"] == known["target_month"].max()]
        if known.empty:
            return None
        return float(known["cpi_yoy"].iloc[-1])


class ExpertBenchmark:
    """One entry point for the external benchmarks, sharing a download cache."""

    def __init__(self, downloader: CachedDownloader | None = None) -> None:
        downloader = downloader or CachedDownloader()
        self.spf = SPF(downloader)
        self.cleveland = ClevelandNowcast(downloader)

    def get_cpi_nowcast(
        self, as_of_date: date, target_month: date | None = None
    ) -> float | None:
        return self.cleveland.nowcast(as_of_date, target_month)

    def get_unemployment_spf(
        self, as_of_date: date, target_month: date
    ) -> float | None:
        return self.spf.unemployment(target_month, as_of_date)

    def get_cpi_spf(
        self, as_of_date: date, target_month: date, cpi_monthly: pd.Series
    ) -> float | None:
        return self.spf.cpi_yoy(target_month, as_of_date, cpi_monthly)
