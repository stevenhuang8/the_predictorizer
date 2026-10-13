"""Client for arXiv's search API: monthly submission counts.

Usage:
    from eco_prediction.ai.sources.arxiv_client import ArxivClient

    with ArxivClient.from_env() as arxiv:
        n = arxiv.count_submissions(date(2024, 1, 1))  # cs.AI/LG/CL, Jan 2024

A count is one query, `(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND
submittedDate:[month]`, read from the feed's `opensearch:totalResults`. `cat:`
matches a paper's primary category and its cross-lists, and a paper listed in
several of the categories is counted once. `submittedDate` is the first
version's submission time, so later revisions don't move a paper's month.

OAI-PMH isn't used: its `from`/`until` select records by last modification,
not submission, so counting submissions through it means harvesting the whole
cs set.

arXiv asks for at most one request every 3 seconds, but in practice it
blocks for minutes after a short burst at that pace (429, no Retry-After), so
the client sends one every 5 seconds and backs off for minutes on a 429.
It answers errors as an Atom feed with a single entry whose id ends in
`/api/errors`, sometimes with status 200, and `max_results=0` gets a 500, so
queries ask for one result.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests

from eco_prediction.ai.sources.base_client import BaseAPIClient, SourceAPIError

BASE_URL = "https://export.arxiv.org/api"
AI_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL")

_ATOM = "{http://www.w3.org/2005/Atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


@dataclass(frozen=True)
class MonthlyCount:
    month: date  # first day of the month
    count: int


class ArxivClient(BaseAPIClient):
    SOURCE = "arxiv"
    BASE_URL = BASE_URL
    MAX_CALLS = 1
    PERIOD = 5.0

    def __init__(self, **kwargs: Any) -> None:
        # arXiv answers 429 with no Retry-After and blocks for minutes (about 4
        # on 2026-10-12), so back off from 15s up to 5 minutes: ~13 minutes of
        # retries in all, enough to wait a block out.
        kwargs.setdefault("backoff_base", 15.0)
        kwargs.setdefault("backoff_max", 300.0)
        kwargs.setdefault("max_retries", 6)
        super().__init__(**kwargs)

    def count_submissions(
        self, month: date, categories: Sequence[str] = AI_CATEGORIES
    ) -> int:
        """Papers first submitted in `month` and listed in any of `categories`."""
        resp = self._get(
            "query",
            {"search_query": submissions_query(month, categories), "max_results": "1"},
        )
        return parse_total_results(resp.text)

    def monthly_counts(
        self, start: date, end: date, categories: Sequence[str] = AI_CATEGORIES
    ) -> list[MonthlyCount]:
        """One count per month from `start`'s month through `end`'s, inclusive."""
        return [
            MonthlyCount(m, self.count_submissions(m, categories))
            for m in iter_months(start, end)
        ]

    def _error_message(self, resp: requests.Response) -> str:
        return _feed_error(resp.text) or resp.text[:500]


def submissions_query(month: date, categories: Sequence[str]) -> str:
    """Search query for papers first submitted in `month` (GMT), in any category."""
    first = month_start(month)
    last = next_month(first) - timedelta(days=1)
    cats = " OR ".join(f"cat:{c}" for c in categories)
    return f"({cats}) AND submittedDate:[{first:%Y%m%d}0000 TO {last:%Y%m%d}2359]"


def parse_total_results(feed: str) -> int:
    """`opensearch:totalResults` of a search feed; raises on an error feed."""
    try:
        root = ET.fromstring(feed)
    except ET.ParseError as exc:
        raise SourceAPIError(f"Invalid XML from arxiv: {exc}", 200, "arxiv") from None
    error = _feed_error(root)
    if error:
        raise SourceAPIError(f"arxiv query failed: {error}", 200, "arxiv")
    total = root.findtext(f"{_OPENSEARCH}totalResults")
    if total is None or not total.strip().isdigit():
        raise SourceAPIError("arxiv feed has no totalResults", 200, "arxiv")
    return int(total)


def month_start(d: date) -> date:
    return d.replace(day=1)


def next_month(d: date) -> date:
    return date(d.year + d.month // 12, d.month % 12 + 1, 1)


def iter_months(start: date, end: date) -> Iterator[date]:
    """First day of each month from `start`'s through `end`'s."""
    month, last = month_start(start), month_start(end)
    while month <= last:
        yield month
        month = next_month(month)


def last_complete_month(today: date) -> date:
    """First day of the latest month that has fully ended before `today`."""
    return month_start(month_start(today) - timedelta(days=1))


def _feed_error(feed: str | ET.Element) -> str | None:
    """The message of an arXiv error feed, or None if it isn't one."""
    if isinstance(feed, str):
        try:
            feed = ET.fromstring(feed)
        except ET.ParseError:
            return None
    for entry in feed.iter(f"{_ATOM}entry"):
        if (entry.findtext(f"{_ATOM}id") or "").endswith("/api/errors"):
            return (entry.findtext(f"{_ATOM}summary") or "unknown error").strip()
    return None
