"""Client for arXiv's search API: monthly submission counts, and abstracts.

Usage:
    from eco_prediction.ai.sources.arxiv_client import ArxivClient

    with ArxivClient.from_env() as arxiv:
        n = arxiv.count_submissions(date(2024, 1, 1))  # cs.AI/LG/CL, Jan 2024
        for paper in arxiv.fetch_abstracts(date(2024, 1, 1), date(2024, 12, 31)):
            ...                                        # ArxivPaper, AI task 17

A count is one query, `(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND
submittedDate:[month]`, read from the feed's `opensearch:totalResults`. `cat:`
matches a paper's primary category and its cross-lists, and a paper listed in
several of the categories is counted once. `submittedDate` is the first
version's submission time, so later revisions don't move a paper's month.

`fetch_abstracts` runs one search per `TOPIC_QUERIES` entry within the same
categories and dates, `PAGE_SIZE` results a page, oldest first, and yields
each paper once. The topics are the ones the forecasts ask about (the
benchmarks, technical reports and system cards, scaling, frontier models,
training compute): about 4,500 papers from 2023 to 2026, against ~4,000-11,000
a month in the categories overall (ARXIV_AI_SUBMISSIONS). An entry's abstract is its latest version's;
`published` is the first version's submission time.

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

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any

import requests

from eco_prediction.ai.months import MonthlyCount, iter_months, month_start, next_month
from eco_prediction.ai.sources.base_client import BaseAPIClient, SourceAPIError

BASE_URL = "https://export.arxiv.org/api"
AI_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL")
PAGE_SIZE = 500
EMPTY_PAGE_RETRIES = 2  # arXiv sometimes returns an empty page mid-listing

TOPIC_QUERIES: dict[str, str] = {
    "swe_bench": 'abs:"SWE-bench"',
    "gpqa": "abs:GPQA",
    "arc_agi": 'abs:"ARC-AGI"',
    "frontiermath": 'abs:"FrontierMath"',
    "hle": 'abs:"Last Exam"',
    "technical_report": 'ti:"technical report"',
    "system_card": 'ti:"system card"',
    "scaling_laws": 'abs:"scaling law" OR abs:"scaling laws"',
    "frontier_models": (
        'abs:"frontier model" OR abs:"frontier models" OR abs:"frontier AI"'
    ),
    "training_compute": 'abs:"training compute"',
}

_ATOM = "{http://www.w3.org/2005/Atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str  # without the version, e.g. '2310.06770'
    version: int  # of the abstract returned (the latest)
    title: str
    abstract: str
    published: datetime  # first version's submission, UTC
    updated: datetime  # latest version's
    primary_category: str | None
    categories: tuple[str, ...]
    authors: tuple[str, ...]
    topics: tuple[str, ...] = ()  # the TOPIC_QUERIES keys that found it


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

    def count_months(
        self, months: Iterable[date], categories: Sequence[str] = AI_CATEGORIES
    ) -> Iterator[MonthlyCount]:
        """One query per month, each count yielded as soon as it arrives."""
        for m in months:
            yield MonthlyCount(month_start(m), self.count_submissions(m, categories))

    def monthly_counts(
        self, start: date, end: date, categories: Sequence[str] = AI_CATEGORIES
    ) -> list[MonthlyCount]:
        """One count per month from `start`'s month through `end`'s, inclusive."""
        return list(self.count_months(iter_months(start, end), categories))

    def search_page(
        self, search_query: str, start: int, max_results: int = PAGE_SIZE
    ) -> tuple[list[ArxivPaper], int]:
        """One page of a search, oldest submissions first, and the total."""
        resp = self._get(
            "query",
            {
                "search_query": search_query,
                "start": str(start),
                "max_results": str(max_results),
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            },
        )
        return parse_papers(resp.text), parse_total_results(resp.text)

    def iter_search(
        self, search_query: str, page_size: int = PAGE_SIZE
    ) -> Iterator[ArxivPaper]:
        """Every result of a search, a page at a time."""
        start, total, empty = 0, None, 0
        while total is None or start < total:
            papers, total = self.search_page(search_query, start, page_size)
            if not papers and start < total:
                empty += 1
                if empty > EMPTY_PAGE_RETRIES:
                    raise SourceAPIError(
                        f"arxiv returned no results at {start} of {total}",
                        200,
                        self.SOURCE,
                    )
                self._sleep(self._backoff(empty, None))
                continue
            empty = 0
            yield from papers
            start += len(papers)

    def fetch_abstracts(
        self,
        start_date: date,
        end_date: date,
        topics: Mapping[str, str] = TOPIC_QUERIES,
        categories: Sequence[str] = AI_CATEGORIES,
    ) -> Iterator[ArxivPaper]:
        """Papers first submitted from `start_date` through `end_date` (GMT)
        matching any of `topics`, each yielded once, with every topic that
        found it."""
        cats = " OR ".join(f"cat:{c}" for c in categories)
        dates = f"submittedDate:[{start_date:%Y%m%d}0000 TO {end_date:%Y%m%d}2359]"
        found: dict[str, ArxivPaper] = {}
        for name, topic in topics.items():
            for paper in self.iter_search(f"({cats}) AND ({topic}) AND {dates}"):
                seen = found.get(paper.arxiv_id)
                found[paper.arxiv_id] = replace(
                    paper, topics=(*(seen.topics if seen else ()), name)
                )
        yield from sorted(found.values(), key=lambda p: (p.published, p.arxiv_id))

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


_VERSIONED_ID = re.compile(r"abs/(?P<id>.+?)v(?P<version>\d+)$")


def _text(entry: ET.Element, tag: str) -> str:
    return " ".join((entry.findtext(tag) or "").split())


def _timestamp(value: str | None) -> datetime:
    if not value:
        raise SourceAPIError("arxiv entry without a date", 200, "arxiv")
    return datetime.fromisoformat(value)


def parse_papers(feed: str) -> list[ArxivPaper]:
    """The entries of a search feed; raises on an error feed."""
    try:
        root = ET.fromstring(feed)
    except ET.ParseError as exc:
        raise SourceAPIError(f"Invalid XML from arxiv: {exc}", 200, "arxiv") from None
    error = _feed_error(root)
    if error:
        raise SourceAPIError(f"arxiv query failed: {error}", 200, "arxiv")
    papers = []
    for entry in root.iter(f"{_ATOM}entry"):
        match = _VERSIONED_ID.search(entry.findtext(f"{_ATOM}id") or "")
        if match is None:
            raise SourceAPIError("arxiv entry with an unversioned id", 200, "arxiv")
        primary = entry.find(f"{_ARXIV}primary_category")
        papers.append(
            ArxivPaper(
                arxiv_id=match["id"],
                version=int(match["version"]),
                title=_text(entry, f"{_ATOM}title"),
                abstract=_text(entry, f"{_ATOM}summary"),
                published=_timestamp(entry.findtext(f"{_ATOM}published")),
                updated=_timestamp(entry.findtext(f"{_ATOM}updated")),
                primary_category=None if primary is None else primary.get("term"),
                categories=tuple(
                    c.get("term", "") for c in entry.iter(f"{_ATOM}category")
                ),
                authors=tuple(
                    _text(a, f"{_ATOM}name") for a in entry.iter(f"{_ATOM}author")
                ),
            )
        )
    return papers


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
