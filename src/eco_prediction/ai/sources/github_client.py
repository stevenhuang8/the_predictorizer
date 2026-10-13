"""Client for GitHub's repository search: AI repo creation and star counts.

Usage:
    from eco_prediction.ai.sources.github_client import GitHubClient

    with GitHubClient.from_env() as gh:
        n = gh.count_new_repos(date(2024, 1, 1))  # AI repos created Jan 2024
        stars = gh.top_starred_total()            # stars on the top 1,000 now

Two measures, both over public, non-fork repos tagged with any of AI_TOPICS:
- **New repos in a month:** each repo counted once. Search can't OR `topic:`
  qualifiers ("Validation Failed"), but several of them in one query mean
  AND, so the union comes from inclusion-exclusion over every non-empty
  subset of topics: 7 `total_count` queries for 3 topics. A repo counts if it
  carries a topic *now*: topics added later, and repos deleted since, change
  past months.
- **Stars on the top 1,000:** the most-starred 1,000 repos across the topics,
  summed. Search returns at most 1,000 results per query, so each topic's top
  1,000 by stars is fetched (10 pages of 100) and merged by repo id; the
  overall top 1,000 is always among them. Only today's star counts exist, so
  this is a live snapshot that can't be backfilled.

Rate limits: search allows 30 requests a minute with a token (`GITHUB_TOKEN`)
and 10 without. An exhausted limit is a 403 (or 429) with
`X-RateLimit-Remaining: 0`; the client waits until `X-RateLimit-Reset`.
`Retry-After` (secondary limits) wins when present. A result marked
`incomplete_results` (the search timed out) is retried, never returned.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import date, timedelta
from itertools import combinations
from typing import Any, Self

import requests

from eco_prediction.ai.months import MonthlyCount, month_start, next_month
from eco_prediction.ai.sources.base_client import (
    BaseAPIClient,
    SourceAPIError,
    optional_env,
)

BASE_URL = "https://api.github.com"
AI_TOPICS = ("ai", "machine-learning", "deep-learning")
FIRST_MONTH = date(2017, 1, 1)  # GitHub added repository topics in January 2017
TOP_N = 1000
SEARCH_PAGE_SIZE = 100  # GitHub's maximum
SEARCH_RESULT_CAP = 1000  # GitHub returns no results past the 1,000th
INCOMPLETE_RETRIES = 3


class GitHubClient(BaseAPIClient):
    SOURCE = "github"
    BASE_URL = BASE_URL
    MAX_CALLS = 30  # search, with a token
    PERIOD = 60.0

    def __init__(self, *, now: Callable[[], float] = time.time, **kwargs: Any) -> None:
        self._now = now  # wall clock, for X-RateLimit-Reset (epoch seconds)
        super().__init__(**kwargs)

    @classmethod
    def from_env(cls, **kwargs: Any) -> Self:
        return cls(api_key=optional_env("GITHUB_TOKEN"), **kwargs)

    def _rate_limit(self) -> tuple[int, float]:
        return (self.MAX_CALLS if self.api_key else 10), self.PERIOD

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _is_retryable(self, resp: requests.Response) -> bool:
        if resp.status_code == 403 and _rate_limit_exhausted(resp):
            return True
        return super()._is_retryable(resp)

    def _retry_after(self, resp: requests.Response) -> float | None:
        hint = super()._retry_after(resp)
        if hint is None and _rate_limit_exhausted(resp):
            try:
                reset = float(resp.headers["X-RateLimit-Reset"])
            except (KeyError, ValueError):
                return None
            hint = max(reset - self._now(), 0.0) + 1.0
        return hint

    def _error_message(self, resp: requests.Response) -> str:
        try:
            return str(resp.json().get("message", resp.text[:500]))
        except ValueError:
            return resp.text[:500]

    # --- searches -------------------------------------------------------------

    def search_repositories(
        self, query: str, *, page: int = 1, per_page: int = 1, sort: str | None = None
    ) -> dict[str, Any]:
        """One page of repository search, retried while results are incomplete."""
        params = {"q": query, "per_page": str(per_page), "page": str(page)}
        if sort:
            params |= {"sort": sort, "order": "desc"}
        for attempt in range(INCOMPLETE_RETRIES + 1):
            body = self._get_json("search/repositories", params)
            if not body.get("incomplete_results"):
                if not isinstance(body.get("total_count"), int):
                    raise SourceAPIError(
                        f"github search has no total_count: {query!r}", 200, "github"
                    )
                return body  # type: ignore[no-any-return]
            if attempt < INCOMPLETE_RETRIES:
                self._sleep(self._backoff(attempt + 1, None))
        raise SourceAPIError(
            f"github search stayed incomplete after {INCOMPLETE_RETRIES + 1} "
            f"attempts: {query!r}",
            200,
            "github",
        )

    def count_repos(self, query: str) -> int:
        return int(self.search_repositories(query)["total_count"])

    def count_new_repos(self, month: date, topics: Sequence[str] = AI_TOPICS) -> int:
        """Repos created in `month` (UTC) carrying any of `topics`, each once."""
        if month_start(month) < FIRST_MONTH:
            raise ValueError(f"GitHub topics can't count months before {FIRST_MONTH}")
        created = created_range(month)
        total = 0
        for size in range(1, len(topics) + 1):
            sign = 1 if size % 2 else -1
            for subset in combinations(topics, size):
                total += sign * self.count_repos(topic_query(subset, created))
        return total

    def count_months(
        self, months: Iterable[date], topics: Sequence[str] = AI_TOPICS
    ) -> Iterator[MonthlyCount]:
        """New AI repos per month, each yielded as soon as it's counted."""
        for m in months:
            yield MonthlyCount(month_start(m), self.count_new_repos(m, topics))

    def top_starred(
        self, topics: Sequence[str] = AI_TOPICS, n: int = TOP_N
    ) -> list[tuple[int, int]]:
        """(repo id, stars) of the `n` most-starred repos with any of `topics`."""
        if n > SEARCH_RESULT_CAP:
            raise ValueError(f"search can't see past result {SEARCH_RESULT_CAP}")
        stars: dict[int, int] = {}
        pages = -(-n // SEARCH_PAGE_SIZE)
        for topic in topics:
            for page in range(1, pages + 1):
                body = self.search_repositories(
                    f"topic:{topic}", page=page, per_page=SEARCH_PAGE_SIZE, sort="stars"
                )
                items = body.get("items", [])
                for repo in items:
                    stars[int(repo["id"])] = int(repo["stargazers_count"])
                if len(items) < SEARCH_PAGE_SIZE:
                    break
        return sorted(stars.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def top_starred_total(
        self, topics: Sequence[str] = AI_TOPICS, n: int = TOP_N
    ) -> int:
        return sum(s for _, s in self.top_starred(topics, n))


def topic_query(topics: Iterable[str], created: str) -> str:
    return " ".join(f"topic:{t}" for t in topics) + f" created:{created}"


def created_range(month: date) -> str:
    first = month_start(month)
    last = next_month(first) - timedelta(days=1)
    return f"{first.isoformat()}..{last.isoformat()}"


def _rate_limit_exhausted(resp: requests.Response) -> bool:
    return resp.headers.get("X-RateLimit-Remaining") == "0"
