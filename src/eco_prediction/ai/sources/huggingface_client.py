"""Client for the Hugging Face Hub API: new public models per month, and
model cards.

Usage:
    from eco_prediction.ai.sources.huggingface_client import HuggingFaceClient

    with HuggingFaceClient.from_env() as hf:
        counts = list(hf.count_months([date(2026, 8, 1), date(2026, 9, 1)]))
        for model in hf.iter_author_models("Qwen"):          # AI task 17
            card = hf.card_as_of(model.id, model.created_at + timedelta(days=30))

Model cards (`card_as_of`) are read as they stood at a moment: the README at
the latest commit on or before it (commits are listed newest first, 50 a
page, through the `Link` header). A model's first commit rarely has a
README. Gated models (Llama, Gemma) refuse both the commit list and the file
without an approved token, and a model may have no README at all; either
way `card_as_of` returns None.

The Hub has no count endpoint, so counting means listing models newest first
(`sort=createdAt&direction=-1`, 1,000 per page, following the `Link` header's
cursor) and tallying `createdAt` by UTC month until the listing passes the
earliest month asked for. At ~4,400 new models a day (October 2026), a month
is ~130 pages.

What the count can and can't see:
- Only models that are public and still exist today. Deleted models (much of
  it spam) are gone from the listing, so older months undercount what was
  uploaded then. Models made public later count in the month they were created.
- `createdAt` is the same instant, 2022-03-02T23:29:04Z, for every model older
  than that (HF's platform migration), so months before April 2022 can't be
  counted. FIRST_MONTH enforces that.

Rate limit, for the "api" bucket (listings, commit histories): 500 requests
per 5-minute fixed window without a token, per IP; 1,000 for a free account's
`HF_TOKEN` (huggingface.co/docs/hub/rate-limits, September 2025). The client
stays at 90% of that: 450, or 900 with a token. On a 429 it waits the `t=`
seconds the `ratelimit` header gives. README downloads (`/raw/`) count
against the separate "resolvers" bucket (3,000 / 5,000), which isn't the
bottleneck. The limit is per IP or per token, so two processes running at
once share it and slow each other down.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Self

import requests

from eco_prediction.ai.months import MonthlyCount, month_start
from eco_prediction.ai.sources.base_client import (
    BaseAPIClient,
    SourceAPIError,
    optional_env,
)

BASE_URL = "https://huggingface.co/api"
FIRST_MONTH = date(2022, 4, 1)
PAGE_SIZE = 1000
FILE_URL = "https://huggingface.co/{model}/raw/{revision}/README.md"
NOT_AVAILABLE = frozenset({401, 403, 404})  # gated, private, or no such file

# The listing is sorted by createdAt but paged by a cursor on the model's
# internal id. In case the two orders disagree slightly, a month is only
# closed once the listing is this far before its first day.
ORDER_SLACK = timedelta(days=2)

_RESET = re.compile(r"\bt=(\d+)")


@dataclass(frozen=True)
class HFModel:
    id: str  # 'author/name'
    created_at: datetime
    pipeline_tag: str | None = None
    tags: tuple[str, ...] = ()
    gated: bool = False


@dataclass(frozen=True)
class Card:
    text: str
    revision: str  # commit id
    committed_at: datetime


class HuggingFaceClient(BaseAPIClient):
    SOURCE = "huggingface"
    BASE_URL = BASE_URL
    MAX_CALLS = 450  # anonymous; TOKEN_MAX_CALLS with HF_TOKEN
    TOKEN_MAX_CALLS = 900
    PERIOD = 300.0

    @classmethod
    def from_env(cls, **kwargs: Any) -> Self:
        return cls(api_key=optional_env("HF_TOKEN"), **kwargs)

    def _rate_limit(self) -> tuple[int, float]:
        return (self.TOKEN_MAX_CALLS if self.api_key else self.MAX_CALLS), self.PERIOD

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _retry_after(self, resp: requests.Response) -> float | None:
        """`Retry-After`, else on a 429 the window reset from `ratelimit: ...;t=N`."""
        hint = super()._retry_after(resp)
        if hint is None and resp.status_code == 429:
            match = _RESET.search(resp.headers.get("ratelimit", ""))
            if match:
                hint = float(match.group(1))
        return hint

    def iter_created(self) -> Iterator[datetime]:
        """`createdAt` of every public model, newest first, one page at a time."""
        url: str | None = "models"
        params: dict[str, str] | None = {
            "sort": "createdAt",
            "direction": "-1",
            "limit": str(PAGE_SIZE),
            "expand[]": "createdAt",
        }
        while url:
            resp = self._get(url, params)
            try:
                page = resp.json()
            except ValueError:
                raise SourceAPIError(
                    "Invalid JSON from huggingface models", 200, self.SOURCE
                ) from None
            for model in page:
                yield _parse_created(model)
            # The next link carries every parameter, cursor included.
            url, params = resp.links.get("next", {}).get("url"), None

    def _pages(self, url: str | None, params: dict[str, str] | None) -> Iterator[Any]:
        """Each JSON page of a listing, following the `Link` header."""
        while url:
            resp = self._get(url, params)
            try:
                yield resp.json()
            except ValueError:
                raise SourceAPIError(
                    f"Invalid JSON from huggingface {url}", 200, self.SOURCE
                ) from None
            url, params = resp.links.get("next", {}).get("url"), None

    def iter_author_models(self, author: str) -> Iterator[HFModel]:
        """Every public model of `author` (an organization or user)."""
        params = {
            "author": author,
            "limit": str(PAGE_SIZE),
            "expand[]": ["createdAt", "pipeline_tag", "tags", "gated"],
        }
        for page in self._pages("models", params):  # type: ignore[arg-type]
            for model in page:
                yield HFModel(
                    id=model["id"],
                    created_at=_parse_created(model),
                    pipeline_tag=model.get("pipeline_tag"),
                    tags=tuple(model.get("tags") or ()),
                    gated=bool(model.get("gated")),
                )

    def commits(self, model_id: str) -> list[tuple[str, datetime]] | None:
        """(commit id, date) of every commit on main, newest first; None if
        the model refuses (gated or private)."""
        out: list[tuple[str, datetime]] = []
        try:
            for page in self._pages(f"models/{model_id}/commits/main", None):
                out.extend((c["id"], datetime.fromisoformat(c["date"])) for c in page)
        except SourceAPIError as exc:
            if exc.status_code in NOT_AVAILABLE:
                return None
            raise
        return out

    def readme(self, model_id: str, revision: str) -> str | None:
        """The model's README.md at `revision`, or None if there isn't one."""
        try:
            resp = self._get(FILE_URL.format(model=model_id, revision=revision))
        except SourceAPIError as exc:
            if exc.status_code in NOT_AVAILABLE:
                return None
            raise
        return resp.text

    def card_as_of(self, model_id: str, when: datetime) -> Card | None:
        """The README as of `when` (the latest commit on or before it)."""
        history = self.commits(model_id)
        if not history:
            return None
        known = [(sha, at) for sha, at in history if at <= when]
        if not known:
            return None
        sha, at = known[0]
        text = self.readme(model_id, sha)
        return None if text is None else Card(text, sha, at)

    def count_months(self, months: Iterable[date]) -> Iterator[MonthlyCount]:
        """New models in each of `months`, newest month first, each yielded as
        soon as the listing has passed it. One scan covers all of them."""
        wanted = sorted({month_start(m) for m in months}, reverse=True)
        if not wanted:
            return
        if wanted[-1] < FIRST_MONTH:
            raise ValueError(
                f"Hugging Face createdAt can't count months before {FIRST_MONTH}"
            )
        counts = dict.fromkeys(wanted, 0)
        pending = list(wanted)  # newest first
        for created in self.iter_created():
            day = created.date()
            while pending and day < pending[0] - ORDER_SLACK:
                month = pending.pop(0)
                yield MonthlyCount(month, counts[month])
            if not pending:
                return
            month = month_start(day)
            if month in pending:
                counts[month] += 1
        # The listing ran out (only possible in tests or for the oldest months).
        for month in pending:
            yield MonthlyCount(month, counts[month])


def _parse_created(model: dict[str, Any]) -> datetime:
    try:
        return datetime.fromisoformat(model["createdAt"])
    except (KeyError, TypeError, ValueError):
        raise SourceAPIError(
            f"huggingface model without a valid createdAt: {model.get('id')!r}",
            200,
            "huggingface",
        ) from None
