"""Client for the Hugging Face Hub API: new public models per month.

Usage:
    from eco_prediction.ai.sources.huggingface_client import HuggingFaceClient

    with HuggingFaceClient.from_env() as hf:
        counts = list(hf.count_months([date(2026, 8, 1), date(2026, 9, 1)]))

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

Rate limit: 500 requests per 5-minute fixed window without a token (the
`ratelimit-policy` header). The client stays at 450 per 5 minutes, and on a
429 waits the `t=` seconds the `ratelimit` header gives. `HF_TOKEN` is sent
when set, but isn't needed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
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

# The listing is sorted by createdAt but paged by a cursor on the model's
# internal id. In case the two orders disagree slightly, a month is only
# closed once the listing is this far before its first day.
ORDER_SLACK = timedelta(days=2)

_RESET = re.compile(r"\bt=(\d+)")


class HuggingFaceClient(BaseAPIClient):
    SOURCE = "huggingface"
    BASE_URL = BASE_URL
    MAX_CALLS = 450
    PERIOD = 300.0

    @classmethod
    def from_env(cls, **kwargs: Any) -> Self:
        return cls(api_key=optional_env("HF_TOKEN"), **kwargs)

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
