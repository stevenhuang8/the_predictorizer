"""Client for Polymarket: AI-tagged yes/no markets and their daily price history.

Usage:
    from eco_prediction.ai.sources.polymarket_client import PolymarketClient

    with PolymarketClient.from_env() as pm:
        markets = pm.ai_markets(min_volume=10_000)
        prices = daily_prices(pm.price_history(markets[0].yes_token), date.today())

Two public APIs, no key:
- **Gamma** (gamma-api.polymarket.com) lists events, each with its markets.
  `/events?tag_slug=ai` is paged 100 at a time by `offset`; open and closed
  events are listed separately. A market's `outcomes`, `outcomePrices` and
  `clobTokenIds` are JSON strings.
- **CLOB** (clob.polymarket.com) `/prices-history` gives a token's price over
  time; at `fidelity=1440` it's one point a day since the market opened,
  closed markets included.

Only binary markets (outcomes exactly Yes/No) are kept, and the probability
is the Yes token's price. A market resolved when it's closed with
`umaResolutionStatus = resolved` and its prices are 1/0 (yes) or 0/1 (no);
anything else (open, disputed, a 50/50 resolution) has no outcome.

Polymarket publishes no rate limit for these endpoints; the client keeps to
60 requests per 10 seconds. Python's default urllib User-Agent is refused
with a 403, but requests' is accepted.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from eco_prediction.ai.sources.base_client import BaseAPIClient, SourceAPIError

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
AI_TAG = "ai"
PAGE_SIZE = 100

Outcome = Literal["yes", "no"]


@dataclass(frozen=True)
class Market:
    id: str
    question: str
    description: str
    url: str
    yes_token: str
    volume: float
    opened_at: datetime | None
    closes_at: datetime | None  # scheduled end
    closed: bool
    closed_at: datetime | None
    outcome: Outcome | None
    yes_price: float | None  # current price, while open
    metadata: dict[str, Any] = field(default_factory=dict)


class PolymarketClient(BaseAPIClient):
    SOURCE = "polymarket"
    BASE_URL = GAMMA_URL
    MAX_CALLS = 60
    PERIOD = 10.0

    def iter_events(self, tag: str = AI_TAG) -> Iterator[dict[str, Any]]:
        """Every event with `tag`, open ones first, then closed ones."""
        for closed in ("false", "true"):
            offset = 0
            while True:
                page = self._get_json(
                    "events",
                    {
                        "tag_slug": tag,
                        "closed": closed,
                        "limit": PAGE_SIZE,
                        "offset": offset,
                    },
                )
                if not isinstance(page, list):
                    raise SourceAPIError(
                        f"polymarket events: expected a list, got {type(page).__name__}",
                        source=self.SOURCE,
                    )
                yield from page
                if len(page) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

    def ai_markets(self, min_volume: float = 0.0) -> list[Market]:
        """Binary markets in AI-tagged events with at least `min_volume` USD
        traded, each once, in listing order."""
        markets: dict[str, Market] = {}
        for event in self.iter_events():
            for raw in event.get("markets") or ():
                market = parse_market(raw, event)
                if market and market.volume >= min_volume:
                    markets.setdefault(market.id, market)
        return list(markets.values())

    def price_history(self, token: str) -> list[tuple[datetime, float]]:
        """(time, price) of `token`, one point a day, oldest first."""
        body = self._get_json(
            f"{CLOB_URL}/prices-history",
            {"market": token, "interval": "max", "fidelity": 1440},
        )
        try:
            points = [
                (datetime.fromtimestamp(int(p["t"]), UTC), float(p["p"]))
                for p in body["history"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceAPIError(
                f"polymarket prices-history for {token[:12]}...: bad body ({exc!r})",
                source=self.SOURCE,
            ) from None
        return sorted(points)


def parse_market(raw: Mapping[str, Any], event: Mapping[str, Any]) -> Market | None:
    """A `Market`, or None if it isn't a binary Yes/No market with tokens."""
    try:
        if _json_list(raw, "outcomes") != ["Yes", "No"]:
            return None
        market_id = str(raw["id"])
        question = str(raw["question"])
        tokens = _json_list(raw, "clobTokenIds")
        if not tokens:
            return None  # never tradable (e.g. an inactive placeholder): no prices
        yes_token = str(tokens[0])
        prices = [float(p) for p in _json_list(raw, "outcomePrices")]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SourceAPIError(
            f"polymarket market {raw.get('id')!r}: bad field ({exc!r})",
            source="polymarket",
        ) from None

    closed = bool(raw.get("closed"))
    status = raw.get("umaResolutionStatus")
    outcome: Outcome | None = None
    if closed and status == "resolved":
        if prices == [1.0, 0.0]:
            outcome = "yes"
        elif prices == [0.0, 1.0]:
            outcome = "no"
    closed_at = _time(raw.get("closedTime")) or (
        _time(raw.get("umaEndDate")) if closed else None
    )
    event_slug = event.get("slug")
    return Market(
        id=market_id,
        question=question,
        description=str(raw.get("description") or ""),
        url=(
            f"https://polymarket.com/event/{event_slug}"
            if event_slug
            else f"https://polymarket.com/market/{raw.get('slug')}"
        ),
        yes_token=yes_token,
        volume=float(raw.get("volumeNum") or 0.0),
        opened_at=_time(raw.get("startDate")),
        closes_at=_time(raw.get("endDate")),
        closed=closed,
        closed_at=closed_at if closed else None,
        outcome=outcome,
        yes_price=None if closed or not prices else prices[0],
        metadata={
            "event_id": event.get("id"),
            "event_title": event.get("title"),
            "slug": raw.get("slug"),
            "condition_id": raw.get("conditionId"),
            "yes_token": yes_token,
            "volume": float(raw.get("volumeNum") or 0.0),
            "uma_resolution_status": status,
            # When trading stopped, kept even when there's no clean outcome.
            "closed_at": closed_at.isoformat() if closed and closed_at else None,
            "outcome_prices": prices,
        },
    )


def daily_prices(
    history: list[tuple[datetime, float]], before: date
) -> dict[date, float]:
    """The last price of each UTC day before `before` (complete days only)."""
    days: dict[date, float] = {}
    for when, price in sorted(history):
        day = when.astimezone(UTC).date()
        if day < before:
            days[day] = price
    return days


def _json_list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError(f"{key} is not a list")
    return value


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
