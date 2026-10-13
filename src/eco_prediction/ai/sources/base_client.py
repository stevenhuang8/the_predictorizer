"""Shared HTTP plumbing for the AI-domain API clients.

Usage, in a source module:

    class ArxivClient(BaseAPIClient):
        SOURCE = "arxiv"
        BASE_URL = "https://export.arxiv.org/api"
        MAX_CALLS, PERIOD = 1, 3.0  # arXiv asks for one request every 3 seconds

        def search(self, query: str) -> str:
            return self._get("query", {"search_query": query}).text

    with ArxivClient.from_env() as arxiv:
        feed = arxiv.search("cat:cs.LG")

`BaseAPIClient` follows `data/fred_client.py`: every request waits on a
sliding-window `RateLimiter` (the same class FRED uses, shared across threads)
and is retried with exponential backoff on 429, 5xx, timeouts and connection
errors, honoring `Retry-After`. Other errors raise `SourceAPIError` at once.

Subclasses set `SOURCE`, `BASE_URL` and their rate limit, and override hooks
only where the API differs:
- `_auth_params` / `_auth_headers`: credentials added to every request
- `_is_retryable`: e.g. GitHub's 403 when its rate limit is exhausted
- `_retry_after`: a server's own hint for how long to wait
- `_error_message`: pull the message out of an error body
- `from_env`: load an API key; the default builds a keyless client

`_get` returns the raw response, since arXiv answers in Atom XML and Epoch in
CSV; `_get_json` parses JSON. Only GET is supported, which covers every source
in the PRD. Secrets passed as `api_key` are redacted from errors and logs.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, ClassVar, Self

import requests
from dotenv import load_dotenv

from eco_prediction.data.fred_client import RateLimiter
from eco_prediction.db.connection import ConfigError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class SourceAPIError(Exception):
    """An API rejected the request, or it still failed after all retries."""

    def __init__(
        self, message: str, status_code: int | None = None, source: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.source = source


def require_env(name: str) -> str:
    """The value of environment variable `name` (`.env` included), or ConfigError."""
    value = optional_env(name)
    if not value:
        raise ConfigError(f"{name} is not set (see .env.example)")
    return value


def optional_env(name: str) -> str | None:
    """The value of `name`, or None when unset or empty (e.g. an optional token)."""
    load_dotenv()
    return os.environ.get(name) or None


class BaseAPIClient:
    SOURCE: ClassVar[str] = "api"
    BASE_URL: ClassVar[str] = ""
    # Default: at most MAX_CALLS requests per PERIOD seconds, per client.
    MAX_CALLS: ClassVar[int] = 60
    PERIOD: ClassVar[float] = 60.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        rate_limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.session = session or requests.Session()
        self._sleep = sleep
        # The default limiter sleeps through `_wait_for_rate_limit` so the wait
        # is logged. An injected one (e.g. shared between clients) isn't.
        self.rate_limiter = rate_limiter or RateLimiter(
            self.MAX_CALLS, self.PERIOD, clock=clock, sleep=self._wait_for_rate_limit
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> Self:
        """Build a client from the environment. Keyed sources override this."""
        return cls(**kwargs)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- hooks for subclasses -------------------------------------------------

    def _auth_params(self) -> dict[str, str]:
        return {}

    def _auth_headers(self) -> dict[str, str]:
        return {}

    def _is_retryable(self, resp: requests.Response) -> bool:
        return resp.status_code in RETRYABLE_STATUS

    def _retry_after(self, resp: requests.Response) -> float | None:
        """Seconds the server asked us to wait, from `Retry-After` (in seconds)."""
        try:
            return float(resp.headers["Retry-After"])
        except (KeyError, ValueError):
            return None

    def _error_message(self, resp: requests.Response) -> str:
        return resp.text[:500]

    # --- requests -------------------------------------------------------------

    def _get_json(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        resp = self._get(endpoint, params, headers=headers)
        try:
            return resp.json()
        except ValueError:
            raise SourceAPIError(
                f"Invalid JSON from {self.SOURCE} {endpoint}", 200, self.SOURCE
            ) from None

    def _get(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> requests.Response:
        """GET `endpoint` (relative to `base_url`, or an absolute URL such as a
        pagination link) with rate limiting and retries; returns the 200 response.
        """
        url = self._url(endpoint)
        query = {**(params or {}), **self._auth_params()}
        all_headers = {**(headers or {}), **self._auth_headers()}
        attempts = self.max_retries + 1
        last_error = ""
        retry_after: float | None = None
        for attempt in range(attempts):
            if attempt:
                delay = self._backoff(attempt, retry_after)
                logger.warning(
                    "%s %s: %s; retrying in %.1fs (attempt %d of %d)",
                    self.SOURCE,
                    endpoint,
                    last_error,
                    delay,
                    attempt + 1,
                    attempts,
                )
                self._sleep(delay)
                retry_after = None
            self.rate_limiter.acquire()
            logger.debug("%s GET %s", self.SOURCE, self._redact(url))
            try:
                resp = self.session.get(
                    url, params=query, headers=all_headers, timeout=self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                # requests' messages can embed the full URL, query key included.
                last_error = self._redact(f"{type(exc).__name__}: {exc}")
                continue

            if resp.status_code == 200:
                return resp

            message = self._redact(self._error_message(resp))
            if not self._is_retryable(resp):
                raise SourceAPIError(
                    f"{self.SOURCE} {endpoint} failed ({resp.status_code}): {message}",
                    resp.status_code,
                    self.SOURCE,
                )
            last_error = f"{resp.status_code}: {message}"
            retry_after = self._retry_after(resp)

        raise SourceAPIError(
            f"{self.SOURCE} {endpoint} failed after {attempts} attempts ({last_error})",
            source=self.SOURCE,
        )

    def _url(self, endpoint: str) -> str:
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self.backoff_base * 2 ** (attempt - 1), self.backoff_max)
        return max(delay, retry_after or 0.0)

    def _wait_for_rate_limit(self, seconds: float) -> None:
        logger.info(
            "%s: rate limit of %d requests per %gs reached; waiting %.1fs",
            self.SOURCE,
            self.MAX_CALLS,
            self.PERIOD,
            seconds,
        )
        self._sleep(seconds)

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text
