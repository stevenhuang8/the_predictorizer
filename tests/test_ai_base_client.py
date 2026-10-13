"""Tests for the AI sources' shared HTTP client.

A fake session and fake clock keep them offline and instant, except the
thread-safety test, which uses the real clock with a 0.1s rate-limit window.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import requests

from eco_prediction.ai.sources import (
    BaseAPIClient,
    SourceAPIError,
    optional_env,
    require_env,
)
from eco_prediction.data.fred_client import RateLimiter
from eco_prediction.db.connection import ConfigError

TOKEN = "secret-token-123"


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    """Replays queued responses (or raises queued exceptions) and records calls."""

    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def get(
        self,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ExampleClient(BaseAPIClient):
    """A keyed source: token in a header and in the query, like real APIs do."""

    SOURCE = "example"
    BASE_URL = "https://api.example.com/v1/"
    MAX_CALLS = 2
    PERIOD = 10.0

    @classmethod
    def from_env(cls, **kwargs: Any) -> ExampleClient:
        return cls(api_key=require_env("ECO_TEST_EXAMPLE_TOKEN"), **kwargs)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _auth_params(self) -> dict[str, str]:
        return {"key": self.api_key or ""}

    def _error_message(self, resp: requests.Response) -> str:
        return str(resp.json().get("message", resp.text))

    def thing(self, thing_id: int) -> Any:
        return self._get_json(f"things/{thing_id}", {"verbose": "1"})


def make_client(
    *responses: FakeResponse | Exception, **kwargs: Any
) -> tuple[ExampleClient, FakeSession, FakeClock]:
    """`rate_limiter` defaults to a roomy one; pass None for the class limit."""
    session = FakeSession(*responses)
    clock = FakeClock()
    kwargs.setdefault("rate_limiter", RateLimiter(max_calls=1000, period=60))
    client = ExampleClient(
        api_key=TOKEN,
        session=session,  # type: ignore[arg-type]
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )
    return client, session, clock


def ok(body: Any = None) -> FakeResponse:
    return FakeResponse(body={"ok": True} if body is None else body)


# --- requests ---------------------------------------------------------------


def test_get_json_builds_url_and_adds_credentials() -> None:
    client, session, clock = make_client(ok({"id": 7}))

    assert client.thing(7) == {"id": 7}
    call = session.calls[0]
    assert call["url"] == "https://api.example.com/v1/things/7"
    assert call["params"] == {"verbose": "1", "key": TOKEN}
    assert call["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert clock.sleeps == []


def test_absolute_url_is_used_as_is() -> None:
    """Pagination links (GitHub's and Hugging Face's `Link` header) are absolute."""
    client, session, _ = make_client(ok())

    client._get("https://api.example.com/v1/things?page=2")

    assert session.calls[0]["url"] == "https://api.example.com/v1/things?page=2"


def test_get_returns_raw_response_for_non_json_sources() -> None:
    client, _, _ = make_client(FakeResponse(body="<feed>atom</feed>"))

    assert client._get("query").text == "<feed>atom</feed>"


def test_invalid_json_raises() -> None:
    client, _, _ = make_client(FakeResponse(body=ValueError("not json")))

    with pytest.raises(SourceAPIError, match="Invalid JSON from example"):
        client.thing(1)


def test_keyless_client_from_env_needs_no_configuration() -> None:
    class KeylessClient(BaseAPIClient):
        BASE_URL = "https://open.example.com"

    with KeylessClient.from_env() as client:
        assert client.api_key is None
        assert client._redact("nothing to hide") == "nothing to hide"
        assert client.base_url == "https://open.example.com"


def test_keyed_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECO_TEST_EXAMPLE_TOKEN", "from-env")

    with ExampleClient.from_env() as client:
        assert client.api_key == "from-env"


def test_missing_required_env_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ECO_TEST_EXAMPLE_TOKEN", raising=False)

    with pytest.raises(ConfigError, match="ECO_TEST_EXAMPLE_TOKEN is not set"):
        ExampleClient.from_env()


def test_optional_env_treats_empty_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECO_TEST_OPTIONAL", "")
    assert optional_env("ECO_TEST_OPTIONAL") is None
    monkeypatch.setenv("ECO_TEST_OPTIONAL", "x")
    assert optional_env("ECO_TEST_OPTIONAL") == "x"


# --- rate limiting ----------------------------------------------------------


def test_requests_wait_once_the_window_is_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, session, clock = make_client(ok(), ok(), ok(), rate_limiter=None)

    with caplog.at_level(logging.INFO, logger="eco_prediction.ai.sources"):
        for i in range(3):
            client.thing(i)

    # MAX_CALLS = 2 per 10s: the third request waits out the window.
    assert clock.sleeps == [10.0]
    assert len(session.calls) == 3
    assert "example: rate limit of 2 requests per 10s reached; waiting 10.0s" in (
        caplog.text
    )


def test_injected_rate_limiter_overrides_class_limit() -> None:
    clock = FakeClock()
    shared = RateLimiter(max_calls=1, period=5.0, clock=clock, sleep=clock.sleep)
    client, _, _ = make_client(ok(), ok(), rate_limiter=shared)

    client.thing(1)
    client.thing(2)

    assert clock.sleeps == [5.0]


# --- retries ----------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        FakeResponse(429, {"message": "slow down"}),
        FakeResponse(500, {"message": "oops"}),
        FakeResponse(502, {"message": "bad gateway"}),
        FakeResponse(503, {"message": "unavailable"}),
        FakeResponse(504, {"message": "gateway timeout"}),
        requests.Timeout("read timed out"),
        requests.ConnectionError("connection reset"),
    ],
)
def test_transient_failures_are_retried(failure: FakeResponse | Exception) -> None:
    client, session, clock = make_client(failure, failure, ok({"id": 1}))

    assert client.thing(1) == {"id": 1}
    assert len(session.calls) == 3
    assert clock.sleeps == [1.0, 2.0]  # exponential: base * 2**(attempt-1)


def test_retries_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    client, _, _ = make_client(FakeResponse(503, {"message": "busy"}), ok())

    with caplog.at_level(logging.WARNING, logger="eco_prediction.ai.sources"):
        client.thing(1)

    assert "example things/1: 503: busy; retrying in 1.0s (attempt 2 of 6)" in (
        caplog.text
    )


def test_backoff_is_capped_and_honors_retry_after() -> None:
    failures = [FakeResponse(500, {"message": "x"}) for _ in range(4)]
    failures[1] = FakeResponse(429, {"message": "x"}, headers={"Retry-After": "30"})
    client, _, clock = make_client(*failures, ok(), backoff_max=3.0)

    client.thing(1)

    # 1, then 2 raised to Retry-After's 30, then 4 and 8 capped at 3.
    assert clock.sleeps == [1.0, 30.0, 3.0, 3.0]


def test_client_error_raises_immediately_without_retry() -> None:
    client, session, clock = make_client(FakeResponse(404, {"message": "no thing"}))

    with pytest.raises(SourceAPIError) as excinfo:
        client.thing(99)

    assert excinfo.value.status_code == 404
    assert excinfo.value.source == "example"
    assert "example things/99 failed (404): no thing" in str(excinfo.value)
    assert len(session.calls) == 1
    assert clock.sleeps == []


def test_gives_up_after_max_retries_and_redacts_secret() -> None:
    leak = requests.ConnectionError(f"https://api.example.com/v1?key={TOKEN}")
    client, session, _ = make_client(*[leak] * 3, max_retries=2)

    with pytest.raises(SourceAPIError, match="failed after 3 attempts") as excinfo:
        client.thing(1)

    assert TOKEN not in str(excinfo.value)
    assert "key=***" in str(excinfo.value)
    assert len(session.calls) == 3


def test_subclass_can_retry_extra_responses() -> None:
    """GitHub signals an exhausted rate limit with 403, not 429."""

    class GitHubLike(ExampleClient):
        def _is_retryable(self, resp: requests.Response) -> bool:
            if (
                resp.status_code == 403
                and resp.headers.get("X-RateLimit-Remaining") == "0"
            ):
                return True
            return super()._is_retryable(resp)

    session = FakeSession(
        FakeResponse(403, {"message": "rate limited"}, {"X-RateLimit-Remaining": "0"}),
        ok({"id": 1}),
        FakeResponse(403, {"message": "forbidden"}),
    )
    clock = FakeClock()
    client = GitHubLike(
        api_key=TOKEN,
        session=session,  # type: ignore[arg-type]
        rate_limiter=RateLimiter(max_calls=1000, period=60),
        clock=clock,
        sleep=clock.sleep,
    )

    assert client.thing(1) == {"id": 1}
    with pytest.raises(SourceAPIError, match=r"\(403\): forbidden"):
        client.thing(2)


# --- lifecycle --------------------------------------------------------------


def test_context_manager_closes_session() -> None:
    client, session, _ = make_client(ok())

    with client as c:
        assert c is client
        c.thing(1)

    assert session.closed


def test_context_manager_closes_session_on_error() -> None:
    client, session, _ = make_client(FakeResponse(404, {"message": "nope"}))

    with pytest.raises(SourceAPIError), client:
        client.thing(1)

    assert session.closed


# --- thread safety ----------------------------------------------------------


class EchoSession:
    """Thread-safe fake: answers each request with its own params."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def get(
        self,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        with self._lock:
            self.calls += 1
        return FakeResponse(body={"url": url, "verbose": params["verbose"]})

    def close(self) -> None:
        pass


def test_concurrent_requests_share_the_rate_limit() -> None:
    """20 requests from 4 threads at 5 per 0.1s need at least 3 full windows."""
    session = EchoSession()
    client = ExampleClient(
        api_key=TOKEN,
        session=session,  # type: ignore[arg-type]
        rate_limiter=RateLimiter(max_calls=5, period=0.1),
    )

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(client.thing, range(20)))
    elapsed = time.monotonic() - start

    assert session.calls == 20
    assert [r["url"] for r in results] == [
        f"https://api.example.com/v1/things/{i}" for i in range(20)
    ]
    assert elapsed >= 0.3
