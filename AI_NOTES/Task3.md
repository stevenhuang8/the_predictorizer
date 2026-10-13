# AI Task 3: AI package skeleton and shared HTTP client

Done on 2026-10-12.

## Quick reference

```sh
uv run pytest tests/test_ai_base_client.py
```

```python
class ArxivClient(BaseAPIClient):
    SOURCE = "arxiv"
    BASE_URL = "https://export.arxiv.org/api"
    MAX_CALLS, PERIOD = 1, 3.0

    def search(self, query: str) -> str:
        return self._get("query", {"search_query": query}).text
```

---

## What I did

### `src/eco_prediction/ai/` and `ai/sources/`
- `ai/__init__.py`: the package. Nothing outside it imports it.
- `ai/sources/__init__.py` re-exports `BaseAPIClient`, `SourceAPIError`, `require_env` and `optional_env`.

### `ai/sources/base_client.py`
- **`BaseAPIClient`.** A subclass sets `SOURCE`, `BASE_URL`, `MAX_CALLS` and `PERIOD` (default 60 per 60s), then writes its endpoint methods on top of two helpers:
  - `_get(endpoint, params, headers=...)` returns the raw 200 response.
  - `_get_json(...)` parses it, and raises `SourceAPIError` on invalid JSON.
  - `endpoint` is relative to `base_url`, or an absolute URL. Absolute URLs are for the `Link`-header pagination of GitHub and Hugging Face.
- **Rate limiting:** a sliding window, using the `RateLimiter` class that FRED uses (thread-safe, with a lock).
  - The default limiter is per client. Pass `rate_limiter=` to share one between clients, for example several GitHub clients on one token.
- **Retries:** the same rules as FRED.
  - Retried: 429, 500, 502, 503, 504, timeouts and connection errors.
  - Backoff is `backoff_base * 2**(attempt-1)`, capped at `backoff_max`, and raised to `Retry-After` when the server sends one.
  - Any other status raises `SourceAPIError(message, status_code, source)` at once.
- **Hooks**, for subclasses to override where their API differs:
  - `_auth_params` and `_auth_headers`: credentials sent with every request.
  - `_is_retryable`: for example GitHub, which signals an exhausted rate limit with a 403.
  - `_retry_after`: the server's hint for how long to wait.
  - `_error_message`: pulls the message out of an error body.
  - `from_env`: loads the API key.
- **Logging** (`logging.getLogger`):
  - DEBUG for each request, with the URL redacted.
  - WARNING for each retry, with the reason, delay and attempt number.
  - INFO when the default rate limiter makes a request wait.
- **Secrets:** `api_key` is replaced with `***` in errors and logs.
- **`require_env(name)` / `optional_env(name)`:** load `.env`. `require_env` raises the existing `ConfigError`. `optional_env` returns `None` for an unset or empty variable, for tokens that only raise a limit, such as GitHub's.

### `tests/test_ai_base_client.py` (25 tests, not tagged `economy`)
- **Requests:**
  - URL building, and credentials in both the query and the headers.
  - Absolute URLs pass through unchanged.
  - Raw text from `_get`.
  - Invalid JSON raises.
- **Configuration:** a keyless `from_env`, a keyed `from_env`, a missing key raising `ConfigError`, and empty treated as unset.
- **Rate limiting:**
  - The third request in a 2-per-10s window waits 10s, and the wait is logged.
  - An injected limiter overrides the class limit.
- **Retries:**
  - Each of 429, 500, 502, 503, 504, a timeout and a connection error is retried, with 1s and 2s backoff.
  - Retries are logged.
  - The backoff cap and `Retry-After` both apply.
  - A 404 raises at once, with no retry and no sleep.
  - The client gives up after `max_retries`, and the token is redacted from the error.
  - A subclass's `_is_retryable` can add GitHub's 403-when-rate-limited.
- **Context manager:** closes the session, including when an error is raised inside the block.
- **Thread safety:** 20 requests from 4 threads through one 5-per-0.1s limiter, on the real clock.
  - Every request gets its own response back.
  - The run takes at least 3 windows (0.3s).

## Where I changed the task's plan
- **No abstract `_make_request`.** The base class does the request loop, which is the part every source shares. Subclasses add endpoint methods and override small hooks. If each source implemented `_make_request` itself, all six would copy the retry loop.
- **`from_env` is not abstract.** arXiv, Epoch and Polymarket need no key, so by default it builds a keyless client. Keyed sources override it.
- **No tenacity.** The retry loop is written out, as in FRED. That avoids a new dependency, the behavior is the same as the economy client's, and tests control the sleep directly.
- **`RateLimiter` is imported from `data/fred_client.py`, not copied or moved.** Copying it would mean two versions to keep in step. Moving it to `utils/` would mean editing economy code. Importing it changes no economy code.
- **`_get` returns the raw response,** not parsed JSON. arXiv answers in Atom XML, and Epoch's datasets are CSV.

## Tradeoffs
- **GET only.** Every source in the PRD is read through GET. A POST (for example GitHub GraphQL) would need a `method` argument added.
- **`Retry-After` isn't capped**, as in FRED. A server asking for an hour gets an hour. `backoff_max` only caps the client's own backoff.
- **The AI package depends on an economy module** (`data.fred_client.RateLimiter`). If FRED's limiter ever changes, these clients change with it. The FRED tests cover the class.
- **Rate-limit waits are logged only for the default limiter.** An injected limiter waits without logging.
- **The `requests.Session` is shared by all threads.** That is the common practice, since its connection pool is thread-safe, but requests doesn't officially promise it. The test checks that responses don't get crossed between threads.
- **No jitter in the backoff.** The behavior matches FRED. With many threads, retries are only spread out by the shared rate limiter.

## Checks
- `uv run pytest tests/test_ai_base_client.py`: 25 passed.
- `uv run pytest`: 358 passed, exit 0. That is 317 economy, 16 AI schema and 25 new. The task 1 safety net reported nothing missing or skipped.
- `ruff check`, `ruff format --check` and `mypy` pass on `src/eco_prediction/ai` and the new test file.
- No live API calls yet. The first real use comes with the arXiv client (AI task 4).
