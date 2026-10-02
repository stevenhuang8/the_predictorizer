# Task 6: FRED/ALFRED API client

Done on 2026-10-01.

## Quick reference

```python
from eco_prediction.data.fred_client import FREDClient

with FREDClient.from_env() as fred:                 # reads FRED_API_KEY from .env
    info = fred.get_series_info("CPIAUCSL")         # name, units, frequency, seasonal adjustment
    obs = fred.get_series_observations("CPIAUCSL")  # every vintage, as Observation rows
    dates = fred.get_vintage_dates("CPIAUCSL")      # every publish/revision date
```

```sh
uv run pytest tests/test_fred_client.py
```

---

## What I did

### `src/eco_prediction/data/fred_client.py`
- `Observation(observed_at, as_of, value)` matches the `observations` table. `as_of` is the date the value was published (ALFRED's `realtime_start`). FRED's `.` (missing) becomes `None`.
- `SeriesInfo` matches the `series` table, plus the observation range, last-updated time and notes.
- `get_series_observations()` fetches **every vintage** by default. You can narrow it with `realtime_start`/`realtime_end`, ask for specific `vintage_dates`, and limit the periods with `observation_start`/`observation_end`.
- **Rate limit:** at most 120 requests in any rolling 60 seconds. It's thread-safe.
- **Retries:** 429, 5xx, timeouts and connection errors are retried with exponential backoff (1s, 2s, 4s… capped at 60s). It respects `Retry-After`. Any other error raises `FREDAPIError` right away, with FRED's own message and the status code.
- **The API key is removed from error messages.** `requests` puts the full URL, key included, in its exceptions.

### Things to remember (found in Task 7)
- **FRED allows at most 2000 vintages per request.** The daily yield spreads (T10Y2Y, T10Y3M) have more than 3100. The client looks up the vintage dates first and splits the request into windows when needed. You don't have to do anything.
- **FRED dates values to the start of the window.** If you ask for `realtime_start=2020-01-01`, a value published in 1994 comes back with `as_of=2020-01-01`. The client removes those repeats between its windows. If you call it with your own `realtime_start`, the rows from before that date will carry that date.
- **`series/vintagedates` returns 500 (not an empty list)** when no vintage falls in the requested range. `get_vintage_dates()` always fetches the full list and filters it locally, so this never happens.

---

## Checks
- Live: the key works, and all 12 V1 series fetch.
- Chunked CPIAUCSL (forced into 7 windows) is identical to the single-request result (3362 rows).
