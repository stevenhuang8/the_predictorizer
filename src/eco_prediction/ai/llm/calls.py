"""Logging LLM calls to `llm_calls` (migration 014, AI task 16).

    from eco_prediction.ai.llm.calls import LLMCall, cached_call, context_hash, log_call

    key = context_hash(model_id, prompt, {"max_tokens": 2000})
    cached = cached_call(conn, model_id, key)        # a backtest re-run: no new call
    if cached is None:
        ...                                          # call the API
        call_id = log_call(conn, LLMCall(model_id, prompt, raw_response=text,
                                         context_hash=key, input_tokens=..., ...))
    attach_forecast(conn, [call_id], forecast_id)    # once the forecast is stored

Every call is logged, failed ones too (`error` set, `raw_response` usually
None), so spend and failure rates can be counted from the table. The
context hash covers everything that decides the response: the model, the
prompt and the request parameters. Two calls with the same hash asked the
same thing, so a logged successful response can stand in for a new call.
Nothing here commits; the caller owns the transaction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json

__all__ = [
    "LLMCall",
    "LoggedCall",
    "attach_forecast",
    "cached_call",
    "context_hash",
    "log_call",
]


def context_hash(
    model_id: str, prompt: str, params: Mapping[str, Any] | None = None
) -> str:
    """SHA-256 (hex) of the model, prompt and parameters, key order ignored."""
    payload = json.dumps(
        {"model": model_id, "prompt": prompt, "params": dict(params or {})},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class LLMCall:
    model_id: str
    prompt: str
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    error: str | None = None
    context_hash: str | None = None
    prompt_version: str | None = None
    as_of_date: date | None = None
    question_id: int | None = None
    forecast_id: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class LoggedCall(LLMCall):
    id: int = 0
    created_at: datetime | None = None


_COLUMNS = (
    "model_id", "prompt", "raw_response", "parsed_response", "error",
    "context_hash", "prompt_version", "as_of_date", "question_id", "forecast_id",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "cost_usd",
)  # fmt: skip


def log_call(conn: Connection, call: LLMCall) -> int:
    """Insert one call; returns its id. Needs a response or an error."""
    if call.raw_response is None and call.error is None:
        raise ValueError("a logged call needs a raw_response or an error")
    values = [getattr(call, c) for c in _COLUMNS]
    values[_COLUMNS.index("parsed_response")] = (
        None if call.parsed_response is None else Json(call.parsed_response)
    )
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO llm_calls ({', '.join(_COLUMNS)})"
            f" VALUES ({', '.join(['%s'] * len(_COLUMNS))}) RETURNING id",
            values,
        )
        return int(cur.fetchone()[0])  # type: ignore[index]


def cached_call(conn: Connection, model_id: str, key: str) -> LoggedCall | None:
    """The latest successful call with this model and context hash, if any."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {", ".join(_COLUMNS)}, cost_usd::float8, id, created_at
            FROM llm_calls
            WHERE model_id = %s AND context_hash = %s
              AND error IS NULL AND raw_response IS NOT NULL
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (model_id, key),
        )
        row = cur.fetchone()
    if row is None:
        return None
    fields = dict(zip(_COLUMNS, row[: len(_COLUMNS)]))
    fields["cost_usd"] = row[len(_COLUMNS)]
    return LoggedCall(**fields, id=int(row[-2]), created_at=row[-1])


def attach_forecast(
    conn: Connection, call_ids: Sequence[int], forecast_id: int
) -> None:
    """Link calls to the forecast they produced."""
    if not call_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE llm_calls SET forecast_id = %s WHERE id = ANY(%s)",
            (forecast_id, list(call_ids)),
        )
