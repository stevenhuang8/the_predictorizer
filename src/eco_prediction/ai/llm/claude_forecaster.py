"""The Claude forecaster: one structured API call per forecast (AI task 18).

    from eco_prediction.ai.llm.claude_forecaster import ClaudeForecaster
    from eco_prediction.ai.llm.context import build_context_bundle, load_question

    question = load_question(conn, 82)
    bundle = build_context_bundle(conn, question, date(2026, 10, 14))
    forecaster = ClaudeForecaster(conn=conn)        # logs every call to llm_calls
    forecaster.predict_numeric(question, bundle, question.horizon_months)
    # {'point_forecast': 31250.0, 'lower_80': 29800.0, 'upper_80': 32900.0,
    #  'rationale': '...', 'call_id': 17, 'cached': False}
    forecaster.predict_binary(question, bundle)     # {'probability': 0.35, ...}

**The request.** `claude-opus-5-5` (`MODEL`) with adaptive thinking at
`effort="high"` (forecasting is reasoning-heavy; this model's default is
medium). The reply is constrained to a JSON schema (`output_config.format`):
`rationale` first, so the model explains before it commits to numbers, then
`point_forecast` / `lower_80` / `upper_80`, or `probability`. A request
refused by a safety classifier is re-run server-side on Anthropic's
recommended fallback model (`fallbacks: "default"`); a refusal that survives
that raises.

**The prompt** (`build_prompt`, version `PROMPT_VERSION`) is the question
and resolution rule, the series' last 12 months, the baselines' forecasts,
and the retrieved documents, all as known on the bundle's `as_of_date`. The
system prompt tells the model the date and to reason only from what it was
given. That can't remove what the model learned in training: a backtest
before its training cutoff is contaminated, which AI task 19 deals with.

**Checks.** A numeric question whose series has no values known by the
cutoff raises `ForecastError` before any call: there's nothing to forecast
from. The answer must parse and be coherent: lower <= point <= upper,
a probability in [0, 1]. A malformed answer, a refusal, a truncated reply
or an API error is logged with its error and raises `ForecastError`
(`retryable` for rate limits, overload, timeouts and connection errors; the
SDK has already retried those `max_retries` times).

**The key** is `ANTHROPIC_API_KEY`, from the environment, then `.env.local`,
then `.env` (`anthropic_api_key`).

**Logging and cost.** With a `conn`, every call is logged to `llm_calls`:
prompt, raw and parsed reply, tokens (cache reads and writes included), and
`cost_usd` from `PRICES` (per attempt, so a fallback is priced at its own
model's rates). A call whose context hash (model, system prompt, prompt,
request parameters) matches an earlier successful one reuses that reply
instead of paying again (`reuse=False` to force a new call). Nothing here
commits.

`register_prompt_version(conn)` records the model and prompt version in
`model_versions`, for the forecasts Task 21 stores.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import anthropic
from dotenv import dotenv_values
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.llm.calls import LLMCall, cached_call, context_hash, log_call
from eco_prediction.ai.llm.context import ContextBundle, LLMQuestion
from eco_prediction.db.forecasts import get_or_create_model_version

MODEL = "claude-opus-5-5"
EFFORT = "high"
MAX_TOKENS = 16000
PROMPT_VERSION = "v1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DOCUMENT_CHARS = 1500  # per document in the prompt

# USD per million tokens: input, output, 5-minute cache write, cache read.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}

SYSTEM_PROMPT = """\
You are a careful, calibrated forecaster of AI progress: research output, \
model releases, benchmark records and training compute.

The forecast is being made on the date given in the request, and you know \
only what was published by the end of the day before it. Reason from the \
data and documents provided; do not use knowledge of anything that happened \
after that date, even if you have it.

Work as a good forecaster does: start from the base rates and the statistical \
baselines, then adjust for specific evidence in the documents, and say what \
moved you and by how much. Weigh how often a series moves at all, not only \
its trend. Be calibrated rather than bold: an 80% interval should contain \
the outcome four times in five, and a probability of 0.9 should come true \
nine times in ten.

Answer in the JSON format requested. Keep the rationale under 200 words."""

NUMERIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "point_forecast": {"type": "number"},
        "lower_80": {"type": "number"},
        "upper_80": {"type": "number"},
    },
    "required": ["rationale", "point_forecast", "lower_80", "upper_80"],
    "additionalProperties": False,
}
BINARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "probability": {"type": "number"},
    },
    "required": ["rationale", "probability"],
    "additionalProperties": False,
}

RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
)


def anthropic_api_key() -> str | None:
    """ANTHROPIC_API_KEY from the environment, else `.env.local`, else `.env`.

    An already-set variable wins, and a commented-out line counts as unset.
    None lets the SDK fall back to its other credential sources.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        for name in (".env.local", ".env"):
            value = dotenv_values(Path.cwd() / name).get("ANTHROPIC_API_KEY")
            if value:
                return value
    return os.environ.get("ANTHROPIC_API_KEY") or None


class ForecastError(Exception):
    """The forecast couldn't be made; the call (if any) is logged."""

    def __init__(
        self, message: str, *, retryable: bool = False, call_id: int | None = None
    ):
        super().__init__(message)
        self.retryable = retryable
        self.call_id = call_id


# --- the prompt -------------------------------------------------------------------------


def _number(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def build_prompt(bundle: ContextBundle) -> str:
    """The user message for `bundle`. Deterministic, so identical bundles hash alike."""
    q = bundle.question
    forecast_date = bundle.as_of_date + timedelta(days=1)
    parts = [
        (
            f"Today is {forecast_date:%Y-%m-%d}. You know what was published through "
            f"{bundle.as_of_date:%Y-%m-%d}."
        ),
        "<question>\n" + _question_text(q) + "\n</question>",
    ]
    if q.resolution_rule:
        parts.append(f"<resolution_rule>\n{q.resolution_rule}\n</resolution_rule>")
    if bundle.recent_values:
        rows = "\n".join(f"{m:%Y-%m}: {_number(v)}" for m, v in bundle.recent_values)
        parts.append(
            f'<recent_values series="{bundle.series_id}" name="{bundle.series_name}" '
            f'units="{bundle.units}">\n{rows}\n</recent_values>'
        )
    if bundle.baselines:
        lines = []
        for b in bundle.baselines:
            if b.probability is not None:
                lines.append(f"{b.name}: P(yes) = {b.probability:.3f}")
            else:
                interval = (
                    f", 80% interval [{_number(b.lower)}, {_number(b.upper)}]"
                    if b.lower is not None and b.upper is not None
                    else ""
                )
                lines.append(f"{b.name}: {_number(b.point)}{interval}")
        parts.append(
            "<baseline_forecasts>\n" + "\n".join(lines) + "\n</baseline_forecasts>"
        )
    if bundle.documents:
        docs = []
        for d in bundle.documents:
            text = (
                d.content
                if len(d.content) <= DOCUMENT_CHARS
                else d.content[:DOCUMENT_CHARS] + " …"
            )
            title = (d.title or "").replace('"', "'")
            docs.append(
                f'<document type="{d.document_type}" published="{d.published_at:%Y-%m-%d}" '
                f'title="{title}">\n{text}\n</document>'
            )
        parts.append("<documents>\n" + "\n".join(docs) + "\n</documents>")
    parts.append(_task(q))
    return "\n\n".join(parts)


def _question_text(q: LLMQuestion) -> str:
    if q.question_text:
        return q.question_text
    if q.binary:
        series, threshold = q.question_key.split(">=", 1)
        return f"Will {series} for {q.target_date:%Y-%m} be at least {threshold}?"
    return f"What will {q.question_key} be for {q.target_date:%Y-%m}?"


def _task(q: LLMQuestion) -> str:
    if q.binary:
        return (
            "Give the probability that this question resolves yes, as a number "
            "between 0 and 1."
        )
    return (
        "Give a point forecast (the median) for the value the resolution rule "
        "will report, and an 80% interval: lower_80 and upper_80 such that the "
        "value falls below lower_80 one time in ten and above upper_80 one time "
        "in ten. Use the series' units."
    )


# --- cost ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None


def usage_and_cost(response: Any) -> Usage:
    """Token totals over every attempt, and their price (None for an unpriced model)."""
    usage = response.usage
    attempts = [
        (getattr(it, "model", None) or response.model, it)
        for it in (getattr(usage, "iterations", None) or [])
    ] or [(response.model, usage)]
    totals = [0, 0, 0, 0]
    cost: float | None = 0.0
    for model, u in attempts:
        counts = (
            u.input_tokens or 0,
            u.output_tokens or 0,
            u.cache_read_input_tokens or 0,
            u.cache_creation_input_tokens or 0,
        )
        totals = [a + b for a, b in zip(totals, counts)]
        price = PRICES.get(model)
        if price is None or cost is None:
            cost = None
            continue
        inp, out, write, read = price
        cost += (
            counts[0] * inp + counts[1] * out + counts[2] * read + counts[3] * write
        ) / 1e6
    inp_t, out_t, read_t, write_t = totals
    return Usage(
        inp_t, out_t, read_t, write_t, None if cost is None else round(cost, 6)
    )


# --- the forecaster ----------------------------------------------------------------------


def _validate(kind: str, parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise TypeError("the reply is not a JSON object")
    if kind == "binary":
        p = parsed.get("probability")
        if not isinstance(p, int | float) or not 0 <= p <= 1 or math.isnan(p):
            raise ValueError(f"probability {p!r} is not in [0, 1]")
        return {"probability": float(p), "rationale": str(parsed.get("rationale", ""))}
    values = [parsed.get(k) for k in ("lower_80", "point_forecast", "upper_80")]
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
        raise ValueError(f"non-numeric forecast {values!r}")
    low, point, high = (float(v) for v in values)  # type: ignore[arg-type]
    if not low <= point <= high:
        raise ValueError(f"interval [{low}, {high}] doesn't contain the point {point}")
    return {
        "point_forecast": point,
        "lower_80": low,
        "upper_80": high,
        "rationale": str(parsed.get("rationale", "")),
    }


class ClaudeForecaster:
    """See the module docstring."""

    def __init__(
        self,
        model: str = MODEL,
        api_key: str | None = None,
        *,
        conn: Connection | None = None,
        client: anthropic.Anthropic | None = None,
        effort: str = EFFORT,
        max_tokens: int = MAX_TOKENS,
        fallbacks: bool = True,
        reuse: bool = True,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.conn = conn
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.reuse = reuse
        self.client = client or anthropic.Anthropic(
            api_key=api_key or anthropic_api_key(), max_retries=max_retries
        )

    def predict_numeric(
        self,
        question: LLMQuestion,
        context_bundle: ContextBundle,
        horizon_months: int | None = None,
    ) -> dict[str, Any]:
        if question.binary:
            raise ValueError("a binary question; use predict_binary")
        if horizon_months is not None and question.horizon_months not in (
            None,
            horizon_months,
        ):
            raise ValueError(
                f"question {question.id} is {question.horizon_months} months ahead, "
                f"not {horizon_months}"
            )
        if not context_bundle.recent_values:
            # Nothing to forecast from: don't pay for a guess.
            raise ForecastError(
                f"no {question.series_id} values known by {context_bundle.as_of_date}"
            )
        return self._forecast("numeric", question, context_bundle, NUMERIC_SCHEMA)

    def predict_binary(
        self, question: LLMQuestion, context_bundle: ContextBundle
    ) -> dict[str, Any]:
        if not question.binary:
            raise ValueError("a numeric question; use predict_numeric")
        return self._forecast("binary", question, context_bundle, BINARY_SCHEMA)

    def request(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        """The Messages API request body (without `betas`)."""
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": dict(schema)},
            },
        }
        if self.fallbacks:
            body["fallbacks"] = "default"
        return body

    def _forecast(
        self,
        kind: str,
        question: LLMQuestion,
        bundle: ContextBundle,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        prompt = build_prompt(bundle)
        body = self.request(prompt, schema)
        params = {k: v for k, v in body.items() if k not in ("model", "messages")}
        key = context_hash(
            self.model, prompt, {**params, "prompt_version": PROMPT_VERSION}
        )
        base = LLMCall(
            self.model, prompt, context_hash=key, prompt_version=PROMPT_VERSION,
            as_of_date=bundle.as_of_date, question_id=question.id,
        )  # fmt: skip

        if self.conn is not None and self.reuse:
            hit = cached_call(self.conn, self.model, key)
            if hit is not None and hit.parsed_response is not None:
                return {**hit.parsed_response, "call_id": hit.id, "cached": True}

        betas = [FALLBACK_BETA] if self.fallbacks else []
        try:
            response = self.client.beta.messages.create(**body, betas=betas)
        except anthropic.APIError as exc:
            retryable = (
                isinstance(exc, RETRYABLE) or getattr(exc, "status_code", 0) == 529
            )
            call_id = self._log(base, error=f"{type(exc).__name__}: {exc}")
            raise ForecastError(f"Claude API call failed: {exc}", retryable=retryable,
                                call_id=call_id) from exc  # fmt: skip

        usage = usage_and_cost(response)
        text = "".join(b.text for b in response.content if b.type == "text")
        logged = {
            "raw_response": text or None,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "cost_usd": usage.cost_usd,
        }
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            call_id = self._log(base, error=f"refusal ({category})", **logged)
            raise ForecastError(
                f"Claude declined the request ({category})", call_id=call_id
            )
        if response.stop_reason == "max_tokens":
            call_id = self._log(base, error="max_tokens: reply truncated", **logged)
            raise ForecastError("the reply hit max_tokens", call_id=call_id)
        try:
            parsed = _validate(kind, json.loads(text))
        except (ValueError, TypeError) as exc:
            call_id = self._log(base, error=f"malformed reply: {exc}", **logged)
            raise ForecastError(f"malformed reply: {exc}", call_id=call_id) from exc

        if response.model != self.model:
            parsed["served_by"] = response.model
        call_id = self._log(base, parsed_response=parsed, **logged)
        return {
            **parsed,
            "call_id": call_id,
            "cached": False,
            "cost_usd": usage.cost_usd,
        }

    def _log(self, base: LLMCall, **fields: Any) -> int | None:
        if self.conn is None:
            return None
        call = LLMCall(**{**base.__dict__, **fields})
        if call.raw_response is None and call.error is None:
            call = LLMCall(**{**call.__dict__, "error": "empty reply"})
        return log_call(self.conn, call)


def prompt_hash() -> str:
    """Hash of the system prompt, schemas and version: changes when the prompt does."""
    payload = json.dumps(
        [PROMPT_VERSION, SYSTEM_PROMPT, NUMERIC_SCHEMA, BINARY_SCHEMA], sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def register_prompt_version(conn: Connection, forecaster: ClaudeForecaster) -> int:
    """The `model_versions` id for this model, effort and prompt version."""
    return get_or_create_model_version(
        conn,
        "claude_llm",
        f"{forecaster.model}:{PROMPT_VERSION}:{forecaster.effort}",
        parameters={
            "model": forecaster.model,
            "effort": forecaster.effort,
            "prompt_version": PROMPT_VERSION,
            "max_tokens": forecaster.max_tokens,
            "fallbacks": "default" if forecaster.fallbacks else None,
        },
        code_hash=prompt_hash(),
    )
