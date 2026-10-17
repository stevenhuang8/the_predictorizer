"""Tests for the Claude forecaster and its context bundles (AI task 18).

The Anthropic API is never called: a fake client returns canned responses
or raises the SDK's own errors. Bundle and logging tests use a throwaway
database (skipped if Postgres isn't reachable).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.ingest_arxiv import SERIES as ARXIV_SERIES
from eco_prediction.ai.llm.claude_forecaster import (
    BINARY_SCHEMA,
    FALLBACK_BETA,
    MODEL,
    NUMERIC_SCHEMA,
    SYSTEM_PROMPT,
    ClaudeForecaster,
    ForecastError,
    anthropic_api_key,
    build_prompt,
    register_prompt_version,
    usage_and_cost,
)
from eco_prediction.ai.llm.context import (
    Baseline,
    ContextBundle,
    LLMQuestion,
    build_context_bundle,
    load_question,
)
from eco_prediction.ai.llm.retrieval import Document, upsert_documents
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.ai.store import (
    AIObservation,
    insert_ai_observations,
    upsert_ai_series,
)
from eco_prediction.data.ingest import ensure_source
from eco_prediction.db.forecasts import get_or_create_question
from eco_prediction.db.migrate import migrate

ARXIV = "ARXIV_AI_SUBMISSIONS"
AS_OF = date(2026, 10, 14)


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=UTC)


# --- fakes ---------------------------------------------------------------------------


def usage(
    inp: int = 1000,
    out: int = 200,
    read: int = 0,
    write: int = 0,
    iterations: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=inp, output_tokens=out, cache_read_input_tokens=read,
        cache_creation_input_tokens=write, iterations=iterations,
    )  # fmt: skip


def response(
    text: str, stop: str = "end_turn", model: str = MODEL, **kwargs: Any
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""),
                 SimpleNamespace(type="text", text=text)],
        stop_reason=stop, model=model, usage=kwargs.pop("usage", usage()),
        stop_details=kwargs.pop("stop_details", None),
    )  # fmt: skip


class FakeClient:
    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(cls: type, status: int) -> Exception:
    return cls("boom", response=httpx2.Response(status, request=REQUEST), body=None)  # type: ignore[no-any-return]


NUMERIC_Q = LLMQuestion(
    82, "numeric_interval", ARXIV, date(2026, 11, 1), horizon_months=1,
    resolution_rule="arXiv submissions in cs.AI, cs.LG, cs.CL by month of first submission.",
)  # fmt: skip
THRESHOLD_Q = LLMQuestion(
    90, "binary", f"{ARXIV}>=33000", date(2027, 1, 1), horizon_months=3,
    created_at=utc(2026, 10, 15),
)  # fmt: skip
BUNDLE = ContextBundle(
    NUMERIC_Q, AS_OF, ARXIV, "arXiv submissions", "Papers",
    recent_values=[(date(2026, 8, 1), 29000.0), (date(2026, 9, 1), 30120.0)],
    baselines=[Baseline("random walk", 30120.0, 28000.0, 32500.0)],
    documents=[Document("arxiv_abstract", "2610.1", "A new benchmark.", utc(2026, 10, 3),
                        title='The "X" benchmark')],
)  # fmt: skip
GOOD = json.dumps({"rationale": "trend", "point_forecast": 31000, "lower_80": 29500,
                   "upper_80": 32800})  # fmt: skip


def forecaster(
    client: FakeClient, conn: Connection | None = None, **kw: Any
) -> ClaudeForecaster:
    return ClaudeForecaster(conn=conn, client=client, **kw)  # type: ignore[arg-type]


# --- the prompt and request ----------------------------------------------------------------


def test_the_prompt_has_every_part_of_the_bundle() -> None:
    prompt = build_prompt(BUNDLE)
    assert prompt.startswith(
        "Today is 2026-10-15. You know what was published through 2026-10-14."
    )
    assert "What will ARXIV_AI_SUBMISSIONS be for 2026-11?" in prompt
    assert "<resolution_rule>\narXiv submissions in cs.AI" in prompt
    assert (
        '<recent_values series="ARXIV_AI_SUBMISSIONS" name="arXiv submissions" units="Papers">'
        in prompt
    )
    assert "2026-08: 29,000\n2026-09: 30,120" in prompt
    assert "random walk: 30,120, 80% interval [28,000, 32,500]" in prompt
    assert (
        '<document type="arxiv_abstract" published="2026-10-03" title="The \'X\' benchmark">'
        in prompt
    )
    assert prompt.rstrip().endswith("Use the series' units.")
    assert build_prompt(BUNDLE) == prompt  # deterministic


def test_binary_prompts_ask_for_a_probability() -> None:
    bundle = ContextBundle(
        THRESHOLD_Q, AS_OF, ARXIV, "arXiv", "Papers",
        baselines=[Baseline("historical base rate", probability=0.25)],
    )  # fmt: skip
    prompt = build_prompt(bundle)
    assert "Will ARXIV_AI_SUBMISSIONS for 2027-01 be at least 33000?" in prompt
    assert "historical base rate: P(yes) = 0.250" in prompt
    assert "probability that this question resolves yes" in prompt


def test_the_request_uses_structured_output_thinking_and_fallbacks() -> None:
    client = FakeClient(response(GOOD))
    result = forecaster(client).predict_numeric(NUMERIC_Q, BUNDLE, 1)
    assert result == {
        "point_forecast": 31000.0, "lower_80": 29500.0, "upper_80": 32800.0,
        "rationale": "trend", "call_id": None, "cached": False, "cost_usd": 0.008,
    }  # fmt: skip
    (req,) = client.requests
    assert req["model"] == "claude-opus-5-5" and req["system"] == SYSTEM_PROMPT
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {
        "effort": "high", "format": {"type": "json_schema", "schema": NUMERIC_SCHEMA},
    }  # fmt: skip
    assert req["fallbacks"] == "default" and req["betas"] == [FALLBACK_BETA]
    assert req["messages"] == [{"role": "user", "content": build_prompt(BUNDLE)}]
    assert next(iter(NUMERIC_SCHEMA["properties"])) == "rationale"  # reasons first

    client = FakeClient(response('{"rationale": "r", "probability": 0.3}'))
    f = forecaster(client, fallbacks=False)
    bundle = ContextBundle(THRESHOLD_Q, AS_OF, ARXIV, None, None)
    assert f.predict_binary(THRESHOLD_Q, bundle)["probability"] == 0.3
    assert "fallbacks" not in client.requests[0] and client.requests[0]["betas"] == []
    assert client.requests[0]["output_config"]["format"]["schema"] == BINARY_SCHEMA


def test_wrong_question_types_and_horizons_are_refused() -> None:
    f = forecaster(FakeClient())
    with pytest.raises(ValueError, match="predict_binary"):
        f.predict_numeric(THRESHOLD_Q, BUNDLE)
    with pytest.raises(ValueError, match="predict_numeric"):
        f.predict_binary(NUMERIC_Q, BUNDLE)
    with pytest.raises(ValueError, match="1 months ahead, not 3"):
        f.predict_numeric(NUMERIC_Q, BUNDLE, 3)
    client = FakeClient()
    with pytest.raises(ForecastError, match="no ARXIV_AI_SUBMISSIONS values known"):
        forecaster(client).predict_numeric(
            NUMERIC_Q, ContextBundle(NUMERIC_Q, AS_OF, ARXIV, None, None)
        )
    assert client.requests == []  # refused before paying for a call


# --- bad replies and errors ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "malformed reply"),
        ("[1, 2]", "not a JSON object"),
        ('{"rationale": "r", "point_forecast": 5, "lower_80": 6, "upper_80": 9}',
         "doesn't contain the point"),
        ('{"rationale": "r", "point_forecast": "x", "lower_80": 1, "upper_80": 9}',
         "non-numeric"),
    ],
)  # fmt: skip
def test_malformed_replies_raise(text: str, message: str) -> None:
    with pytest.raises(ForecastError, match=message) as err:
        forecaster(FakeClient(response(text))).predict_numeric(NUMERIC_Q, BUNDLE)
    assert not err.value.retryable


def test_probabilities_must_be_in_range() -> None:
    bundle = ContextBundle(THRESHOLD_Q, AS_OF, ARXIV, None, None)
    client = FakeClient(response('{"rationale": "r", "probability": 1.2}'))
    with pytest.raises(ForecastError, match=r"not in \[0, 1\]"):
        forecaster(client).predict_binary(THRESHOLD_Q, bundle)


def test_refusals_truncation_and_api_errors_raise() -> None:
    refused = response(
        "", stop="refusal", stop_details=SimpleNamespace(category="cyber")
    )
    with pytest.raises(ForecastError, match=r"declined the request \(cyber\)"):
        forecaster(FakeClient(refused)).predict_numeric(NUMERIC_Q, BUNDLE)
    with pytest.raises(ForecastError, match="max_tokens"):
        forecaster(FakeClient(response('{"rati', stop="max_tokens"))).predict_numeric(
            NUMERIC_Q, BUNDLE
        )
    with pytest.raises(ForecastError) as err:
        forecaster(
            FakeClient(api_error(anthropic.RateLimitError, 429))
        ).predict_numeric(NUMERIC_Q, BUNDLE)
    assert err.value.retryable
    with pytest.raises(ForecastError) as err:
        forecaster(
            FakeClient(anthropic.APIConnectionError(request=REQUEST))
        ).predict_numeric(NUMERIC_Q, BUNDLE)
    assert err.value.retryable
    with pytest.raises(ForecastError) as err:
        forecaster(
            FakeClient(api_error(anthropic.BadRequestError, 400))
        ).predict_numeric(NUMERIC_Q, BUNDLE)
    assert not err.value.retryable


# --- cost --------------------------------------------------------------------------------------


def test_cost_is_priced_per_attempt() -> None:
    # Opus 5.5: $4 in, $20 out, $5 cache write, $0.20 cache read per million.
    one = usage_and_cost(
        response("", usage=usage(10_000, 2_000, read=50_000, write=4_000))
    )
    assert one.cost_usd == pytest.approx(
        (10_000 * 4 + 2_000 * 20 + 50_000 * 0.2 + 4_000 * 5) / 1e6
    )
    assert (one.input_tokens, one.cache_read_tokens, one.cache_write_tokens) == (
        10_000,
        50_000,
        4_000,
    )
    # A refusal on Opus 5.5 rescued by Opus 4.8 ($5 / $25): each at its own rates.
    attempts = [
        SimpleNamespace(type="message", model="claude-opus-5-5", input_tokens=1000,
                        output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        SimpleNamespace(type="fallback_message", model="claude-opus-4-8", input_tokens=1000,
                        output_tokens=400, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    ]  # fmt: skip
    both = usage_and_cost(
        response("", model="claude-opus-4-8", usage=usage(iterations=attempts))
    )
    assert both.cost_usd == pytest.approx(
        (1000 * 4 + 50 * 20 + 1000 * 5 + 400 * 25) / 1e6
    )
    assert both.output_tokens == 450
    unknown = usage_and_cost(response("", model="claude-future-9", usage=usage()))
    assert unknown.cost_usd is None and unknown.input_tokens == 1000


# --- logging, caching and bundles (database) ----------------------------------------------------


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def calls(conn: Connection) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT question_id, prompt_version, as_of_date, error, raw_response IS NOT NULL,"
            " parsed_response, input_tokens, cost_usd::float8 FROM llm_calls ORDER BY id"
        )
        return cur.fetchall()


def test_calls_are_logged_and_reused(migrated: Connection) -> None:
    question, _ = get_or_create_question(
        migrated, "arxiv_ai_submissions", date(2026, 11, 1), 1, question_key=ARXIV
    )
    q = LLMQuestion(question, "numeric_interval", ARXIV, date(2026, 11, 1), 1)
    values = [(date(2026, 9, 1), 30120.0)]
    bundle = ContextBundle(q, AS_OF, ARXIV, None, None, values)
    client = FakeClient(response("oops"), response(GOOD))
    f = forecaster(client, migrated)
    with pytest.raises(ForecastError) as err:
        f.predict_numeric(q, bundle)
    first = f.predict_numeric(q, bundle)  # the failure isn't reused
    again = f.predict_numeric(q, bundle)  # the success is: no third request
    assert len(client.requests) == 2
    assert again["cached"] and again["call_id"] == first["call_id"] != err.value.call_id
    assert again["point_forecast"] == 31000.0
    rows = calls(migrated)
    assert rows[0][:5] == (
        question,
        "v1",
        AS_OF,
        "malformed reply: Expecting value: line 1 column 1 (char 0)",
        True,
    )
    assert rows[1][3] is None and rows[1][5]["upper_80"] == 32800 and rows[1][6] == 1000
    assert rows[1][7] == pytest.approx(0.008)
    # A new call is forced with reuse off, or by any change to the prompt.
    fresh = forecaster(FakeClient(response(GOOD)), migrated, reuse=False)
    assert not fresh.predict_numeric(q, bundle)["cached"]
    other = ContextBundle(q, date(2026, 10, 15), ARXIV, None, None, values)
    forecaster(FakeClient(response(GOOD)), migrated).predict_numeric(q, other)
    assert len(calls(migrated)) == 4

    # API errors are logged without a response.
    with pytest.raises(ForecastError):
        forecaster(
            FakeClient(api_error(anthropic.RateLimitError, 429)), migrated
        ).predict_numeric(
            q, ContextBundle(q, date(2026, 10, 16), ARXIV, None, None, values)
        )
    assert (
        calls(migrated)[-1][3].startswith("RateLimitError")
        and not calls(migrated)[-1][4]
    )


def test_prompt_versions_are_registered_once(migrated: Connection) -> None:
    f = forecaster(FakeClient())
    first = register_prompt_version(migrated, f)
    assert register_prompt_version(migrated, f) == first
    assert (
        register_prompt_version(migrated, forecaster(FakeClient(), effort="max"))
        != first
    )


def seed(conn: Connection) -> None:
    source = ensure_source(conn, *ARXIV_SERIES.source)
    series = upsert_ai_series(conn, source, ARXIV_SERIES.info)
    months = list(iter_months(date(2023, 1, 1), date(2026, 9, 1)))
    insert_ai_observations(
        conn, series,
        [AIObservation(m, next_month(m), 20_000 * 1.01**i, True) for i, m in enumerate(months)],
    )  # fmt: skip
    upsert_documents(conn, [
        Document("arxiv_abstract", f"a{i}", "Language models research growth.", utc(2026, 9, i + 1),
                 title=f"Paper {i}")
        for i in range(6)
    ] + [
        Document("arxiv_abstract", "late", "Language models research.", utc(2026, 10, 20)),
        Document("arxiv_abstract", "old", "Language models research.", utc(2024, 1, 1)),
        Document("github_release", "r1", "Machine learning framework release.", utc(2026, 9, 1)),
    ])  # fmt: skip


def test_numeric_bundles_are_point_in_time(migrated: Connection) -> None:
    seed(migrated)
    qid, _ = get_or_create_question(
        migrated, "arxiv_ai_submissions", date(2026, 11, 1), 1, question_key=ARXIV,
        resolution_rule="arXiv submissions by month.",
    )  # fmt: skip
    q = load_question(migrated, qid)
    assert (q.series_id, q.binary, q.resolution_rule) == (
        ARXIV,
        False,
        "arXiv submissions by month.",
    )
    bundle = build_context_bundle(migrated, q, date(2026, 8, 14))
    # Known through 2026-07 on 2026-08-14 (each month lands on the 1st of the next).
    assert len(bundle.recent_values) == 12 and bundle.recent_values[-1][0] == date(
        2026, 7, 1
    )
    assert bundle.units == "Papers"
    assert [b.name for b in bundle.baselines] == [
        "random walk",
        "linear trend (12 months)",
        "ARIMA",
    ]
    assert bundle.baselines[0].point == pytest.approx(20_000 * 1.01**42)
    # Only documents published by 08-14, and none older than a year.
    assert bundle.documents == []
    later = build_context_bundle(migrated, q, date(2026, 10, 14))
    types = [d.document_type for d in later.documents]
    assert types.count("arxiv_abstract") == 4 and types.count("github_release") == 1
    assert {d.source_id for d in later.documents}.isdisjoint({"late", "old"})
    with pytest.raises(ValueError, match="no question"):
        load_question(migrated, 9999)


def test_curated_bundles_carry_the_market_price(migrated: Connection) -> None:
    market = PlatformQuestion(
        "polymarket", "m1", "Will GPT-6 be released by December 31?",
        opened_at=utc(2026, 6, 1), closes_at=utc(2026, 12, 31),
    )  # fmt: skip
    pq, _, _ = upsert_platform_question(migrated, market)
    with migrated.cursor() as cur:
        cur.execute(
            "INSERT INTO platform_probabilities VALUES (%s, '2026-10-13', 0.42, false),"
            " (%s, '2026-10-15', 0.9, false)",
            (pq, pq),
        )
    qid, _ = get_or_create_question(
        migrated, "ai_binary_event", date(2026, 12, 31), lead_days=77,
        question_key="polymarket:m1", question_text=market.title, platform_question_id=pq,
    )  # fmt: skip
    q = load_question(migrated, qid)
    assert q.curated and q.series_id is None and q.ask_date == date(2026, 10, 15)
    bundle = build_context_bundle(migrated, q, date(2026, 10, 14))
    assert bundle.recent_values == []
    price = next(b for b in bundle.baselines if b.name.startswith("market price"))
    assert price.probability == pytest.approx(0.42)  # not the later 0.9


def test_the_real_sdk_sends_and_parses_the_request() -> None:
    """The actual SDK over a mock transport: the wire request and a real BetaMessage."""
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{"type": "thinking", "thinking": "", "signature": "sig"},
                        {"type": "text", "text": GOOD}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 200,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        })  # fmt: skip

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(
            transport=httpx2.MockTransport(handler)
        ),
    )
    result = ClaudeForecaster(client=client).predict_numeric(NUMERIC_Q, BUNDLE)
    assert result["point_forecast"] == 31000.0 and result["cost_usd"] == pytest.approx(
        0.008
    )
    (request,) = sent
    assert request.url.path == "/v1/messages"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["fallbacks"] == "default" and body["model"] == MODEL
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "betas" not in body


def test_the_key_comes_from_env_local_then_env(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert anthropic_api_key() is None
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-env\n")
    assert anthropic_api_key() == "from-env"
    (tmp_path / ".env.local").write_text("# ANTHROPIC_API_KEY=commented\n")
    assert anthropic_api_key() == "from-env"  # a commented line is unset
    (tmp_path / ".env.local").write_text("ANTHROPIC_API_KEY=from-local\n")
    assert anthropic_api_key() == "from-local"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    assert anthropic_api_key() == "from-shell"
