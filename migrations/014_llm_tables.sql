-- LLM forecaster tables (AI task 16): a log of every model call, and a
-- corpus of dated documents to retrieve point-in-time context from.
-- Purely additive.

-- One row per API call, successful or not. A call is made before its
-- forecast is stored, so forecast_id is filled in afterwards (and stays NULL
-- for failed calls and for backtests that store no forecast); question_id
-- links it either way. context_hash identifies the exact request (model,
-- prompt, parameters), so a backtest re-run can reuse a logged response
-- instead of paying for it again.
CREATE TABLE llm_calls (
    id                 BIGSERIAL PRIMARY KEY,
    forecast_id        INTEGER REFERENCES forecasts(id) ON DELETE CASCADE,
    question_id        INTEGER REFERENCES questions(id) ON DELETE CASCADE,
    model_id           VARCHAR(100) NOT NULL,  -- e.g. 'claude-opus-5-5'
    prompt_version     VARCHAR(50),            -- the prompt template's version
    as_of_date         DATE,                   -- the data cutoff the prompt was built from
    prompt             TEXT NOT NULL,
    context_hash       VARCHAR(64) CHECK (context_hash ~ '^[0-9a-f]{64}$'),
    raw_response       TEXT,                   -- NULL if the call failed
    parsed_response    JSONB,                  -- the structured output, if it parsed
    error              TEXT,                   -- why the call or its parsing failed
    input_tokens       INTEGER CHECK (input_tokens >= 0),
    output_tokens      INTEGER CHECK (output_tokens >= 0),
    cache_read_tokens  INTEGER CHECK (cache_read_tokens >= 0),
    cache_write_tokens INTEGER CHECK (cache_write_tokens >= 0),
    cost_usd           NUMERIC(10, 6) CHECK (cost_usd >= 0),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (raw_response IS NOT NULL OR error IS NOT NULL)
);

CREATE INDEX idx_llm_calls_forecast ON llm_calls (forecast_id);
CREATE INDEX idx_llm_calls_question ON llm_calls (question_id);
CREATE INDEX idx_llm_calls_model ON llm_calls (model_id);
CREATE INDEX idx_llm_calls_context ON llm_calls (model_id, context_hash);

-- Text the forecaster may read, each dated by when it became public.
-- Retrieval only returns documents published by the forecast's cutoff.
-- `search` is maintained by Postgres for full-text queries.
CREATE TABLE retrieval_documents (
    id            BIGSERIAL PRIMARY KEY,
    document_type VARCHAR(50) NOT NULL,   -- 'polymarket_question', 'arxiv_abstract', ...
    source_id     VARCHAR(200) NOT NULL,  -- the source's own id (arXiv id, market id, ...)
    title         TEXT,
    content       TEXT NOT NULL,
    published_at  TIMESTAMPTZ NOT NULL,   -- for point-in-time filtering
    metadata      JSONB NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(metadata) = 'object'),
    search        TSVECTOR GENERATED ALWAYS AS (
                      to_tsvector('english', coalesce(title, '') || ' ' || content)
                  ) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_type, source_id)
);

CREATE INDEX idx_retrieval_published ON retrieval_documents (published_at);
CREATE INDEX idx_retrieval_type ON retrieval_documents (document_type, published_at);
CREATE INDEX idx_retrieval_search ON retrieval_documents USING GIN (search);
