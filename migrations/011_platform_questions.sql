-- Forecasting platforms (Polymarket now, Metaculus later): their questions and
-- point-in-time probabilities (AI task 8).
--
-- These are benchmarks to compare our forecasts against, not targets, so they
-- live apart from `series`/`observations` and `questions`. Purely additive.

CREATE TABLE platform_questions (
    id                  SERIAL PRIMARY KEY,
    platform            VARCHAR(20) NOT NULL
                        CHECK (platform IN ('polymarket', 'metaculus')),
    external_id         TEXT NOT NULL,  -- the platform's own id (Polymarket market id)
    title               TEXT NOT NULL,
    resolution_criteria TEXT,
    url                 TEXT,
    opened_at           TIMESTAMPTZ,
    closes_at           TIMESTAMPTZ,  -- scheduled close
    resolved_at         TIMESTAMPTZ,
    -- NULL until resolved, and for resolutions that are neither (e.g. 50/50).
    resolved_outcome    TEXT CHECK (resolved_outcome IN ('yes', 'no')),
    -- Prices are stored with no gaps through this day (from history or live),
    -- so history is only re-requested when it falls behind.
    prices_through      DATE,
    metadata            JSONB NOT NULL DEFAULT '{}',  -- platform specifics
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (platform, external_id),
    CHECK ((resolved_outcome IS NULL) OR (resolved_at IS NOT NULL))
);

-- The platform's probability of "yes" as known on `as_of`. Like
-- observations.reconstructed: TRUE = backfilled from the platform's own
-- timestamped history, FALSE = captured live on that day.
CREATE TABLE platform_probabilities (
    question_id   INTEGER NOT NULL REFERENCES platform_questions(id) ON DELETE CASCADE,
    as_of         DATE NOT NULL,
    probability   NUMERIC NOT NULL CHECK (probability BETWEEN 0 AND 1),
    reconstructed BOOLEAN NOT NULL,
    PRIMARY KEY (question_id, as_of)
);
