-- Why each scored forecast missed (or didn't), Task 20.
--
-- One row per scored forecast. The classifier (postmortem/classifier.py)
-- writes and refreshes 'auto' rows; a person can override any row with a
-- 'manual' one, which the classifier never touches again. `evidence` holds
-- the numbers behind the call so it can be checked.

CREATE TYPE miss_category AS ENUM (
    'bad_data',           -- the target was revised; against the revision it was fine
    'bad_model',          -- a normal-sized move (or a priced decision) the model missed
    'regime_change',      -- a move beyond historical norms (or an unpriced decision)
    'expected_variance'   -- within what the forecast's own uncertainty allowed
);

CREATE TABLE postmortems (
    id                  SERIAL PRIMARY KEY,
    forecast_id         INTEGER NOT NULL UNIQUE REFERENCES forecasts(id) ON DELETE CASCADE,
    miss_category       miss_category NOT NULL,
    notes               TEXT,
    -- |error vs first release| - |error vs latest value|: how much of the error
    -- later revisions took away (negative if they made it worse). Numeric only.
    revised_data_impact NUMERIC,
    evidence            JSONB CHECK (jsonb_typeof(evidence) = 'object'),
    classified_by       TEXT NOT NULL DEFAULT 'auto'
                        CHECK (classified_by IN ('auto', 'manual')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_postmortems_category ON postmortems (miss_category);
