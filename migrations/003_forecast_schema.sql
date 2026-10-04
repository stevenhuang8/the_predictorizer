-- Forecast questions and their resolutions.
--
-- A question is one (target, target_date, horizon) to forecast; forecasts
-- against it come later (model tracking schema). A resolution records the
-- actual outcome once it is published. Scores belong to individual forecasts,
-- since each question collects forecasts from several models.

CREATE TYPE question_type AS ENUM ('numeric_interval', 'probability');
CREATE TYPE forecast_target AS ENUM ('cpi_yoy', 'unemployment', 'fomc_decision');

CREATE TABLE questions (
    id              SERIAL PRIMARY KEY,
    target          forecast_target NOT NULL,
    question_type   question_type NOT NULL,
    horizon_months  INTEGER NOT NULL CHECK (horizon_months > 0),  -- V1: 1, 3 or 6
    -- Period or event the question is about, keyed like observations.observed_at:
    -- the first of the month for CPI/unemployment, the meeting date for FOMC.
    target_date     DATE NOT NULL,
    resolution_rule TEXT,  -- how the actual value is determined
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (target, target_date, horizon_months),
    -- FOMC decisions are categorical; CPI and unemployment are numeric.
    CHECK ((target = 'fomc_decision') = (question_type = 'probability'))
);

CREATE TABLE resolutions (
    id             SERIAL PRIMARY KEY,
    question_id    INTEGER NOT NULL UNIQUE REFERENCES questions(id) ON DELETE CASCADE,
    actual_value   NUMERIC,  -- numeric questions
    actual_outcome TEXT CHECK (actual_outcome IN ('cut', 'hold', 'hike')),  -- FOMC
    -- Vintage of the data used to resolve, so later revisions can be compared.
    actual_as_of   DATE,
    resolved_at    TIMESTAMPTZ NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (num_nonnulls(actual_value, actual_outcome) = 1)
);
