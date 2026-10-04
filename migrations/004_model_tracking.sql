-- Model versions, the features each forecast saw, and the forecasts themselves.
--
-- A forecast links a question to the model version and feature snapshot that
-- produced it. Revising a forecast inserts a new row pointing at the one it
-- replaces. Scores are filled in per forecast once its question resolves.

CREATE TABLE model_versions (
    id             SERIAL PRIMARY KEY,
    model_type     VARCHAR(50) NOT NULL,  -- 'random_walk', 'arima', 'lightgbm', ...
    version_tag    VARCHAR(50) NOT NULL,
    parameters     JSONB CHECK (jsonb_typeof(parameters) = 'object'),
    training_start DATE,
    training_end   DATE,
    code_hash      VARCHAR(64),  -- git commit or file hash
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_type, version_tag),
    CHECK (training_start <= training_end)
);

CREATE TABLE feature_snapshots (
    id            SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,  -- data as known on this date
    features      JSONB NOT NULL CHECK (jsonb_typeof(features) = 'object'),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE forecasts (
    id                   SERIAL PRIMARY KEY,
    question_id          INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    model_version_id     INTEGER NOT NULL REFERENCES model_versions(id),
    feature_snapshot_id  INTEGER REFERENCES feature_snapshots(id),
    forecast_date        DATE NOT NULL,  -- the date the forecast is made as of
    is_backtest          BOOLEAN NOT NULL DEFAULT FALSE,  -- false = made live
    prediction           NUMERIC,  -- point estimate
    interval_lower       NUMERIC,
    interval_upper       NUMERIC,
    probabilities        JSONB,  -- categorical: {"cut": 0.2, "hold": 0.7, "hike": 0.1}
    shap_values          JSONB,  -- feature attributions
    rationale            TEXT,
    previous_forecast_id INTEGER REFERENCES forecasts(id) ON DELETE SET NULL,  -- revisions
    -- Scores, set when the question resolves.
    error                NUMERIC,  -- prediction - actual
    in_interval          BOOLEAN,
    brier_score          NUMERIC,
    scored_at            TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (num_nonnulls(prediction, probabilities) >= 1),
    CHECK ((interval_lower IS NULL) = (interval_upper IS NULL)),
    CHECK (interval_lower <= interval_upper),
    CHECK (jsonb_typeof(probabilities) = 'object'),
    CHECK (jsonb_typeof(shap_values) = 'object'),
    CHECK (previous_forecast_id <> id)
);

CREATE INDEX idx_forecasts_question ON forecasts (question_id);
CREATE INDEX idx_forecasts_model_version ON forecasts (model_version_id);
