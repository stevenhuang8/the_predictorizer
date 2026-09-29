-- Core schema: data sources, series metadata, and vintage-tracked observations.
--
-- observations stores every revision of a data point. `as_of` is the vintage
-- date (ALFRED realtime_start): the date this value became known. A
-- point-in-time read takes the latest as_of <= the query date.

CREATE TABLE sources (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) UNIQUE NOT NULL,
    url         TEXT,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE series (
    id                  SERIAL PRIMARY KEY,
    source_id           INTEGER NOT NULL REFERENCES sources(id),
    series_id           VARCHAR(100) UNIQUE NOT NULL,  -- source's code, e.g. 'CPIAUCSL'
    name                TEXT,
    units               VARCHAR(50),
    frequency           VARCHAR(20),                   -- monthly, quarterly, etc.
    seasonal_adjustment VARCHAR(50),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE observations (
    id          BIGSERIAL PRIMARY KEY,
    series_id   INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    observed_at DATE NOT NULL,   -- period the value describes
    as_of       DATE NOT NULL,   -- vintage date: when this value was published
    value       NUMERIC,         -- NULL = source reported missing ('.')
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (series_id, observed_at, as_of)
);

-- The UNIQUE constraint's index (series_id, observed_at, as_of) already serves
-- per-series and point-in-time lookups, so no separate (series_id, observed_at)
-- index is needed.
CREATE INDEX idx_obs_as_of ON observations (as_of);
