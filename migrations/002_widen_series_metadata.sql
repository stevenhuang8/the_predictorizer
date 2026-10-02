-- Series metadata is stored verbatim from the source, and FRED's strings
-- outgrow the original limits (ICSA's frequency is "Weekly, Ending Saturday",
-- 23 chars vs VARCHAR(20)). TEXT costs nothing extra in Postgres.

ALTER TABLE series
    ALTER COLUMN units               TYPE TEXT,
    ALTER COLUMN frequency           TYPE TEXT,
    ALTER COLUMN seasonal_adjustment TYPE TEXT;
