-- The AI domain, part 1: new enum values and columns (AI task 2).
--
-- Purely additive: every existing row keeps its meaning, and every new column
-- either defaults to the economy's value or is NULL. The constraints that use
-- the new enum values are in 010, because Postgres won't let an enum value be
-- used in the transaction that adds it (each migration is one transaction).

-- Numeric AI targets, monthly like CPI and unemployment.
ALTER TYPE forecast_target ADD VALUE 'arxiv_ai_submissions';
ALTER TYPE forecast_target ADD VALUE 'hf_new_models';
ALTER TYPE forecast_target ADD VALUE 'github_ai_activity';
ALTER TYPE forecast_target ADD VALUE 'benchmark_sota';
ALTER TYPE forecast_target ADD VALUE 'largest_training_run';
-- Every yes/no event question shares this target; `question_key` tells them apart.
ALTER TYPE forecast_target ADD VALUE 'ai_binary_event';

ALTER TYPE question_type ADD VALUE 'binary';

ALTER TABLE series
    ADD COLUMN domain VARCHAR(20) NOT NULL DEFAULT 'economy'
    CHECK (domain IN ('economy', 'ai'));
ALTER TABLE questions
    ADD COLUMN domain VARCHAR(20) NOT NULL DEFAULT 'economy'
    CHECK (domain IN ('economy', 'ai'));

-- Event questions. A binary question is asked `lead_days` before its
-- resolution date (target_date), like the FOMC week-before question, so it
-- reuses that column rather than adding a horizon_days.
--   question_key:  stable id of a binary question, e.g. 'metaculus:12345' or
--                  'swe_bench_verified>=80'; part of the question's unique key.
--   question_text: the question as asked, for curated questions.
ALTER TABLE questions ADD COLUMN question_key TEXT;
ALTER TABLE questions ADD COLUMN question_text TEXT;

-- Most AI sources keep no revision history. Values backfilled from
-- timestamped records (arXiv submission dates, Epoch publication dates)
-- rather than captured live are flagged, so they're never mistaken for
-- genuine vintages.
ALTER TABLE observations ADD COLUMN reconstructed BOOLEAN NOT NULL DEFAULT FALSE;
