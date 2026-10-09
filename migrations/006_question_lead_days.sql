-- Questions asked a fixed number of days before an event (Task 22 follow-up).
--
-- The monthly run asks about months M+1, M+3 and M+6 (horizon_months). FOMC
-- forecasts are only skilful in the last week before a meeting, so the
-- scheduler also asks about each meeting `lead_days` (7) before it. These are
-- separate questions, so accuracy is scored per lead time instead of mixing
-- week-ahead and months-ahead forecasts. A question has exactly one of
-- horizon_months and lead_days.

ALTER TABLE questions ADD COLUMN lead_days INTEGER CHECK (lead_days >= 0);
ALTER TABLE questions ALTER COLUMN horizon_months DROP NOT NULL;
ALTER TABLE questions
    ADD CONSTRAINT questions_one_horizon
    CHECK (num_nonnulls(horizon_months, lead_days) = 1);

ALTER TABLE questions
    DROP CONSTRAINT questions_target_target_date_horizon_months_key;
ALTER TABLE questions
    ADD CONSTRAINT questions_target_date_horizon_key
    UNIQUE NULLS NOT DISTINCT (target, target_date, horizon_months, lead_days);
