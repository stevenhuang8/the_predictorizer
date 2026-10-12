-- The AI domain, part 2: constraints using 009's enum values (AI task 2).
--
-- Every economy row already satisfies these, and they reject nothing the old
-- constraints allowed for economy targets. Constraint names come from the
-- catalog, and the drops have no IF EXISTS, so a wrong name fails loudly
-- instead of leaving the old constraint in place.

-- Each target has exactly one question type. (Replaces: FOMC <=> probability.)
ALTER TABLE questions DROP CONSTRAINT questions_check;
ALTER TABLE questions
    ADD CONSTRAINT questions_target_type CHECK (
        question_type = CASE target
            WHEN 'fomc_decision' THEN 'probability'::question_type
            WHEN 'ai_binary_event' THEN 'binary'::question_type
            ELSE 'numeric_interval'::question_type
        END
    );

ALTER TABLE questions
    ADD CONSTRAINT questions_target_domain CHECK (
        domain = CASE
            WHEN target IN ('cpi_yoy', 'unemployment', 'fomc_decision') THEN 'economy'
            ELSE 'ai'
        END
    );

-- Binary questions need a key; nothing else has one.
ALTER TABLE questions
    ADD CONSTRAINT questions_key_iff_binary
    CHECK ((question_key IS NOT NULL) = (question_type = 'binary'));

-- Two binary questions can resolve on the same date. Economy rows have a NULL
-- key, and NULLS NOT DISTINCT keeps their uniqueness exactly as before. The
-- name is kept because db/forecasts.py uses ON CONFLICT ON CONSTRAINT.
ALTER TABLE questions DROP CONSTRAINT questions_target_date_horizon_key;
ALTER TABLE questions
    ADD CONSTRAINT questions_target_date_horizon_key
    UNIQUE NULLS NOT DISTINCT (target, target_date, horizon_months, lead_days, question_key);

-- Binary questions resolve yes/no. Which outcomes fit which question type
-- spans two tables, so db/resolutions.py enforces it (AI task 11).
ALTER TABLE resolutions DROP CONSTRAINT resolutions_actual_outcome_check;
ALTER TABLE resolutions
    ADD CONSTRAINT resolutions_actual_outcome_check
    CHECK (actual_outcome IN ('cut', 'hold', 'hike', 'yes', 'no'));
