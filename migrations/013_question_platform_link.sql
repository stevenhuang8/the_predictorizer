-- Curated binary questions link to the platform market they mirror (AI task 10).
--
-- A curated question is asked several times before its market closes (one
-- row per lead), so the link isn't unique. Its question_key is also
-- 'polymarket:<external_id>', which keeps it unique per lead; this column
-- makes the link a real foreign key, so benchmark prices join without
-- parsing keys. Only binary questions can have one. Purely additive.

ALTER TABLE questions
    ADD COLUMN platform_question_id INTEGER REFERENCES platform_questions(id);
ALTER TABLE questions
    ADD CONSTRAINT questions_platform_link_binary
    CHECK (platform_question_id IS NULL OR question_type = 'binary');
CREATE INDEX questions_platform_question_id_idx
    ON questions (platform_question_id) WHERE platform_question_id IS NOT NULL;
