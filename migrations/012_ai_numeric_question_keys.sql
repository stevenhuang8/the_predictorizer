-- Numeric AI questions name their series in question_key (AI task 9).
--
-- Three benchmark series (EPOCH_GPQA_SOTA, EPOCH_SWEBENCH_SOTA,
-- EPOCH_ARC2_SOTA) share the target `benchmark_sota`, so a NULL key would
-- let only one of them have a question per month and horizon. Every AI
-- question now has a key: the series id for numeric questions (which also
-- says which series resolves it), the event key for binary ones. Economy
-- questions still have none, so their uniqueness is exactly as before.
--
-- Replaces 010's "key iff binary". Binary questions are all AI, so they still
-- need a key. Adding the constraint validates every existing row, so a keyless
-- numeric AI question fails this migration rather than being let through.

ALTER TABLE questions DROP CONSTRAINT questions_key_iff_binary;
ALTER TABLE questions
    ADD CONSTRAINT questions_key_iff_ai
    CHECK ((question_key IS NOT NULL) = (domain = 'ai'));
