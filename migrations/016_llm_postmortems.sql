-- LLM-drafted post-mortem notes (AI task 22).
--
-- A draft supplements a post-mortem's category; it never replaces it. The
-- category (and `notes`, the classifier's one-line reason) still come from the
-- rule-based classifier; `draft_note` is the LLM's 2-3 sentence explanation of
-- the miss, kept apart until a person confirms it. Economy post-mortems get the
-- defaults (no draft) and are unaffected. Purely additive.

ALTER TABLE postmortems
    ADD COLUMN llm_drafted     BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN human_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN draft_note      TEXT,
    ADD COLUMN draft_call_id   BIGINT REFERENCES llm_calls(id) ON DELETE SET NULL,
    ADD COLUMN confirmed_at    TIMESTAMPTZ,
    ADD CONSTRAINT postmortems_draft_has_note
        CHECK (NOT llm_drafted OR draft_note IS NOT NULL),
    ADD CONSTRAINT postmortems_confirmed_was_drafted
        CHECK (NOT human_confirmed OR (llm_drafted AND confirmed_at IS NOT NULL));

-- The review queue: drafts nobody has confirmed yet.
CREATE INDEX idx_postmortems_unreviewed
    ON postmortems (forecast_id) WHERE llm_drafted AND NOT human_confirmed;
