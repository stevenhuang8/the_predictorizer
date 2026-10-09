-- The latest vintage of each numeric question's answer, as a diagnostic.
--
-- `actual_value` is the first release and stays the official resolution:
-- scores never change after a question resolves. `latest_value` is refreshed
-- by every resolution run as revisions arrive, so score summaries can also
-- report error against the revised data (`*_latest` columns). It shows how
-- much revisions matter per target, and whether a model tracks the revised
-- value better than the first print. NULL for FOMC questions (decisions
-- aren't revised).

ALTER TABLE resolutions ADD COLUMN latest_value NUMERIC;
ALTER TABLE resolutions ADD COLUMN latest_as_of DATE;
