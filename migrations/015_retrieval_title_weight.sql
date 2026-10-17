-- Rank retrieval matches in a document's title above matches in its body
-- (AI task 17).
--
-- With one unweighted vector, long texts that repeat the query's words
-- outranked documents whose title is the match (AI_NOTES/Task16.md). The
-- title is now weight A and the content weight B; `ai/llm/retrieval.py`
-- also normalizes the rank by document length. Rebuilds the generated column
-- and its index; no data changes.

DROP INDEX idx_retrieval_search;
ALTER TABLE retrieval_documents DROP COLUMN search;
ALTER TABLE retrieval_documents ADD COLUMN search TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title, '')), 'A')
    || setweight(to_tsvector('english', content), 'B')
) STORED;
CREATE INDEX idx_retrieval_search ON retrieval_documents USING GIN (search);
