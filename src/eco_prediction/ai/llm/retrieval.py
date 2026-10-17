"""The retrieval corpus: dated documents the LLM forecaster may read (AI task 16).

    from eco_prediction.ai.llm.retrieval import (
        Document, index_platform_questions, retrieve_documents, upsert_documents,
    )

    index_platform_questions(conn)       # Polymarket market texts -> the corpus
    conn.commit()

    # A forecast made on 2026-10-15 reads what was public by the end of the 14th.
    docs = retrieve_documents(conn, date(2026, 10, 14), "polymarket_question",
                              query="SWE-bench score")

Documents live in `retrieval_documents` (migration 014), unique per
(document_type, source_id). `published_at` is when the text became public,
and retrieval only returns documents published by the end of `as_of_date`
(UTC), the harness's cutoff convention: pass the day before the forecast.

With a `query`, documents are matched by Postgres full-text search with
English stemming and ranked by `ts_rank`, newest first among ties. Title
words weigh more than body words (migration 015), and the rank is divided by
1 + log(document length), so a long text repeating the query's words doesn't
bury a document whose title is the match. (`ts_rank_cd` scores every
matching stretch of text, so repetition piles up and length normalization
can't undo it.)
- `match="any"` (the default) matches documents with any of the query's
  words, so the ones sharing the most rank first. That suits a query made
  from a question's text: "best AI model end of year" still finds "...the
  best AI model on September 30?", which has no "year".
- `match="all"` uses web-search syntax (`websearch_to_tsquery`): every word
  must appear, with quoted phrases, `or` and `-word`.
A query with no searchable words (only stop words) matches everything.
Without a query the newest documents come first.

Indexers turn a source into documents. This task adds the one source whose
text is already stored, Polymarket's markets (`index_platform_questions`);
arXiv abstracts, model cards and release notes need new fetches and come
with AI task 17. A market's document is its title and resolution criteria,
dated when it opened. Its resolution and prices are deliberately left out:
they're outcomes and signals with their own dates, not background text.

Upserts change a row only when its text, date or metadata differ, so a
monthly re-index reports what actually changed. Nothing here commits.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal, NamedTuple

from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json, execute_values

from eco_prediction.ai.curated_questions import classify

POLYMARKET_QUESTION = "polymarket_question"

__all__ = [
    "POLYMARKET_QUESTION",
    "Document",
    "UpsertResult",
    "end_of_day",
    "index_platform_questions",
    "retrieve_documents",
    "upsert_documents",
]


@dataclass(frozen=True)
class Document:
    document_type: str
    source_id: str
    content: str
    published_at: datetime  # timezone-aware
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class UpsertResult(NamedTuple):
    inserted: int
    updated: int
    unchanged: int


def end_of_day(d: date) -> datetime:
    """The first instant after `d` ends, in UTC: the exclusive cutoff."""
    return datetime.combine(d + timedelta(days=1), time(), tzinfo=UTC)


def upsert_documents(conn: Connection, documents: Iterable[Document]) -> UpsertResult:
    """Insert new documents and update changed ones (the last copy of a key wins)."""
    unique = {(d.document_type, d.source_id): d for d in documents}
    if not unique:
        return UpsertResult(0, 0, 0)
    for d in unique.values():
        if d.published_at.tzinfo is None:
            raise ValueError(f"{d.document_type}:{d.source_id}: published_at is naive")
    rows = [
        (d.document_type, d.source_id, d.title, d.content, d.published_at,
         Json(d.metadata))
        for d in unique.values()
    ]  # fmt: skip
    with conn.cursor() as cur:
        changed = execute_values(
            cur,
            """
            INSERT INTO retrieval_documents
                (document_type, source_id, title, content, published_at, metadata)
            VALUES %s
            ON CONFLICT (document_type, source_id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                published_at = EXCLUDED.published_at,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            WHERE (retrieval_documents.title, retrieval_documents.content,
                   retrieval_documents.published_at, retrieval_documents.metadata)
                  IS DISTINCT FROM
                  (EXCLUDED.title, EXCLUDED.content, EXCLUDED.published_at,
                   EXCLUDED.metadata)
            RETURNING (xmax = 0)
            """,
            rows,
            fetch=True,
        )
    inserted = sum(1 for (new,) in changed if new)
    updated = len(changed) - inserted
    return UpsertResult(inserted, updated, len(unique) - len(changed))


_TSQUERY = {
    "all": "websearch_to_tsquery('english', %(query)s)",
    # The query's own lexemes, already stemmed, ORed ('simple' keeps them as is).
    "any": """to_tsquery('simple', array_to_string(ARRAY(
        SELECT quote_literal(lexeme)
        FROM unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS lexeme
    ), ' | '))""",
}


def retrieve_documents(
    conn: Connection,
    as_of_date: date,
    query_type: str | Sequence[str] | None = None,
    limit: int = 10,
    *,
    query: str | None = None,
    match: Literal["any", "all"] = "any",
    since: date | None = None,
) -> list[Document]:
    """Documents published by the end of `as_of_date` (UTC), best matches first.

    `query_type` limits the document types (one or several); `since` drops
    documents published before that day.
    """
    if match not in _TSQUERY:
        raise ValueError("match must be 'any' or 'all'")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    types = [query_type] if isinstance(query_type, str) else query_type
    where = ["published_at < %(cutoff)s"]
    params: dict[str, Any] = {"cutoff": end_of_day(as_of_date), "limit": limit}
    if types is not None:
        where.append("document_type = ANY(%(types)s)")
        params["types"] = list(types)
    if since is not None:
        where.append("published_at >= %(since)s")
        params["since"] = datetime.combine(since, time(), tzinfo=UTC)
    if query:
        params["query"] = query
        where.append("(numnode(q) = 0 OR search @@ q)")
        rank = "ts_rank(search, q, 1) DESC,"  # title weighted, / (1 + log length)
        source = f"retrieval_documents, LATERAL {_TSQUERY[match]} AS q"
    else:
        rank, source = "", "retrieval_documents"
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT document_type, source_id, content, published_at, title, metadata
            FROM {source}
            WHERE {" AND ".join(where)}
            ORDER BY {rank} published_at DESC, id DESC
            LIMIT %(limit)s
            """,
            params,
        )
        return [Document(*row) for row in cur.fetchall()]


# --- indexers --------------------------------------------------------------------


def index_platform_questions(
    conn: Connection, platform: str = "polymarket"
) -> UpsertResult:
    """Index every market of `platform` as a document (title + criteria).

    Dated by `opened_at`, or when we first saw the market if it has none.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT external_id, title, resolution_criteria, url,
                   COALESCE(opened_at, first_seen_at), closes_at
            FROM platform_questions WHERE platform = %s ORDER BY id
            """,
            (platform,),
        )
        rows = cur.fetchall()
    documents = []
    for external_id, title, criteria, url, published, closes in rows:
        content = title if not criteria else f"{title}\n\n{criteria}"
        documents.append(
            Document(
                f"{platform}_question",
                external_id,
                content,
                published,
                title=title,
                metadata={
                    "platform": platform,
                    "url": url,
                    "closes_at": closes.isoformat() if closes else None,
                    "category": classify(title),
                },
            )
        )
    return upsert_documents(conn, documents)
