"""Write chunks + vectors to pgvector and search them."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from team_knowledge_assistant.config import settings
from team_knowledge_assistant.Ingest.chunking import Chunk


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)  # lets psycopg send/receive numpy arrays as `vector`
    return conn


def insert_chunks(conn: psycopg.Connection, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (source, heading_path, content, embedding_model, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (c.source, c.heading_path, c.text, settings.embedding_model, np.array(e, dtype=np.float32))
                for c, e in zip(chunks, embeddings)
            ],
        )
    conn.commit()
    return len(chunks)


@dataclass
class Hit:
    id: int
    source: str
    heading_path: list[str]
    content: str
    distance: float  # cosine distance: 0 = same direction, 2 = opposite


def vector_search(conn: psycopg.Connection, query_embedding: list[float], k: int = 3) -> list[Hit]:
    rows = conn.execute(
        """
        SELECT id, source, heading_path, content, embedding <=> %s AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s
        """,
        (np.array(query_embedding, dtype=np.float32), k),
    ).fetchall()
    return [Hit(*row) for row in rows]


@dataclass
class KeywordHit:
    id: int
    source: str
    heading_path: list[str]
    content: str
    rank: float  # ts_rank: higher = better match (the opposite of Hit.distance); 0 < rank, no fixed max


# How the query text becomes a tsquery. Both take the raw query as their one parameter.
#   all: websearch_to_tsquery -> every word must appear ('404' & 'found'); understands
#        "quoted phrases", OR and -exclusions. Precise, but full questions often match nothing.
#   any: the query's words (stemmed, stop words dropped by to_tsvector) joined with |, so a chunk
#        matching any word is a candidate and chunks matching more words rank higher.
#        Good for feeding hybrid search, where recall matters and vector search covers meaning.
_TSQUERY = {
    "all": "websearch_to_tsquery('english', %s)",
    "any": "(SELECT string_agg(quote_literal(w), ' | ')::tsquery"
    " FROM unnest(tsvector_to_array(to_tsvector('english', %s))) AS w)",
}


def keyword_search(
    conn: psycopg.Connection, query: str, k: int = 5, match: Literal["all", "any"] = "all"
) -> list[KeywordHit]:
    """Full-text search: chunks containing the words of `query`, best-ranked first.

    Words are stemmed and stop words dropped, so "404 when not found" becomes '404' & 'found'.
    `match` picks whether every word must appear ("all") or any one is enough ("any").
    A query of only stop words matches nothing in either mode.
    """
    # Safe to format in: the fragment comes from the fixed dict above, never from user input.
    # The user's text still goes in as a bound parameter (%s).
    rows = conn.execute(
        f"""
        SELECT id, source, heading_path, content, ts_rank(content_tsv, q) AS rank
        FROM chunks, {_TSQUERY[match]} AS t(q)  -- t(q): name the column, not just the row
        WHERE content_tsv @@ q
        ORDER BY rank DESC, id
        LIMIT %s
        """,
        (query, k),
    ).fetchall()
    return [KeywordHit(*row) for row in rows]


RRF_K = 60  # the constant from the original RRF paper (Cormack et al., 2009); rarely worth tuning


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge several best-first lists of ids into one, best first.

    Each id scores sum(1 / (k + rank)) over the lists it appears in (rank starts at 1).
    Only positions are used, never the raw scores, so a cosine distance and a ts_rank can be
    combined even though they are on unrelated scales. An id near the top of both lists beats
    one at the very top of a single list. Ties keep first-seen order.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, id_ in enumerate(ranked, start=1):
            scores[id_] = scores.get(id_, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@dataclass
class HybridHit:
    id: int
    source: str
    heading_path: list[str]
    content: str
    score: float  # RRF score: higher = better; at most 2 / (RRF_K + 1) ~ 0.033 (rank 1 in both lists)
    vector_rank: int | None  # position in the vector candidates, None if it wasn't there
    keyword_rank: int | None  # position in the keyword candidates, None if it wasn't there


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    query_embedding: list[float],
    k: int = 5,
    candidates: int = 20,
) -> list[HybridHit]:
    """Vector search + any-word keyword search, merged with reciprocal rank fusion.

    Each side returns its top `candidates`; the fused list is cut to `k`. Taking more candidates
    than `k` lets a chunk ranked, say, 8th by both sides still reach the top.
    """
    vector_hits = vector_search(conn, query_embedding, k=candidates)
    keyword_hits = keyword_search(conn, query, k=candidates, match="any")

    vector_ids = [h.id for h in vector_hits]
    keyword_ids = [h.id for h in keyword_hits]
    by_id = {h.id: h for h in [*vector_hits, *keyword_hits]}

    return [
        HybridHit(
            id=id_,
            source=by_id[id_].source,
            heading_path=by_id[id_].heading_path,
            content=by_id[id_].content,
            score=score,
            vector_rank=vector_ids.index(id_) + 1 if id_ in vector_ids else None,
            keyword_rank=keyword_ids.index(id_) + 1 if id_ in keyword_ids else None,
        )
        for id_, score in reciprocal_rank_fusion([vector_ids, keyword_ids])[:k]
    ]
