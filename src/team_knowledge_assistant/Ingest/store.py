"""Write chunks + vectors to pgvector and search them."""

from dataclasses import dataclass

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
    source: str
    heading_path: list[str]
    content: str
    distance: float  # cosine distance: 0 = same direction, 2 = opposite


def vector_search(conn: psycopg.Connection, query_embedding: list[float], k: int = 3) -> list[Hit]:
    rows = conn.execute(
        """
        SELECT source, heading_path, content, embedding <=> %s AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s
        """,
        (np.array(query_embedding, dtype=np.float32), k),
    ).fetchall()
    return [Hit(*row) for row in rows]


@dataclass
class KeywordHit:
    source: str
    heading_path: list[str]
    content: str
    rank: float  # ts_rank: higher = better match (the opposite of Hit.distance); 0 < rank, no fixed max


def keyword_search(conn: psycopg.Connection, query: str, k: int = 5) -> list[KeywordHit]:
    """Full-text search: chunks containing every word of `query`, best-ranked first.

    websearch_to_tsquery stems the words and drops stop words, so "404 when not found"
    becomes '404' & 'found'. It also understands "quoted phrases", OR and -exclusions.
    Every remaining word must appear, so long natural-language questions often match nothing.
    """
    rows = conn.execute(
        """
        SELECT source, heading_path, content, ts_rank(content_tsv, q) AS rank
        FROM chunks, websearch_to_tsquery('english', %s) AS q
        WHERE content_tsv @@ q
        ORDER BY rank DESC
        LIMIT %s
        """,
        (query, k),
    ).fetchall()
    return [KeywordHit(*row) for row in rows]
