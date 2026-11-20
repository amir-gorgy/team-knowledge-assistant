"""The retrieval pipeline every caller shares: hybrid search picks candidates, a cross-encoder orders them.

Needs Postgres and the embeddings server; the first call also loads the reranker.
"""

from team_knowledge_assistant.Ingest.embedding import embed_query
from team_knowledge_assistant.Ingest.rerank import RerankedHit, rerank
from team_knowledge_assistant.Ingest.store import connect, hybrid_search

RERANK_CANDIDATES = 20  # hybrid hits the reranker re-orders; 40 scored no better on the eval and took 2x as long


def retrieve(query: str, k: int = 5) -> list[RerankedHit]:
    """The `k` chunks that best answer `query`, best first, each with a cross-encoder score.

    Always returns `k` hits when the table has that many, however weak: compare `score` with
    settings.reranker_not_found_below to tell a real match from the least-bad one.
    """
    with connect() as conn:
        candidates = hybrid_search(conn, query, embed_query(query), k=RERANK_CANDIDATES)
    return rerank(query, candidates, k=k)
