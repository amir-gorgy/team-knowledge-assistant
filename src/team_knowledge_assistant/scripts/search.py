"""Search the chunks table by meaning (vector), by exact words (keyword), or both merged (hybrid).

    uv run python -m team_knowledge_assistant.scripts.search "how do I declare a query parameter?" -k 3
    uv run python -m team_knowledge_assistant.scripts.search "404 when not found" -k 5 --mode keyword
    uv run python -m team_knowledge_assistant.scripts.search "404 when not found" -k 5 --mode hybrid
"""

import argparse

from team_knowledge_assistant.Ingest.store import connect, hybrid_search, keyword_search, vector_search
from team_knowledge_assistant.Ingest.embedding import embed_query

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["vector", "keyword", "hybrid"],
        default="vector",
        help="vector: closest meaning (lower distance is better); "
        "keyword: Postgres full-text search, every word must appear (higher rank is better); "
        "hybrid: vector + any-word keyword, merged by reciprocal rank fusion (higher score is better)",
    )
    args = parser.parse_args()

    with connect() as conn:
        if args.mode == "keyword":
            hits = keyword_search(conn, args.query, k=args.k)
            scores = [f"rank {h.rank:.3f}" for h in hits]
        elif args.mode == "hybrid":
            hits = hybrid_search(conn, args.query, embed_query(args.query), k=args.k)
            # "v3 k-" = 3rd in the vector list, not in the keyword list
            scores = [
                f"rrf {h.score:.4f} v{h.vector_rank or '-'} k{h.keyword_rank or '-'}" for h in hits
            ]
        else:
            # Imported here so keyword mode works without the embeddings server running.
            # from team_knowledge_assistant.Ingest.embedding import embed_query

            hits = vector_search(conn, embed_query(args.query), k=args.k)
            scores = [f"dist {h.distance:.3f}" for h in hits]

    if not hits:
        print("no matches")
    for i, (h, score) in enumerate(zip(hits, scores), 1):
        preview = " ".join(h.content.split())[:200]
        print(f"{i}. [{score}] {h.source} > {' > '.join(h.heading_path)}\n   {preview}\n")


if __name__ == "__main__":
    main()
