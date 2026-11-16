"""Search the chunks table by meaning (vector) or by exact words (keyword).

    uv run python -m team_knowledge_assistant.scripts.search "how do I declare a query parameter?" -k 3
    uv run python -m team_knowledge_assistant.scripts.search "404 when not found" -k 5 --mode keyword
"""

import argparse

from team_knowledge_assistant.Ingest.store import connect, keyword_search, vector_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["vector", "keyword"],
        default="vector",
        help="vector: closest meaning (lower distance is better); "
        "keyword: Postgres full-text search (higher rank is better)",
    )
    args = parser.parse_args()

    with connect() as conn:
        if args.mode == "keyword":
            hits = keyword_search(conn, args.query, k=args.k)
            scores = [f"rank {h.rank:.3f}" for h in hits]
        else:
            # Imported here so keyword mode works without the embeddings server running.
            from team_knowledge_assistant.Ingest.embedding import embed_query

            hits = vector_search(conn, embed_query(args.query), k=args.k)
            scores = [f"dist {h.distance:.3f}" for h in hits]

    if not hits:
        print("no matches")
    for i, (h, score) in enumerate(zip(hits, scores), 1):
        preview = " ".join(h.content.split())[:200]
        print(f"{i}. [{score}] {h.source} > {' > '.join(h.heading_path)}\n   {preview}\n")


if __name__ == "__main__":
    main()
