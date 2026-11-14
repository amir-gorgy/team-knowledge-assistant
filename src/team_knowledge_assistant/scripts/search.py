"""Similarity search over the chunks table.

    uv run python -m team_knowledge_assistant.scripts.search "how do I declare a query parameter?" -k 3
"""

import argparse

from team_knowledge_assistant.Ingest.embedding import embed_query
from team_knowledge_assistant.Ingest.store import connect, search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    with connect() as conn:
        hits = search(conn, embed_query(args.query), k=args.k)
    for i, h in enumerate(hits, 1):
        preview = " ".join(h.content.split())[:200]
        print(f"{i}. [{h.distance:.3f}] {h.source} > {' > '.join(h.heading_path)}\n   {preview}\n")


if __name__ == "__main__":
    main()
