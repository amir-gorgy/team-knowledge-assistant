"""Chunk Markdown files, embed the chunks and store them in pgvector.

    uv run python -m team_knowledge_assistant.scripts.ingest "data/sample-docs-md/First Steps - FastAPI.md" --limit 20
    uv run python -m team_knowledge_assistant.scripts.ingest data/sample-docs-md --reset
"""

import argparse
from pathlib import Path

from team_knowledge_assistant.Ingest.chunking import MarkdownChunker
from team_knowledge_assistant.Ingest.embedding import embed_documents
from team_knowledge_assistant.Ingest.store import connect, insert_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="a .md file or a folder of them")
    parser.add_argument("--limit", type=int, help="only embed the first N chunks")
    parser.add_argument("--reset", action="store_true", help="empty the chunks table first")
    args = parser.parse_args()

    files = sorted(args.path.glob("*.md")) if args.path.is_dir() else [args.path]
    chunker = MarkdownChunker()
    chunks = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        chunks.extend(chunker.chunk(text, source=str(file)))
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"{len(files)} file(s) -> {len(chunks)} chunks")

    embeddings = embed_documents([c.text for c in chunks])

    with connect() as conn:
        if args.reset:
            conn.execute("TRUNCATE chunks RESTART IDENTITY")
        inserted = insert_chunks(conn, chunks, embeddings)
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"inserted {inserted}; chunks table now has {total} rows")


if __name__ == "__main__":
    main()
