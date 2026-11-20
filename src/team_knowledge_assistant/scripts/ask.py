"""Ask a question and get an answer written from the docs, with its sources, or "not found".

Needs Postgres, the embeddings server and the chat model (LM Studio by default). With
ANSWER_BACKEND=copilot no chat model is used: it prints the sources that passed the gate instead.

    uv run python -m team_knowledge_assistant.scripts.ask "How do I make a query parameter optional?"
    uv run python -m team_knowledge_assistant.scripts.ask "How do I rate-limit requests per user?" --show-hits
"""

import argparse

from team_knowledge_assistant.answer import answer_from_hits, gate
from team_knowledge_assistant.config import settings
from team_knowledge_assistant.retrieval import retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="chunks retrieved before the not-found gate")
    parser.add_argument("--show-hits", action="store_true", help="list the retrieved chunks and which passed the gate")
    args = parser.parse_args()

    hits = retrieve(args.question, k=args.k)
    if args.show_hits:
        # The model sees only the hits that pass the gate, numbered [1], [2], ... in that order.
        number = {h.id: n for n, h in enumerate(gate(hits), 1)}
        print(f"Retrieved (sent to the model if score >= {settings.reranker_not_found_below}):")
        for h in hits:
            mark = f"[{number[h.id]}]" if h.id in number else "dropped"
            print(f"  {h.score:>6.2f}  {mark:<8}{' > '.join(h.heading_path)}")
        print()

    result = answer_from_hits(args.question, hits)
    if result.text is None:  # ANSWER_BACKEND=copilot: no model writes the answer
        print("No answer written (ANSWER_BACKEND=copilot). Sources to answer from:\n")
        for s in result.sources:
            preview = " ".join(s.content.split())[:200]
            print(f"[{s.n}] {s.source} > {' > '.join(s.heading_path)}\n    {preview}\n")
        return
    print(result.text)
    if not result.found:
        print(f"\n({result.not_found_reason})")
        return
    print("\nSources:")
    for c in result.citations:
        print(f"  [{c.n}] {c.source} > {' > '.join(c.heading_path)}")


if __name__ == "__main__":
    main()
