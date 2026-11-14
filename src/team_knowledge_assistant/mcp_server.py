"""MCP server that lets GitHub Copilot (agent mode) search the knowledge base.

Copilot starts it over stdio from .vscode/mcp.json. Needs Postgres and LM Studio running.

    uv run python -m team_knowledge_assistant.mcp_server
"""

import re

from mcp.server.mcpserver import MCPServer

from team_knowledge_assistant.Ingest.embedding import embed_query
from team_knowledge_assistant.Ingest.store import Hit, connect, search

server = MCPServer(
    name="team-knowledge",
    instructions=(
        "Search the team's documentation. Each result has the source file and heading path; "
        "cite them as 'source > heading path'. A larger distance means a weaker match."
    ),
)

# A unified diff hunk starts with "@@ -a,b +c,d @@"; a file section starts with "diff --git".
FILE_HEADER = re.compile(r"^diff --git a/(\S+) b/\S+", re.MULTILINE)
MAX_QUERY_CHARS = 2000  # keep each embedded query well under the model's context


def _as_dict(h: Hit) -> dict:
    return {
        "source": h.source,
        "heading_path": " > ".join(h.heading_path),
        "distance": round(h.distance, 3),
        "content": h.content,
    }


@server.tool()
def search_docs(query: str, k: int = 5) -> list[dict]:
    """Find the documentation chunks most similar to a natural-language query."""
    with connect() as conn:
        return [_as_dict(h) for h in search(conn, embed_query(query), k=k)]


@server.tool()
def find_docs_for_diff(diff: str, commit_message: str = "", k_per_file: int = 3) -> list[dict]:
    """Find documentation chunks a code change may affect.

    Pass a unified diff (e.g. the output of `git show <sha>` or `git diff`) and the commit message.
    Each changed file is searched separately, so one big file can't drown out the others.
    Results are deduplicated and sorted by distance; `matched_files` says which changes hit each chunk.
    """
    parts = FILE_HEADER.split(diff)
    # split() with one capture group gives [preamble, file1, body1, file2, body2, ...]
    per_file = list(zip(parts[1::2], parts[2::2])) or [("(diff)", diff)]

    found: dict[tuple[str, str], dict] = {}
    with connect() as conn:
        for path, body in per_file:
            query = f"{commit_message}\nFile: {path}\n{body}"[:MAX_QUERY_CHARS]
            for h in search(conn, embed_query(query), k=k_per_file):
                key = (h.source, h.content)
                hit = found.setdefault(key, _as_dict(h) | {"matched_files": []})
                hit["distance"] = min(hit["distance"], round(h.distance, 3))
                hit["matched_files"].append(path)
    return sorted(found.values(), key=lambda d: d["distance"])


if __name__ == "__main__":
    server.run()  # stdio
