# Copilot doc-impact agent

Goal: on each commit, find the documentation chunks the change affects and suggest how to reword them
so the docs keep matching the code. GitHub Copilot is the agent; this project supplies retrieval over
the team's docs through an MCP server.

## How it fits together

```
git commit ──► Copilot (agent mode, /doc-impact)
                 │  git show <sha>  →  message + diff
                 ▼
               MCP server  team_knowledge_assistant.mcp_server   (stdio, started by VS Code)
                 │  find_docs_for_diff / search_docs
                 ▼
               LM Studio /v1/embeddings  ──►  Postgres + pgvector (chunks table)
                 nomic-embed-text-v1.5          cosine distance, top-k
                 ▲
               Copilot's model reads the hits and writes the rewording suggestions
```

Everything except Copilot's own model runs locally. Docs and diffs go to the embedding model on this
machine only; the hits and the diff are sent to Copilot's model when it writes suggestions.

## MCP tools

| Tool | Input | Returns |
| --- | --- | --- |
| `search_docs` | `query`, `k` (default 5) | chunks with `source`, `heading_path`, `distance`, `content` |
| `find_docs_for_diff` | unified `diff`, `commit_message`, `k_per_file` (default 3) | the same, deduplicated across files, plus `matched_files` |

`find_docs_for_diff` embeds each changed file's hunks (with the commit message) as its own query, so one
large file can't hide matches for the others.

## Setup

Short version below. For a full walkthrough with expected output and troubleshooting, see [try-it.md](try-it.md).

1. `docker compose up -d db`. On a fresh volume, `db/init.sql` creates the `chunks` table.
2. In LM Studio, load `text-embedding-nomic-embed-text-v1.5` and start the server on port 1234.
3. Ingest the docs:
   `uv run python -m team_knowledge_assistant.scripts.ingest data/sample-docs-md --reset`
4. In VS Code, open `.vscode/mcp.json` and start `team-knowledge` (or run **MCP: List Servers**).
5. In Copilot Chat (agent mode), run `/doc-impact` or `/doc-impact <sha>`.

The embedding endpoint is any OpenAI-compatible server, set through `EMBEDDING_*` in `.env`
(see `.env.example`). Changing the model means changing `EMBEDDING_DIM` and `vector(N)` in
`db/init.sql`, then re-ingesting.

## Known gaps / next steps

- **Distance threshold:** the cut-off (~0.30) is a guess from a few queries. Tune it on the eval set.
- **Vector-only search:** exact names in diffs (functions, flags, endpoints) need keyword search too.
  Add Postgres full-text search and merge the two rankings (hybrid search).
- **Runs on demand:** a `post-commit` hook or a GitHub Action could call the same tools and post the
  report on the PR. An Action can't reach LM Studio on localhost, so it would need a hosted embedding
  endpoint.
- **Chunk noise:** PDF-to-Markdown leftovers ("Ask AI" picture text, blank lines) end up in chunks
  and their embeddings.
