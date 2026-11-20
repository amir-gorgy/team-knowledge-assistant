# Try it: local and Copilot

Two ways to run the knowledge base. Both use the same Postgres database and the same embedding model in
LM Studio:

- **Local:** run the scripts in a terminal. No Copilot needed.
- **Copilot:** VS Code starts the MCP server, and Copilot calls its tools in agent mode.

Run every command from the project root. For how the parts fit together, see
[copilot-agent.md](copilot-agent.md).

## Prerequisites

- Docker with Compose
- [uv](https://docs.astral.sh/uv/)
- [LM Studio](https://lmstudio.ai/) with `text-embedding-nomic-embed-text-v1.5` downloaded, and its `lms` CLI.
  For answers written locally (`ask` with the default `ANSWER_BACKEND=local`), also `qwen/qwen3.5-9b`
  (~10 GB loaded)
- VS Code with GitHub Copilot Chat (Copilot path only)
- Node.js (only for the optional MCP Inspector)

## 0. Start the services and load the docs

Both paths need this. Repeat it after a reboot.

```bash
docker compose up -d db                                   # Postgres + pgvector on :5432
lms server start                                          # LM Studio API on :1234
lms load text-embedding-nomic-embed-text-v1.5             # the embedding model (not a chat model)
curl -s localhost:1234/v1/models | grep nomic             # should print the model id
lms load qwen/qwen3.5-9b --context-length 8192            # chat model, only for `ask` with ANSWER_BACKEND=local
```

Load the chat model with `--context-length 8192` or more. Five chunks can reach about 7k tokens, and a
smaller context cuts the prompt off silently.

Optional: copy `.env.example` to `.env` to change the database URL or the embedding endpoint. Without it,
the defaults in `config.py` point at the services above.

Build the sample docs. `data/sample-docs-md/` isn't in git, so a fresh clone starts without it. This
downloads FastAPI's Markdown docs from GitHub (pinned to release 0.165.0), inlines their code examples
and writes one flat file per page:

```bash
uv run python data/fastapi-docs-to-md.py                  # 131 pages -> data/sample-docs-md
```

It stops if the folder already has `.md` files; move them away first to rebuild.

Embed the sample docs into pgvector. `--reset` empties the table first, so the command can be re-run
safely:

```bash
uv run python -m team_knowledge_assistant.scripts.ingest data/sample-docs-md --reset
```

Expected output: `131 file(s) -> 1203 chunks` and `chunks table now has 1203 rows`. The numbers change if
the docs or the chunker change. What matters is that the row count matches the chunk count.

To test a small batch first, add `--limit 20`.

> `db/init.sql` creates the `chunks` table only when the Docker volume is new. If you get
> `relation "chunks" does not exist`, apply it by hand:
> `docker compose exec -T db psql -U app -d tka < db/init.sql`
>
> A database created before keyword search was added has the table but not its word column. If keyword
> or hybrid search fails with `column "content_tsv" does not exist`, add it once (no re-ingest needed;
> Postgres fills it for the existing rows):
>
> ```bash
> docker compose exec -T db psql -U app -d tka \
>   -c "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;" \
>   -c "CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING gin (content_tsv);"
> ```

## Option A: Local

### 1. Search the docs

```bash
uv run python -m team_knowledge_assistant.scripts.search "how do I set a response status code?" -k 5
```

Each hit prints `[distance] file > heading path` and a preview. Lower distance means a closer match.
Good matches land around 0.20–0.25. Above about 0.30 they're usually unrelated. Try questions you know
the answer to, and check that the right section makes the top 3.

`--mode` picks the search:

```bash
uv run python -m team_knowledge_assistant.scripts.search "404 when not found" -k 5 --mode keyword
uv run python -m team_knowledge_assistant.scripts.search "404 when not found" -k 5 --mode hybrid
```

| Mode | Finds | Score shown |
| --- | --- | --- |
| `vector` (default) | closest meaning | `dist 0.316`: lower is better |
| `keyword` | chunks containing **every** word; long questions often match nothing. Skips LM Studio | `rank 0.183`: higher is better |
| `hybrid` | both lists (keyword with **any** word) merged by reciprocal rank fusion | `rrf 0.0263 v17 k15`: higher is better; 17th by vector, 15th by keyword (`-` = not in that list) |

Hybrid helps when the answer uses an exact name the question also uses (`404`, `HTTPException`), and can
hurt when other pages repeat the question's words more often. Compare the modes on questions you know.

Add `--rerank` to any mode to re-order its top 20 with a cross-encoder, the same second pass
`search_docs` uses. Hits print as `rerank 3.03 was #2` (higher is better; `was` is the first-stage
rank). If even the best score is below -1.5, it says the docs probably don't cover the question. The
first run downloads the model (~80 MB).

To score the modes on the eval questions instead of reading results by eye:

```bash
uv run python -m team_knowledge_assistant.scripts.evaluate --modes vector hybrid hybrid+rerank
```

### 2. Ask a question

`ask` retrieves the top 5 chunks the same way `search_docs` does (hybrid, then rerank), and the chat
model writes an answer from them with `[n]` citations:

```bash
uv run python -m team_knowledge_assistant.scripts.ask "How do I make a query parameter optional?"
uv run python -m team_knowledge_assistant.scripts.ask "How do I rate-limit requests per user?" --show-hits
```

The answer is followed by a **Sources** list, one line per `[n]` the answer uses. `--show-hits` first
lists the 5 retrieved chunks with their rerank score and the `[n]` each was sent as.

It prints `Not found in the docs.` instead of guessing, with the reason underneath:

| Reason | What happened |
| --- | --- |
| `no_relevant_chunks` | Every chunk scored below `RERANKER_NOT_FOUND_BELOW` (-1.5). They are never sent, and the model isn't called. |
| `model_said_not_found` | Some chunks passed, but the model replied `NOT_FOUND`: they don't answer the question. |
| `no_valid_citations` | The model answered without citing any source it was sent, so the answer isn't shown. |

To try the model's own judgment without the score check, turn it off for one run:
`RERANKER_NOT_FOUND_BELOW=-10 uv run python -m team_knowledge_assistant.scripts.ask "..."`.

The same answer is available over HTTP. Start the API, then post a question:

```bash
uv run uvicorn team_knowledge_assistant.main:app --reload
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "My endpoint should return a 404 when the item does not exist. How do I do that?"}'
```

It returns `{"text", "found", "citations": [{"n", "source", "heading_path"}], "not_found_reason",
"sources"}`. A not-found answer is still a `200` with `"found": false`. A `503` means Postgres or LM
Studio is down, and a `502` means the model's reply was cut off.

**Without a chat model.** Set `ANSWER_BACKEND=copilot` in `.env` to skip the local model. The not-found
check still runs, but no answer is written: `text` is `null` and `sources` lists the chunks that passed,
numbered `[n]`. Through MCP, Copilot writes the answer from them, following the `instructions` field the
tool adds. The CLI and `POST /ask` return the sources for you to read. The embedding model is still
needed.

| | `ANSWER_BACKEND=local` (default) | `ANSWER_BACKEND=copilot` |
| --- | --- | --- |
| Writes the answer | the chat model in LM Studio | Copilot (MCP); nobody (CLI, `POST /ask`) |
| "Not found" when no chunk scores above -1.5 | yes | yes |
| Model's `NOT_FOUND` and citation checks | yes, in code | no: Copilot is told the rules but they aren't checked |
| Needs the chat model loaded | yes | no |

### 3. Check a commit's diff against the docs

This is what Copilot's `/doc-impact` does, without Copilot:

```bash
uv run python -c "
import subprocess
from team_knowledge_assistant.mcp_server import find_docs_for_diff
diff = subprocess.run(['git', 'show', 'HEAD'], capture_output=True, text=True).stdout
for r in find_docs_for_diff(diff, 'commit message here')[:5]:
    print(r['distance'], r['source'].split('/')[-1], '>', r['heading_path'], r['matched_files'])
"
```

### 4. (Optional) Browse the MCP tools

```bash
npx @modelcontextprotocol/inspector uv run python -m team_knowledge_assistant.mcp_server
```

Connect, open **Tools**, and call `search_docs` or `ask`. Copilot sees the same three tools.

## Option B: Copilot (VS Code)

### 1. Start the MCP server

Open [`.vscode/mcp.json`](../.vscode/mcp.json) and click **Start** above `"team-knowledge"`. Or open the
Command Palette and run **MCP: List Servers → team-knowledge → Start Server**. Say yes when VS Code asks
whether to trust the server. It should show **Running** with 3 tools.

### 2. Switch Copilot Chat to Agent mode

Pick **Agent** in the chat mode picker. Ask and Edit modes don't call MCP tools. Click the **tools** icon
and check that `search_docs`, `ask` and `find_docs_for_diff` are ticked under `team-knowledge`.

### 3. Try a search

> Use search_docs to find how FastAPI handles optional query parameters, and cite the sources.

Approve the tool call when asked. The answer should cite
`Query Parameters - FastAPI.md > Query Parameters > Optional parameters`.

`ask` works the same way: "Use ask to find out how to make a query parameter optional." With
`ANSWER_BACKEND=copilot`, Copilot writes the answer itself from the sources the tool returns, so the
chat model doesn't need to be loaded. The server reads `.env` only when it starts, so restart it
(**MCP: List Servers → team-knowledge → Restart Server**) after changing the setting.

### 4. Try the commit check

The sample docs are FastAPI docs, so make a change they cover. Use a throwaway branch:

```bash
git switch -c try-doc-impact
```

Add this to `src/team_knowledge_assistant/main.py`:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

Commit only that file:

```bash
git add src/team_knowledge_assistant/main.py
git commit -m "Add item endpoint with optional query parameter"
```

In Copilot Chat (Agent mode), run `/doc-impact` (or `/doc-impact <sha>` for another commit). It reads the
commit, calls `find_docs_for_diff`, drops matches above about 0.30, and reports **Where / Why / Suggested
wording** for each affected section. Expect sections like *Path Parameters* and
*Query Parameters > Optional parameters*. It only suggests edits and doesn't change any files.

### 5. Clean up

```bash
git switch main && git branch -D try-doc-impact
```

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| MCP server won't start or shows Error | Run **MCP: List Servers → team-knowledge → Show Output**. Usually Postgres or LM Studio isn't running (step 0). |
| `Connection refused` on :1234 | Run `lms server start`. |
| `400` from `/embeddings`, or a dims error | The embedding model isn't loaded, or `.env` names a different model. A model with another vector size needs `EMBEDDING_DIM` and `vector(N)` in `db/init.sql` changed, then a re-ingest. |
| `relation "chunks" does not exist` | Apply `db/init.sql` by hand (see step 0). |
| `/doc-impact` doesn't appear | Make sure prompt files are enabled (`chat.promptFiles` setting) and you're in Agent mode. If it still doesn't show, remove the frontmatter from `.github/prompts/doc-impact.prompt.md`. |
| Copilot answers without calling the tools | Name the tool in the request ("use search_docs…") or tick it in the tools picker. |
| Every result is above 0.30 | The change isn't covered by the docs, so "nothing affected" is the correct answer. |
| `ask` fails with `stopped at max_tokens` (or `/ask` returns `502`) | The model is thinking before it answers and uses up `CHAT_MAX_TOKENS`. Keep `CHAT_REASONING_EFFORT=none`, or raise `CHAT_MAX_TOKENS` if you turn thinking on. |
| `ask` fails with an HTTP error from `/chat/completions` | The chat model isn't downloaded or loaded, or `.env` names a different one. Run `lms load qwen/qwen3.5-9b --context-length 8192`. |
