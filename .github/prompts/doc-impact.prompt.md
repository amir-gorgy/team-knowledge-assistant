---
description: Find the docs a commit affects and suggest rewording to match the change
---

Check which documentation a commit affects, using the `team-knowledge` MCP server.

1. Get the commit. Use the SHA the user gave, otherwise `HEAD`. Run `git show --stat --patch <sha>`
   in the terminal and keep the commit message and the diff.
2. Call `find_docs_for_diff` with the diff and the commit message.
3. Treat results with distance above ~0.30 as unrelated and drop them. If nothing is left, say
   "No documentation looks affected" and stop. Do not guess.
4. For each remaining chunk, decide whether the change makes it wrong, incomplete or still correct.
   Use `search_docs` if you need more context around it.
5. Report one row per affected chunk, most affected first:
   - **Where:** `source > heading path`
   - **Why:** one sentence tying it to a specific line of the diff
   - **Suggested wording:** the replacement text, kept in the doc's existing style

Only suggest edits. Do not change any files.
