"""Turn text into vectors through an OpenAI-compatible /v1/embeddings server (LM Studio by default)."""

import httpx2

from team_knowledge_assistant.config import settings

# nomic-embed-text was trained with task prefixes; without them retrieval gets noticeably worse.
# Other models don't want them, so the prefix is only added for nomic.
NOMIC_PREFIXES = {"document": "search_document: ", "query": "search_query: "}


def _with_prefix(texts: list[str], kind: str) -> list[str]:
    if "nomic" not in settings.embedding_model:
        return texts
    return [NOMIC_PREFIXES[kind] + t for t in texts]


def _embed(texts: list[str], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    with httpx2.Client(
        base_url=settings.embedding_base_url,
        headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
        timeout=120,
    ) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = client.post("/embeddings", json={"model": settings.embedding_model, "input": batch})
            resp.raise_for_status()
            # The API may return items out of order; "index" says which input each belongs to.
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)

    for v in vectors:
        if len(v) != settings.embedding_dim:
            raise ValueError(
                f"{settings.embedding_model} returned {len(v)} dims, expected {settings.embedding_dim}"
            )
    return vectors


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed chunks for storage."""
    return _embed(_with_prefix(texts, "document"), batch_size)


def embed_query(text: str) -> list[float]:
    """Embed a search query (uses the query prefix, not the document one)."""
    return _embed(_with_prefix([text], "query"), batch_size=1)[0]
