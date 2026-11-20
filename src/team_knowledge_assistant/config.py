from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://app:app@localhost:5432/tka"

    # Any OpenAI-compatible /v1/embeddings server. Default: LM Studio on this machine.
    embedding_base_url: str = "http://localhost:1234/v1"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_api_key: str = "lm-studio"  # LM Studio ignores it; hosted APIs need a real key
    embedding_dim: int = 768  # must match vector(N) in db/init.sql

    # Cross-encoder for the rerank step, run in-process by fastembed (downloaded on first use).
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # A best rerank score below this means "the docs probably don't cover it". From the eval set:
    # every not-found question's top score was <= -1.57, and 27 of 28 answerable ones were above.
    # The scale is the model's own, so re-measure this whenever reranker_model changes.
    reranker_not_found_below: float = -1.5

    # Who writes `ask` answers. The not-found gate runs in code either way.
    #   local:   the chat model below writes the answer, and its citations are checked.
    #   copilot: no model is called. `ask` returns the numbered sources that passed the gate, and the
    #            caller writes the answer: Copilot through the MCP tool, or whoever calls the CLI/API.
    answer_backend: Literal["local", "copilot"] = "local"

    # Any OpenAI-compatible /v1/chat/completions server, for writing answers when answer_backend is
    # "local". Default: LM Studio.
    # Load the model with a context of at least 8192 tokens: 5 chunks can reach ~7k.
    chat_base_url: str = "http://localhost:1234/v1"
    chat_model: str = "qwen/qwen3.5-9b"
    chat_api_key: str = "lm-studio"
    # "none" turns Qwen's thinking off. With it on, the thinking alone used up 1,000 tokens (18 s)
    # on a two-source test and the answer never came; off, the same test took 1.2 s.
    chat_reasoning_effort: str = "none"
    chat_max_tokens: int = 1024  # an answer cut off at this limit is an error, never shown

settings = Settings()
