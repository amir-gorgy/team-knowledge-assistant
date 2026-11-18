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

settings = Settings()
