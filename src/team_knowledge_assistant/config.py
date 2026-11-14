from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://app:app@localhost:5432/tka"

    # Any OpenAI-compatible /v1/embeddings server. Default: LM Studio on this machine.
    embedding_base_url: str = "http://localhost:1234/v1"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_api_key: str = "lm-studio"  # LM Studio ignores it; hosted APIs need a real key
    embedding_dim: int = 768  # must match vector(N) in db/init.sql

settings = Settings()
