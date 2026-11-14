CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk from the Markdown chunker.
-- vector(768) matches nomic-embed-text-v1.5; a model with another size needs a new column/table.
CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    source          text        NOT NULL,              -- file path, for citations
    heading_path    text[]      NOT NULL DEFAULT '{}', -- e.g. {"SQL Databases","Create the Tables"}
    content         text        NOT NULL,
    embedding_model text        NOT NULL,
    embedding       vector(768) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);
