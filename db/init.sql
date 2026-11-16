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
    created_at      timestamptz NOT NULL DEFAULT now(),
    -- Words of `content` for keyword (full-text) search: lowercased, stemmed, stop words dropped.
    -- Postgres fills it on every insert/update, so ingest code never writes it.
    content_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- GIN index: maps each word to the rows containing it, so `content_tsv @@ query` skips a full scan.
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING gin (content_tsv);
