-- Chunk-level embeddings over weather_documents.narrative_text.
--
-- As with 01_, lakebase.ensure_weather_tables() runs this automatically; this
-- file is for review and manual inspection.
--
-- VECTOR(384) is fixed by the embedding model: sentence-transformers/
-- all-MiniLM-L6-v2 emits 384 dimensions. If you swap the model, this number
-- and lakebase.EMBEDDING_DIM must both change - pgvector rejects inserts whose
-- dimensionality differs from the column.
--   all-MiniLM-L6-v2   -> 384
--   all-mpnet-base-v2  -> 768
--   databricks-gte-large-en / bge-large-en -> 1024

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id          TEXT PRIMARY KEY,           -- '<document_id>_<chunk_index>'
    document_id TEXT NOT NULL
                    REFERENCES weather_documents (id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL,
    model_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);

-- HNSW rather than IVFFlat: IVFFlat picks its centroids at build time and so
-- needs representative rows to already be present, which is wrong for a table
-- that starts empty and grows on every sync. HNSW builds incrementally and
-- needs no `lists` tuning.
--
-- vector_cosine_ops matches the `<=>` operator used by POST /weather/search.
-- An index built with a different opclass (e.g. vector_l2_ops) is silently
-- ignored by a `<=>` ORDER BY, degrading it to a full scan.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding_hnsw
    ON weather_embeddings USING hnsw (embedding vector_cosine_ops);

-- Verify: udt_name should read 'vector', not '_float8'.
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;