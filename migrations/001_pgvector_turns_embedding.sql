-- LexShield AI — pgvector Migration: Semantic Embeddings on `turns`
--
-- MANUAL ONE-TIME STEP (requires superuser / owner privileges):
-- On Supabase, run this ONCE via the SQL Editor. The app's runtime
-- DB role likely does NOT have permission to create extensions,
-- so this is NOT included in _init_db().

CREATE EXTENSION IF NOT EXISTS vector;

-- Add the 384-dim embedding column (idempotent).
-- Existing rows get NULL — filtered out by WHERE embedding IS NOT NULL.
ALTER TABLE turns ADD COLUMN IF NOT EXISTS embedding vector(384);

-- HNSW index for fast cosine similarity search.
-- Uses vector_cosine_ops to match <=> operator and normalize_embeddings=True.
CREATE INDEX IF NOT EXISTS idx_turns_embedding
    ON turns USING hnsw (embedding vector_cosine_ops);
