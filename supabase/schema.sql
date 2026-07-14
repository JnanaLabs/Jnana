-- ─────────────────────────────────────────────────────────────────
-- Jnana — Vedic RAG Schema
-- Using Gemini text-embedding-004 → dim=768
-- If switching to OpenAI text-embedding-3-small, change 768 → 1536
-- If switching to local BAAI/bge-small-en-v1.5, change 768 → 384
-- ─────────────────────────────────────────────────────────────────

-- Enable pgvector
create extension if not exists vector;

-- Main chunks table
create table if not exists vedic_chunks (
  id          bigserial primary key,
  chunk_id    text unique not null,
  book_id     text not null,
  book_name   text,
  chapter     text,
  verse       text,
  url         text,
  title       text,
  translation text,
  chunk_type  text,  -- 'verse_full', 'verse_only', 'purport'
  text        text not null,
  embedding   vector(768),  -- Gemini text-embedding-004
  created_at  timestamptz default now()
);

-- Vector search index
create index if not exists vedic_chunks_embedding_idx
  on vedic_chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Book filter index
create index if not exists vedic_chunks_book_id_idx on vedic_chunks(book_id);

-- ─────────────────────────────────────────────────────────────────
-- RPC: semantic search (no book filter)
-- ─────────────────────────────────────────────────────────────────
create or replace function match_vedic_chunks(
  query_embedding vector(768),
  match_count int default 6,
  similarity_threshold float default 0.75
)
returns table (
  chunk_id    text,
  book_id     text,
  book_name   text,
  chapter     text,
  verse       text,
  url         text,
  title       text,
  translation text,
  chunk_type  text,
  text        text,
  similarity  float
)
language sql stable
as $$
  select
    chunk_id, book_id, book_name, chapter, verse, url, title, translation, chunk_type, text,
    1 - (embedding <=> query_embedding) as similarity
  from vedic_chunks
  where 1 - (embedding <=> query_embedding) > similarity_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- ─────────────────────────────────────────────────────────────────
-- RPC: semantic search with book filter
-- ─────────────────────────────────────────────────────────────────
create or replace function match_vedic_chunks_filtered(
  query_embedding vector(768),
  filter_book_id  text,
  match_count int default 6,
  similarity_threshold float default 0.75
)
returns table (
  chunk_id    text,
  book_id     text,
  book_name   text,
  chapter     text,
  verse       text,
  url         text,
  title       text,
  translation text,
  chunk_type  text,
  text        text,
  similarity  float
)
language sql stable
as $$
  select
    chunk_id, book_id, book_name, chapter, verse, url, title, translation, chunk_type, text,
    1 - (embedding <=> query_embedding) as similarity
  from vedic_chunks
  where book_id = filter_book_id
    and 1 - (embedding <=> query_embedding) > similarity_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
