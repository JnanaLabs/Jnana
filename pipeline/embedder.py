"""
Embedder
Loads chunked JSON, generates embeddings, and upserts into Supabase pgvector.

Requires:
    - SUPABASE_URL and SUPABASE_KEY in .env
    - OPENAI_API_KEY in .env (or set EMBED_MODEL=local to use HuggingFace BAAI/bge-small-en)
    - Supabase table: vedic_chunks (see README for SQL schema)

Usage:
    python pipeline/embedder.py --book bg
    python pipeline/embedder.py --all
"""

import json
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CHUNKED_DATA_DIR = Path("data/chunked")
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")  # "openai" or "local"
OPENAI_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def get_openai_embeddings(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_local_embeddings(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model.encode(texts, normalize_embeddings=True).tolist()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    if EMBED_MODEL == "local":
        return get_local_embeddings(texts)
    return get_openai_embeddings(texts)


def get_supabase_client():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]], supabase):
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "chunk_id": chunk["chunk_id"],
            "book_id": chunk["metadata"]["book_id"],
            "book_name": chunk["metadata"]["book_name"],
            "chapter": chunk["metadata"]["chapter"],
            "verse": chunk["metadata"]["verse"],
            "url": chunk["metadata"]["url"],
            "title": chunk["metadata"]["title"],
            "translation": chunk["metadata"]["translation"],
            "chunk_type": chunk["chunk_type"],
            "text": chunk["text"],
            "embedding": embedding
        })
    # Upsert in batches
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        supabase.table("vedic_chunks").upsert(batch, on_conflict="chunk_id").execute()
        logger.info(f"  Upserted batch {i//BATCH_SIZE + 1} ({len(batch)} rows)")


def embed_book(book_id: str):
    chunk_file = CHUNKED_DATA_DIR / f"{book_id}_chunks.json"
    if not chunk_file.exists():
        logger.error(f"Chunks not found for {book_id}. Run chunker first.")
        return

    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)

    supabase = get_supabase_client()
    logger.info(f"[{book_id}] Embedding {len(chunks)} chunks with model={EMBED_MODEL}")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i+BATCH_SIZE]
        texts = [c["text"] for c in batch_chunks]
        embeddings = get_embeddings(texts)
        upsert_chunks(batch_chunks, embeddings, supabase)
        logger.info(f"[{book_id}] Progress: {min(i+BATCH_SIZE, len(chunks))}/{len(chunks)}")
        time.sleep(0.5)  # rate limit buffer

    logger.success(f"[{book_id}] All chunks embedded and stored in Supabase.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    with open("scraper/books_index.json") as f:
        index = json.load(f)

    books = index["books"]
    if args.all:
        for b in books:
            embed_book(b["id"])
    elif args.book:
        embed_book(args.book)
    else:
        parser.print_help()
