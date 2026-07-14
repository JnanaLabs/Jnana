"""
Embedder
Loads chunked JSON, generates embeddings, and upserts into Supabase pgvector.

Embedding provider priority (auto-detected from env keys):
  1. OpenAI  — OPENAI_API_KEY present → text-embedding-3-small (dim=1536)
  2. Gemini  — GEMINI_API_KEY present, no OpenAI → text-embedding-004 (dim=768)
  3. Local   — EMBED_MODEL=local or no keys → BAAI/bge-small-en-v1.5 (dim=384)

WARNING: All chunks must use the same model/dimension.
If you change models, drop the vedic_chunks table and re-run ingest.

Usage:
    python -m pipeline.embedder --book bg
    python -m pipeline.embedder --all
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
BATCH_SIZE = 100


def _active_embed_provider() -> str:
    setting = os.getenv("EMBED_MODEL", "auto").lower()
    if setting == "local":
        return "local"
    if setting == "openai" or (setting == "auto" and os.getenv("OPENAI_API_KEY")):
        return "openai"
    if setting == "gemini" or (setting == "auto" and os.getenv("GEMINI_API_KEY")):
        if setting == "auto":
            logger.info("EMBED_MODEL=auto: no OpenAI key found, using Gemini embeddings (dim=768).")
        return "gemini"
    logger.warning("No API keys found — using local BAAI/bge-small-en-v1.5 (dim=384).")
    return "local"


def get_embeddings(texts: list[str]) -> list[list[float]]:
    provider = _active_embed_provider()

    if provider == "openai":
        from openai import OpenAI
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        logger.debug(f"Embedding with OpenAI: {model}")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
        logger.debug(f"Embedding with Gemini: {model}")
        result = []
        for text in texts:
            res = genai.embed_content(model=model, content=text, task_type="retrieval_document")
            result.append(res["embedding"])
        return result

    else:  # local
        from sentence_transformers import SentenceTransformer
        logger.debug("Embedding with local BAAI/bge-small-en-v1.5")
        m = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return m.encode(texts, normalize_embeddings=True).tolist()


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """Upsert chunks into Supabase using the write (service role) client."""
    from core.supabase_client import get_write_client
    supabase = get_write_client()

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "chunk_id":    chunk["chunk_id"],
            "book_id":     chunk["metadata"]["book_id"],
            "book_name":   chunk["metadata"]["book_name"],
            "chapter":     chunk["metadata"]["chapter"],
            "verse":       chunk["metadata"]["verse"],
            "url":         chunk["metadata"]["url"],
            "title":       chunk["metadata"]["title"],
            "translation": chunk["metadata"]["translation"],
            "chunk_type":  chunk["chunk_type"],
            "text":        chunk["text"],
            "embedding":   embedding,
        })

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        supabase.table("vedic_chunks").upsert(batch, on_conflict="chunk_id").execute()
        logger.info(f"  Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} rows)")


def embed_book(book_id: str):
    chunk_file = CHUNKED_DATA_DIR / f"{book_id}_chunks.json"
    if not chunk_file.exists():
        logger.error(f"Chunks not found for {book_id}. Run chunker first: python -m pipeline.chunker --book {book_id}")
        return

    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)

    provider = _active_embed_provider()
    logger.info(f"[{book_id}] Embedding {len(chunks)} chunks | provider={provider}")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch_chunks]
        embeddings = get_embeddings(texts)
        upsert_chunks(batch_chunks, embeddings)
        logger.info(f"[{book_id}] Progress: {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")
        time.sleep(0.5)  # gentle rate-limit buffer

    logger.success(f"[{book_id}] Done — all chunks embedded and stored in Supabase.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunked scripture data into Supabase")
    parser.add_argument("--book", type=str, help="Book ID (e.g. bg, sb, cc)")
    parser.add_argument("--all", action="store_true", help="Embed all books")
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
