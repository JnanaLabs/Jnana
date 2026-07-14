"""
Ingest Orchestrator
Runs the full pipeline for one or all books: scrape → chunk → embed.

Usage:
    python -m pipeline.ingest --book bg
    python -m pipeline.ingest --all
"""

import asyncio
import argparse
import json
from loguru import logger

# All imports use absolute module paths (run from project root)
from scraper.vedabase_scraper import scrape_book
from pipeline.chunker import chunk_book
from pipeline.embedder import embed_book


async def ingest_book(book: dict):
    logger.info(f"\n{'='*50}")
    logger.info(f"Ingesting: {book['name']}")
    logger.info(f"{'='*50}")
    await scrape_book(book)
    chunk_book(book["id"], book["name"])
    embed_book(book["id"])
    logger.success(f"{book['name']} — fully ingested.")


async def ingest_all():
    with open("scraper/books_index.json") as f:
        index = json.load(f)
    for book in index["books"]:
        await ingest_book(book)
        await asyncio.sleep(3)  # courtesy pause between books


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full ingest pipeline: scrape → chunk → embed")
    parser.add_argument("--book", type=str, help="Book ID (e.g. bg, sb, cc)")
    parser.add_argument("--all", action="store_true", help="Ingest all books")
    args = parser.parse_args()

    with open("scraper/books_index.json") as f:
        index = json.load(f)
    books = index["books"]

    if args.all:
        asyncio.run(ingest_all())
    elif args.book:
        book = next((b for b in books if b["id"] == args.book), None)
        if book:
            asyncio.run(ingest_book(book))
        else:
            print(f"Book '{args.book}' not found in books_index.json")
            print(f"Available: {[b['id'] for b in books]}")
    else:
        parser.print_help()
