"""
Ingest Orchestrator
Runs the full pipeline: scrape → chunk → embed for one or all books.

Usage:
    python pipeline/ingest.py --book bg
    python pipeline/ingest.py --all
"""

import asyncio
import argparse
import json
from loguru import logger
from scraper.vedabase_scraper import scrape_book
from pipeline.chunker import chunk_book
from pipeline.embedder import embed_book


async def ingest_book(book: dict):
    logger.info(f"=== Ingesting {book['name']} ===")
    await scrape_book(book)
    chunk_book(book["id"], book["name"])
    embed_book(book["id"])
    logger.success(f"=== {book['name']} fully ingested ===")


async def ingest_all():
    with open("scraper/books_index.json") as f:
        index = json.load(f)
    for book in index["books"]:
        await ingest_book(book)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=str)
    parser.add_argument("--all", action="store_true")
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
            print(f"Book '{args.book}' not found")
    else:
        parser.print_help()
