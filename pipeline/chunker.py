"""
Chunker
Converts raw scraped verse JSON into structured chunks ready for embedding.
Each chunk = one verse unit (transliteration + synonyms + translation + purport).
Metadata is preserved for filtering and citation.

Usage:
    python pipeline/chunker.py --book bg
    python pipeline/chunker.py --all
"""

import json
import argparse
from pathlib import Path
from typing import Optional
from loguru import logger

RAW_DATA_DIR = Path("data/raw")
CHUNKED_DATA_DIR = Path("data/chunked")
CHUNKED_DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_PURPORT_CHUNK_CHARS = 1500  # split long purports into sub-chunks


def split_text(text: str, max_chars: int, overlap: int = 200) -> list[str]:
    """Split long text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to split on paragraph boundary
        split_at = text.rfind("\n\n", start, end)
        if split_at == -1 or split_at <= start:
            split_at = text.rfind(". ", start, end)
        if split_at == -1 or split_at <= start:
            split_at = end
        else:
            split_at += 2  # include the period/newline
        chunks.append(text[start:split_at].strip())
        start = split_at - overlap
    return [c for c in chunks if c.strip()]


def build_chunk_text(verse: dict, include_purport: bool = True) -> str:
    """Build the main text content for a verse chunk."""
    parts = []
    if verse.get("title"):
        parts.append(f"Reference: {verse['title']}")
    if verse.get("transliteration"):
        parts.append(f"Transliteration:\n{verse['transliteration']}")
    if verse.get("synonyms"):
        parts.append(f"Word-for-word:\n{verse['synonyms']}")
    if verse.get("translation"):
        parts.append(f"Translation:\n{verse['translation']}")
    if include_purport and verse.get("purport"):
        parts.append(f"Purport:\n{verse['purport']}")
    return "\n\n".join(parts)


def chunk_verse(verse: dict, book_name: str) -> list[dict]:
    """Convert a single scraped verse into one or more chunks."""
    base_meta = {
        "book_id": verse.get("book_id"),
        "book_name": book_name,
        "chapter": verse.get("chapter"),
        "verse": verse.get("verse"),
        "url": verse.get("url"),
        "title": verse.get("title"),
        "translation": verse.get("translation"),  # kept for quick display
    }

    purport = verse.get("purport") or ""
    chunks = []

    if len(purport) <= MAX_PURPORT_CHUNK_CHARS:
        # Single chunk: full verse + purport
        text = build_chunk_text(verse, include_purport=True)
        chunks.append({
            "chunk_id": f"{verse.get('book_id')}_{verse.get('chapter')}_{verse.get('verse')}_0",
            "text": text,
            "chunk_type": "verse_full",
            "metadata": base_meta
        })
    else:
        # Verse-only chunk
        verse_text = build_chunk_text(verse, include_purport=False)
        chunks.append({
            "chunk_id": f"{verse.get('book_id')}_{verse.get('chapter')}_{verse.get('verse')}_verse",
            "text": verse_text,
            "chunk_type": "verse_only",
            "metadata": base_meta
        })
        # Split purport into sub-chunks
        purport_splits = split_text(purport, MAX_PURPORT_CHUNK_CHARS)
        for i, p_chunk in enumerate(purport_splits):
            chunks.append({
                "chunk_id": f"{verse.get('book_id')}_{verse.get('chapter')}_{verse.get('verse')}_purport_{i}",
                "text": f"Reference: {verse.get('title')}\n\nPurport (part {i+1}):\n{p_chunk}",
                "chunk_type": "purport",
                "metadata": {**base_meta, "purport_part": i}
            })

    return chunks


def chunk_book(book_id: str, book_name: str):
    raw_file = RAW_DATA_DIR / f"{book_id}.json"
    if not raw_file.exists():
        logger.error(f"Raw data not found for {book_id}. Run the scraper first.")
        return

    with open(raw_file, encoding="utf-8") as f:
        verses = json.load(f)

    all_chunks = []
    for verse in verses:
        all_chunks.extend(chunk_verse(verse, book_name))

    output_file = CHUNKED_DATA_DIR / f"{book_id}_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.success(f"[{book_id}] {len(verses)} verses → {len(all_chunks)} chunks → {output_file}")


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
            chunk_book(b["id"], b["name"])
    elif args.book:
        book = next((b for b in books if b["id"] == args.book), None)
        if book:
            chunk_book(book["id"], book["name"])
        else:
            print(f"Book '{args.book}' not found")
    else:
        parser.print_help()
