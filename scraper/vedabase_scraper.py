"""
Vedabase.io Scraper
Asynchronously scrapes all scripture pages from vedabase.io.
Outputs structured JSON files per book into data/raw/.

Usage:
    python scraper/vedabase_scraper.py --book bg
    python scraper/vedabase_scraper.py --all
"""

import asyncio
import json
import os
import re
import time
import argparse
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeout
from loguru import logger

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_DELAY = 1.5  # seconds between requests (be polite)
MAX_RETRIES = 3


async def get_verse_urls_for_chapter(page: Page, chapter_url: str) -> list[str]:
    """Extract all verse URLs from a chapter index page."""
    try:
        await page.goto(chapter_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_selector("a[href]", timeout=10000)
        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(el => el.href)"
        )
        # Filter to verse links (pattern: /en/library/bg/1/1/ etc.)
        verse_links = [
            l for l in links
            if re.search(r"/en/library/[a-z]+/[\w-]+/[\w-]+/?$", l)
            and l != chapter_url
        ]
        return list(dict.fromkeys(verse_links))  # deduplicate preserving order
    except PlaywrightTimeout:
        logger.warning(f"Timeout getting verse URLs from {chapter_url}")
        return []


async def scrape_verse_page(page: Page, url: str) -> Optional[dict]:
    """Scrape a single verse page and return structured data."""
    for attempt in range(MAX_RETRIES):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_selector(".r-verse-text, .r-synonyms, .r-translation", timeout=15000)

            data = await page.evaluate("""
                () => {
                    const getText = (sel) => {
                        const el = document.querySelector(sel);
                        return el ? el.innerText.trim() : null;
                    };
                    const getAll = (sel) => {
                        return [...document.querySelectorAll(sel)].map(el => el.innerText.trim());
                    };

                    // Breadcrumb for book/chapter/verse metadata
                    const breadcrumbs = getAll(".breadcrumb li, nav[aria-label='breadcrumb'] li");

                    return {
                        url: window.location.href,
                        title: getText("h1, .r-title"),
                        devanagari: getText(".r-verse-text.devanagari, .devanagari"),
                        transliteration: getText(".r-verse-text.transliteration, .transliteration"),
                        synonyms: getText(".r-synonyms"),
                        translation: getText(".r-translation"),
                        purport: getText(".r-purport, .purport"),
                        breadcrumbs: breadcrumbs
                    };
                }
            """)

            if data and data.get("translation"):
                # Parse book/chapter/verse from URL
                parts = url.rstrip("/").split("/")
                data["book_id"] = parts[-3] if len(parts) >= 3 else None
                data["chapter"] = parts[-2] if len(parts) >= 2 else None
                data["verse"] = parts[-1] if len(parts) >= 1 else None
                return data
            else:
                logger.warning(f"No translation found at {url}, attempt {attempt+1}")

        except PlaywrightTimeout:
            logger.warning(f"Timeout on {url}, attempt {attempt+1}/{MAX_RETRIES}")
            await asyncio.sleep(2 ** attempt)

    return None


async def scrape_book(book: dict, headless: bool = True):
    """Scrape an entire book from vedabase.io."""
    book_id = book["id"]
    output_file = RAW_DATA_DIR / f"{book_id}.json"

    if output_file.exists():
        logger.info(f"[{book_id}] Already scraped, skipping. Delete {output_file} to re-scrape.")
        return

    logger.info(f"[{book_id}] Starting scrape of {book['name']}")
    all_verses = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (compatible; JnanaLabsBot/1.0; +https://github.com/JnanaLabs/vedic-rag)"
        )
        page = await context.new_page()

        # Get chapter-level URLs from book index
        base_url = book["base_url"]
        await page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(1)

        chapter_links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(el => el.href)"
        )
        chapter_links = [
            l for l in chapter_links
            if re.search(rf"/en/library/{book_id}/[\w-]+/?$", l)
        ]
        chapter_links = list(dict.fromkeys(chapter_links))
        logger.info(f"[{book_id}] Found {len(chapter_links)} chapters")

        for ch_url in chapter_links:
            verse_urls = await get_verse_urls_for_chapter(page, ch_url)
            logger.info(f"[{book_id}] Chapter {ch_url} — {len(verse_urls)} verses")

            for v_url in verse_urls:
                verse_data = await scrape_verse_page(page, v_url)
                if verse_data:
                    all_verses.append(verse_data)
                    logger.debug(f"  Scraped: {v_url}")
                await asyncio.sleep(BASE_DELAY)

        await browser.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_verses, f, ensure_ascii=False, indent=2)

    logger.success(f"[{book_id}] Done. {len(all_verses)} verses saved to {output_file}")


async def scrape_all():
    with open("scraper/books_index.json") as f:
        index = json.load(f)
    for book in index["books"]:
        await scrape_book(book)
        await asyncio.sleep(3)  # courtesy delay between books


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vedabase.io scraper")
    parser.add_argument("--book", type=str, help="Book ID to scrape (e.g. bg, sb, cc)")
    parser.add_argument("--all", action="store_true", help="Scrape all books")
    parser.add_argument("--visible", action="store_true", help="Run with visible browser")
    args = parser.parse_args()

    if args.all:
        asyncio.run(scrape_all())
    elif args.book:
        with open("scraper/books_index.json") as f:
            index = json.load(f)
        book = next((b for b in index["books"] if b["id"] == args.book), None)
        if book:
            asyncio.run(scrape_book(book, headless=not args.visible))
        else:
            print(f"Book '{args.book}' not found in books_index.json")
    else:
        parser.print_help()
