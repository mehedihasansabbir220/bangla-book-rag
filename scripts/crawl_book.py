"""CLI entry point: crawl the selected Bengali Wikisource book.

Usage:
    python scripts/crawl_book.py

Writes chapter pages to data/raw/book_pages.json.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BOOK_TITLE, BOOK_URL, RAW_DIR  # noqa: E402
from src.crawler import crawl_book  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    print(f"Book: {BOOK_TITLE}")
    print(f"Start URL: {BOOK_URL}")
    print(f"Output: {RAW_DIR / 'book_pages.json'}")
    print("Crawling only this book's chapter/subpage hierarchy.\n")

    pages = crawl_book(BOOK_URL, str(RAW_DIR))
    total_chars = sum(page.get("char_count", 0) for page in pages)
    print("\nCrawl complete.")
    print(f"  Chapters/pages extracted: {len(pages)}")
    print(f"  Total characters: {total_chars}")
    print(f"  Saved to: {RAW_DIR / 'book_pages.json'}")
    if pages:
        print("  First chapter:", pages[0].get("chapter"))
        print("  Last chapter:", pages[-1].get("chapter"))


if __name__ == "__main__":
    main()
