"""Wikisource crawler for the selected Bengali prose book.

Responsibilities (to be implemented later):
- Fetch the book index page from Bengali Wikisource.
- Discover chapter / section / subpage URLs.
- Download each page with requests + BeautifulSoup.
- Persist raw HTML (and extracted text) under data/raw/.

Must be polite: set a User-Agent and delay between requests.
Must not crawl pages outside the selected book.
"""

from __future__ import annotations

from typing import Any


def discover_chapter_urls(book_url: str) -> list[str]:
    """Return chapter/subpage URLs belonging to the selected book.

    Args:
        book_url: Wikisource index URL of the book.

    Returns:
        Ordered list of chapter page URLs.
    """
    raise NotImplementedError("Wikisource URL discovery is not implemented yet.")


def fetch_page(url: str) -> str:
    """Download a single Wikisource page as HTML.

    Args:
        url: Absolute URL of a book chapter or subpage.

    Returns:
        Raw HTML string.
    """
    raise NotImplementedError("Page fetching is not implemented yet.")


def crawl_book(book_url: str, output_dir: str | None = None) -> list[dict[str, Any]]:
    """Crawl the complete book and save raw pages.

    Args:
        book_url: Wikisource index URL of the book.
        output_dir: Directory under data/raw/ to write crawled files.

    Returns:
        List of records with url, title, html/text, and chapter metadata.
    """
    raise NotImplementedError("Book crawling is not implemented yet.")
