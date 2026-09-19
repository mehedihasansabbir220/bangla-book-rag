"""Wikisource crawler for the selected Bengali prose book.

Starts at BOOK_URL, discovers chapter/subpage URLs that belong to that book's
hierarchy, downloads each page politely, extracts article text, and writes
`data/raw/book_pages.json`.

Does not crawl the rest of Wikisource. Embeddings and FAISS are out of scope.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import BOOK_TITLE, BOOK_URL, RAW_DIR

logger = logging.getLogger(__name__)

USER_AGENT = (
    "BanglaBookRAG/1.0 (university assignment; "
    "contact: mohammadmehedihasansabbir@gmail.com; "
    "local Wikisource book crawler)"
)
CRAWL_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
OUTPUT_FILENAME = "book_pages.json"

# Namespaces / page types that are never book prose.
_SKIP_TITLE_PREFIXES = (
    "বিশেষ:",
    "Special:",
    "পাতা:",
    "Page:",
    "নির্ঘণ্ট:",
    "Index:",
    "চিত্র:",
    "File:",
    "Image:",
    "Media:",
    "টেমপ্লেট:",
    "Template:",
    "বিষয়শ্রেণী:",
    "Category:",
    "সাহায্য:",
    "Help:",
    "উইকিসংকলন:",
    "Wikisource:",
    "লেখক:",
    "Author:",
    "রচনা:",
    "আলাপ:",
    "Talk:",
    "ব্যবহারকারী:",
    "User:",
    "মডিউল:",
    "Module:",
    "মিডিয়াউইকি:",
    "MediaWiki:",
    "প্রবেশদ্বার:",
    "Portal:",
)

_UNWANTED_SELECTORS = (
    "script",
    "style",
    "noscript",
    "iframe",
    ".mw-editsection",
    ".mw-editsection-like",
    "sup.reference",
    "ol.references",
    "div.references",
    "#references",
    "nav",
    ".navbox",
    ".navbar",
    "#toc",
    ".toc",
    ".mw-indicators",
    ".ws-noexport",
    ".headertemplate",
    ".footertemplate",
    ".wikisource-header-template",
    ".noprint",
    ".printfooter",
    ".mw-empty-elt",
    "table.pr_quality",
    ".pr_quality",
    "span.pagenum",
    ".ws-pagenum",
    "figure",
    ".thumb",
    ".mw-file-description",
)

_session: requests.Session | None = None
_last_request_at = 0.0


def _session_get() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "bn,en;q=0.8",
            }
        )
    return _session


def _polite_wait() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    remaining = CRAWL_DELAY_SECONDS - elapsed
    if remaining > 0:
        time.sleep(remaining)
    _last_request_at = time.monotonic()


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def wiki_title_from_url(url: str) -> str | None:
    """Return the MediaWiki page title encoded in a Wikisource URL, or None."""
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    if path.startswith("/wiki/"):
        title = path[len("/wiki/") :]
    else:
        query = parse_qs(parsed.query)
        raw_title = query.get("title", [None])[0]
        if not raw_title:
            return None
        title = unquote(raw_title)
    title = title.replace("_", " ").strip()
    if not title:
        return None
    return _nfc(title)


def canonicalize_wiki_url(url: str, base_url: str) -> str:
    """Resolve relative hrefs and strip fragments / tracking query noise."""
    absolute = urljoin(base_url, url.strip())
    parsed = urlparse(absolute)
    # Decode HTML entities that sometimes appear in hrefs.
    path = unquote(parsed.path.replace("&#58;", ":").replace("&#95;", "_"))
    cleaned = parsed._replace(fragment="", query="", path=path)
    return urlunparse(cleaned)


def _host_allowed(url: str, book_url: str) -> bool:
    book_host = urlparse(book_url).netloc.lower()
    host = urlparse(url).netloc.lower().lstrip("www.")
    book_host = book_host.lstrip("www.")
    return bool(host) and host == book_host


def skip_reason(title: str | None, url: str = "") -> str | None:
    """Return a short reason if this URL/title must not be crawled."""
    lowered = url.lower()
    if "action=edit" in lowered or "redlink=1" in lowered:
        return "edit/redlink"
    if title is None:
        return "not a wiki title"
    if title.startswith("বিশেষ:") or title.lower().startswith("special:"):
        return "special page"
    for prefix in _SKIP_TITLE_PREFIXES:
        if title.startswith(prefix) or title.startswith(prefix.replace(" ", "_")):
            return f"namespace {prefix.rstrip(':')}"
    return None


def belongs_to_book(title: str, book_root: str) -> bool:
    """True iff title is the book root or a subpage of that root."""
    root = _nfc(book_root)
    candidate = _nfc(title)
    return candidate == root or candidate.startswith(root + "/")


def fetch_page(url: str, *, allow_http_error: bool = False) -> tuple[str, str, int]:
    """Download a single Wikisource page as HTML.

    Args:
        url: Absolute URL of a book page.
        allow_http_error: If True, return body even for 4xx/5xx after retries.

    Returns:
        Tuple of (final_url, html, status_code).
    """
    session = _session_get()
    last_exc: Exception | None = None
    response: requests.Response | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        _polite_wait()
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                logger.warning(
                    "HTTP %s for %s (attempt %s/%s); retrying",
                    response.status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(CRAWL_DELAY_SECONDS * attempt)
                continue
            if not allow_http_error:
                response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.url, response.text, response.status_code
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Request error for %s (attempt %s/%s): %s",
                url,
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(CRAWL_DELAY_SECONDS * attempt)

    if allow_http_error and response is not None:
        response.encoding = response.apparent_encoding or "utf-8"
        return response.url, response.text, response.status_code
    raise RuntimeError(f"Failed to download {url}") from last_exc


def _is_missing_page(html: str) -> bool:
    soup = BeautifulSoup(html, "lxml")
    return soup.select_one(".noarticletext") is not None


def _content_soup(html: str) -> BeautifulSoup | None:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one("#mw-content-text")
    return root


def _prefix_index_url(base_url: str, prefix: str) -> str:
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    # English special-page name redirects to the Bengali equivalent.
    return (
        f"{origin}/w/index.php?title=Special:PrefixIndex"
        f"&namespace=0&prefix={quote(prefix)}"
    )


def _titles_from_prefix_index(book_url: str, prefix: str) -> list[str]:
    """List main-namespace titles that start with `prefix` via Special:PrefixIndex."""
    titles: list[str] = []
    seen: set[str] = set()
    next_url: str | None = _prefix_index_url(book_url, prefix)

    while next_url:
        final_url, html, status = fetch_page(next_url, allow_http_error=True)
        if status >= 400:
            logger.warning("PrefixIndex HTTP %s for prefix %r", status, prefix)
            break
        soup = BeautifulSoup(html, "lxml")
        for link in soup.select("ul.mw-prefixindex-list li a[href]"):
            href = link.get("href")
            if not href:
                continue
            abs_url = canonicalize_wiki_url(str(href), final_url)
            title = wiki_title_from_url(abs_url)
            reason = skip_reason(title, abs_url)
            if reason or title is None:
                continue
            if title not in seen:
                seen.add(title)
                titles.append(title)

        next_link = soup.select_one("a.mw-nextlink, a[rel='next']")
        if next_link and next_link.get("href"):
            candidate = canonicalize_wiki_url(str(next_link.get("href")), final_url)
            # Keep query string for PrefixIndex pagination.
            candidate = urljoin(final_url, str(next_link.get("href")))
            if candidate == next_url:
                break
            next_url = candidate
        else:
            next_url = None

    return titles


def _choose_book_root(configured_title: str, prefix_titles: list[str]) -> str:
    """Pick the Wikisource title that is the root of this book's subpages."""
    configured = _nfc(configured_title)
    children = [t for t in prefix_titles if "/" in t]
    roots = [t for t in prefix_titles if "/" not in t]

    direct_children = [t for t in children if t.startswith(configured + "/")]
    if direct_children or configured in roots:
        return configured

    # e.g. configured "রাজর্ষি" but pages live under "রাজর্ষি (১৯৬১)".
    matching_roots = [r for r in roots if r == configured or r.startswith(configured + " ")]
    if not matching_roots:
        matching_roots = roots
    if not matching_roots:
        return configured

    def child_count(root: str) -> int:
        return sum(1 for t in children if t.startswith(root + "/"))

    matching_roots.sort(key=lambda r: (-child_count(r), len(r), r))
    chosen = matching_roots[0]
    logger.info(
        "Configured title %r has no subpages; using book root %r (%s chapters)",
        configured,
        chosen,
        child_count(chosen),
    )
    return chosen


def _toc_chapter_titles(html: str, page_url: str, book_root: str) -> tuple[list[str], int]:
    """Chapter titles in TOC/document order, plus a count of skipped in-content links."""
    content = _content_soup(html)
    if content is None:
        return [], 0

    ordered: list[str] = []
    seen: set[str] = set()
    skipped = 0

    for link in content.select("a[href]"):
        href = str(link.get("href") or "")
        abs_url = canonicalize_wiki_url(href, page_url)
        if not _host_allowed(abs_url, page_url):
            skipped += 1
            logger.info("Skipped URL (off-site): %s", abs_url)
            continue
        title = wiki_title_from_url(abs_url)
        reason = skip_reason(title, href)
        if reason:
            skipped += 1
            logger.info("Skipped URL (%s): %s", reason, abs_url)
            continue
        if title is None or not belongs_to_book(title, book_root):
            skipped += 1
            logger.info("Skipped URL (outside book hierarchy): %s", abs_url)
            continue
        if title == book_root:
            continue
        if title not in seen:
            seen.add(title)
            ordered.append(title)

    return ordered, skipped


def discover_chapter_urls(book_url: str) -> list[str]:
    """Return chapter/subpage URLs belonging to the selected book.

    Discovery order:
    1. Fetch BOOK_URL (follows redirects; 404 is allowed).
    2. List PrefixIndex titles for the URL title / BOOK_TITLE.
    3. Resolve the real book root (handles missing `রাজর্ষি` → `রাজর্ষি (১৯৬১)`).
    4. Take TOC links from the book root in reading order.
    5. Union with PrefixIndex subpages so no chapter is missed.
    """
    configured_title = wiki_title_from_url(book_url) or BOOK_TITLE
    logger.info("Starting discovery from %s (title %r)", book_url, configured_title)

    final_url, html, status = fetch_page(book_url, allow_http_error=True)
    if status >= 400 or _is_missing_page(html):
        logger.info(
            "BOOK_URL is missing (HTTP %s). Resolving the book via PrefixIndex.",
            status,
        )
        prefix_titles = _titles_from_prefix_index(book_url, configured_title)
    else:
        prefix_titles = _titles_from_prefix_index(final_url, configured_title)
        redirected_title = wiki_title_from_url(final_url)
        if redirected_title and redirected_title != configured_title:
            logger.info("Followed redirect %r → %r", configured_title, redirected_title)
            extra = _titles_from_prefix_index(final_url, redirected_title)
            prefix_titles = list(OrderedDict.fromkeys(prefix_titles + extra))

    if not prefix_titles:
        logger.warning("PrefixIndex returned no titles for %r", configured_title)

    book_root = _choose_book_root(configured_title, prefix_titles)
    origin = f"{urlparse(book_url).scheme}://{urlparse(book_url).netloc}"
    root_url = f"{origin}/wiki/{book_root.replace(' ', '_')}"

    root_final, root_html, root_status = fetch_page(root_url, allow_http_error=True)
    toc_titles: list[str] = []
    skipped = 0
    if root_status < 400 and not _is_missing_page(root_html):
        toc_titles, skipped = _toc_chapter_titles(root_html, root_final, book_root)
        logger.info("TOC on book root yielded %s chapter links", len(toc_titles))
    else:
        logger.warning("Could not load book root %s (HTTP %s)", root_url, root_status)

    prefix_children = [
        t for t in prefix_titles if belongs_to_book(t, book_root) and t != book_root
    ]
    # Also list PrefixIndex under the resolved root, in case the configured
    # prefix was shorter than the edition title.
    if book_root != configured_title:
        prefix_children = list(
            OrderedDict.fromkeys(
                prefix_children + [
                    t
                    for t in _titles_from_prefix_index(book_url, book_root + "/")
                    if belongs_to_book(t, book_root) and t != book_root
                ]
            )
        )

    ordered = list(OrderedDict.fromkeys(toc_titles + prefix_children))
    urls = [f"{origin}/wiki/{title.replace(' ', '_')}" for title in ordered]

    logger.info("Book root: %s", book_root)
    logger.info("Discovered URLs: %s", len(urls))
    for discovered in urls:
        logger.info("  %s", discovered)
    logger.info("Skipped URLs: %s", skipped)

    if not urls:
        raise RuntimeError(
            f"No chapter/subpage URLs found for {book_url}. "
            "Refusing to ingest only the main page."
        )
    return urls


def extract_main_text(html: str) -> str:
    """Extract readable article text from a Wikisource HTML page."""
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        content = soup.select_one("#mw-content-text")
    if content is None or content.select_one(".noarticletext"):
        return ""

    for selector in _UNWANTED_SELECTORS:
        for tag in content.select(selector):
            tag.decompose()

    # Novels do not need TOC / infobox / layout tables.
    for table in content.find_all("table"):
        table.decompose()

    raw = content.get_text("\n", strip=True)
    raw = _nfc(raw)
    raw = raw.replace("\ufeff", "").replace("\u200b", "")
    lines: list[str] = []
    for line in raw.splitlines():
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _split_chapter_section(full_title: str, book_root: str) -> tuple[str, str]:
    if full_title == book_root:
        return book_root, ""
    rest = full_title[len(book_root) :].lstrip("/") if full_title.startswith(book_root) else full_title
    parts = [part for part in rest.split("/") if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], "/".join(parts[1:])


def _page_record(book: str, title: str, book_root: str, source_url: str, text: str) -> dict[str, Any]:
    chapter, section = _split_chapter_section(title, book_root)
    return {
        "book": book,
        "chapter": chapter,
        "section": section,
        "source_url": source_url,
        "wikisource_title": title,
        "text": text,
        "char_count": len(text),
    }


def crawl_book(book_url: str | None = None, output_dir: str | None = None) -> list[dict[str, Any]]:
    """Crawl the complete book and save `data/raw/book_pages.json`.

    Args:
        book_url: Wikisource index URL. Defaults to config BOOK_URL.
        output_dir: Directory for the JSON file. Defaults to data/raw/.

    Returns:
        One dict per successfully extracted chapter/page.
    """
    start_url = book_url or BOOK_URL
    out_dir = RAW_DIR if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chapter_urls = discover_chapter_urls(start_url)
    configured_title = wiki_title_from_url(start_url) or BOOK_TITLE
    # Infer book_root from the first chapter URL (parent of the first subpage).
    sample_title = wiki_title_from_url(chapter_urls[0]) or ""
    book_root = sample_title.split("/")[0] if sample_title else configured_title

    pages: list[dict[str, Any]] = []
    downloaded = 0
    failed = 0

    for url in tqdm(chapter_urls, desc="Downloading chapters", unit="page"):
        try:
            final_url, html, status = fetch_page(url, allow_http_error=True)
        except Exception as exc:  # noqa: BLE001 - crawl should continue
            failed += 1
            logger.error("Failed to download %s: %s", url, exc)
            continue

        downloaded += 1
        if status >= 400 or _is_missing_page(html):
            failed += 1
            logger.error("HTTP %s / missing page, skipping %s", status, url)
            continue

        title = wiki_title_from_url(final_url) or wiki_title_from_url(url) or ""
        if not belongs_to_book(title, book_root):
            logger.info("Skipped URL (outside book hierarchy after redirect): %s", final_url)
            continue

        text = extract_main_text(html)
        if not text:
            failed += 1
            logger.warning("No article text extracted, skipping %s", final_url)
            continue

        pages.append(
            _page_record(
                book=BOOK_TITLE,
                title=title,
                book_root=book_root,
                source_url=final_url,
                text=text,
            )
        )

    output_path = out_dir / OUTPUT_FILENAME
    output_path.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Downloaded pages: %s", downloaded)
    logger.info("Failed / empty pages: %s", failed)
    logger.info("Number of pages successfully extracted: %s", len(pages))
    logger.info("Wrote %s", output_path)

    if not pages:
        raise RuntimeError("Crawler finished but extracted 0 pages.")
    return pages
