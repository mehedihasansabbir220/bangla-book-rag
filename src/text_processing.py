"""Bengali text cleaning and word-based chunking.

Cleans crawled Wikisource prose without transliterating it, then splits each
chapter into overlapping word windows. Chapter/section/source metadata is
copied onto every chunk so later retrieval can cite the book location.

Chunk size and overlap are measured in **words**, not characters.
Defaults (from config): 500-word chunks with 100-word overlap.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from src.config import BOOK_TITLE, CHUNK_OVERLAP, CHUNK_SIZE

# Format chars that are safe to drop. Keep U+200C (ZWNJ) and U+200D (ZWJ):
# they are used in some Bengali conjunct spellings.
_DROP_CHARS = dict.fromkeys(
    map(
        ord,
        (
            "\ufeff",  # BOM
            "\u200b",  # zero-width space
            "\u2060",  # word joiner
            "\u00ad",  # soft hyphen
            "\u180e",  # Mongolian vowel separator
            "\ufffd",  # replacement character
        ),
    )
)

# Unicode space separators -> ASCII space. Newlines are handled separately.
_UNICODE_SPACES = re.compile(
    r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+"
)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_EDGE_NEWLINE_SPACE = re.compile(r"[ \t]*\n[ \t]*")


def clean_bengali_text(raw_text: str) -> str:
    """Clean and normalize Bengali prose extracted from Wikisource.

    - NFC-normalizes Unicode so Bengali code points are consistent.
    - Drops BOM / zero-width artifacts that are not part of the writing system.
    - Collapses whitespace while keeping paragraph breaks.
    - Does **not** transliterate, romanize, or strip Bengali letters/danda.
    """
    if not raw_text:
        return ""

    text = unicodedata.normalize("NFC", raw_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_DROP_CHARS)
    text = _UNICODE_SPACES.sub(" ", text)
    text = _EDGE_NEWLINE_SPACE.sub("\n", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _strip_repeated_heading(text: str, heading: str | None) -> str:
    """Drop a first line that only repeats the chapter/section title."""
    if not text or not heading:
        return text
    heading = unicodedata.normalize("NFC", heading).strip()
    lines = text.splitlines()
    if not lines:
        return text
    if unicodedata.normalize("NFC", lines[0]).strip() == heading:
        return "\n".join(lines[1:]).strip()
    return text


def tokenize_words(text: str) -> list[str]:
    """Split cleaned text into words on whitespace.

    Punctuation stays attached to the neighbouring token so danda, quotes,
    and commas are not discarded.
    """
    if not text:
        return []
    return text.split()


def validate_chunk_params(chunk_size: int, overlap: int) -> None:
    """Reject invalid window settings before any text is split."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )


def _window_starts(n_words: int, chunk_size: int, overlap: int) -> Iterable[int]:
    if n_words == 0:
        return
    step = chunk_size - overlap
    start = 0
    while start < n_words:
        yield start
        if start + chunk_size >= n_words:
            break
        start += step


def chunk_text(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *,
    page_index: int = 0,
) -> list[dict[str, Any]]:
    """Split one cleaned chapter into overlapping word chunks.

    Args:
        text: Cleaned chapter text.
        metadata: Fields such as book, chapter, section, source_url.
        chunk_size: Max words per chunk. Defaults to config CHUNK_SIZE.
        chunk_overlap: Overlapping words. Defaults to config CHUNK_OVERLAP.
        page_index: 1-based page number used in chunk_id.

    Returns:
        Chunk dicts with chunk_id, book, chapter, section, source_url, text.
    """
    size = CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    validate_chunk_params(size, overlap)

    meta = metadata or {}
    words = tokenize_words(text)
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    for local_idx, start in enumerate(_window_starts(len(words), size, overlap), start=1):
        window = words[start : start + size]
        chunks.append(
            {
                "chunk_id": f"{page_index:02d}-{local_idx:03d}",
                "book": meta.get("book") or BOOK_TITLE,
                "chapter": meta.get("chapter") or "",
                "section": meta.get("section") or "",
                "source_url": meta.get("source_url") or "",
                "text": " ".join(window),
            }
        )
    return chunks


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict[str, Any]]:
    """Clean every crawled page and chunk it without crossing chapter boundaries.

    Each input page is processed independently so a chunk never mixes two
    chapters. Overlap applies only inside a chapter.

    Args:
        pages: Records from data/raw/book_pages.json.
        chunk_size: Words per chunk. Defaults to config CHUNK_SIZE (500).
        overlap: Overlapping words. Defaults to config CHUNK_OVERLAP (100).

    Returns:
        Flat list of chunk dicts ready for embedding/indexing.

    Raises:
        ValueError: If overlap >= chunk_size, or either value is invalid.
    """
    size = CHUNK_SIZE if chunk_size is None else chunk_size
    ov = CHUNK_OVERLAP if overlap is None else overlap
    validate_chunk_params(size, ov)

    all_chunks: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        raw = page.get("text") or ""
        cleaned = clean_bengali_text(raw)
        cleaned = _strip_repeated_heading(cleaned, page.get("chapter"))
        cleaned = _strip_repeated_heading(cleaned, page.get("section"))
        metadata = {
            "book": page.get("book") or BOOK_TITLE,
            "chapter": page.get("chapter") or "",
            "section": page.get("section") or "",
            "source_url": page.get("source_url") or "",
        }
        page_chunks = chunk_text(
            cleaned,
            metadata=metadata,
            chunk_size=size,
            chunk_overlap=ov,
            page_index=page_index,
        )
        all_chunks.extend(page_chunks)
    return all_chunks
