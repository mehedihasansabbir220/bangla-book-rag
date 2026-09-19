"""Bengali text cleaning, preprocessing, and chunking.

Responsibilities (to be implemented later):
- Strip Wikisource navigation, templates, footnotes clutter, and boilerplate.
- Normalize Bengali Unicode (NFC) and whitespace.
- Split the book into chunks with configurable size and overlap.
- Preserve chapter / section / source metadata on every chunk.
"""

from __future__ import annotations

from typing import Any


def clean_bengali_text(raw_text: str) -> str:
    """Clean and normalize Bengali prose extracted from Wikisource.

    Args:
        raw_text: Unprocessed text from a crawled page.

    Returns:
        Cleaned Bengali text suitable for chunking and embedding.
    """
    raise NotImplementedError("Bengali text cleaning is not implemented yet.")


def chunk_text(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks, keeping source metadata.

    Args:
        text: Cleaned chapter text.
        metadata: Chapter/section/source fields to copy onto each chunk.
        chunk_size: Maximum chunk length (characters). Uses config default if None.
        chunk_overlap: Overlap between consecutive chunks. Uses config default if None.

    Returns:
        List of chunk dicts with text and citation metadata.
    """
    raise NotImplementedError("Text chunking is not implemented yet.")
