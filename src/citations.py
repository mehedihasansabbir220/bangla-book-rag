"""Citation formatting for RAG answers.

Every grounded answer cites book / chapter / section / source URL from
retrieved chunk metadata. Citations are never invented.
"""

from __future__ import annotations

from typing import Any


def format_citation(chunk_metadata: dict[str, Any]) -> str:
    """Format one chunk's metadata as a human-readable citation string."""
    book = (chunk_metadata.get("book") or "").strip()
    chapter = (chunk_metadata.get("chapter") or "").strip()
    section = (chunk_metadata.get("section") or "").strip()
    url = (chunk_metadata.get("source_url") or "").strip()

    parts = [part for part in (book, chapter, section) if part]
    label = ", ".join(parts) if parts else "নির্বাচিত বই"
    if url:
        return f"{label} — {url}"
    return label


def source_record(chunk: dict[str, Any]) -> dict[str, str]:
    """Structured citation fields required by the assignment."""
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return {
        "book": str(chunk.get("book") or meta.get("book") or ""),
        "chapter": str(chunk.get("chapter") or meta.get("chapter") or ""),
        "section": str(chunk.get("section") or meta.get("section") or ""),
        "source_url": str(chunk.get("source_url") or meta.get("source_url") or ""),
    }


def collect_source_records(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Unique source records in retrieval order (chapter + URL)."""
    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []
    for chunk in chunks:
        record = source_record(chunk)
        key = (record["chapter"], record["source_url"])
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def collect_citations(chunks: list[dict[str, Any]]) -> list[str]:
    """Deduplicated display citations in retrieval order."""
    return [format_citation(record) for record in collect_source_records(chunks)]
