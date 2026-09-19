"""Citation formatting for RAG answers.

Responsibilities (to be implemented later):
- Turn chunk metadata into human-readable chapter / section / source citations.
- Deduplicate overlapping sources from multiple retrieved chunks.
- Produce a consistent citation block for the Streamlit UI and evaluation output.

Every answer must cite the book location it used. If no usable source exists,
the RAG layer should refuse to answer rather than invent a citation.
"""

from __future__ import annotations

from typing import Any


def format_citation(chunk_metadata: dict[str, Any]) -> str:
    """Format a single chunk's metadata as a citation string.

    Args:
        chunk_metadata: Fields such as book title, khondo (volume),
            poricched (chapter), section, and source URL.

    Returns:
        A display-ready citation, e.g. "কপালকুণ্ডলা, প্রথম খণ্ড, প্রথম পরিচ্ছেদ".
    """
    raise NotImplementedError("Citation formatting is not implemented yet.")


def collect_citations(chunks: list[dict[str, Any]]) -> list[str]:
    """Collect unique citations from a list of retrieved chunks.

    Args:
        chunks: Retrieved chunks with metadata.

    Returns:
        Deduplicated list of citation strings, in retrieval order.
    """
    raise NotImplementedError("Citation collection is not implemented yet.")
