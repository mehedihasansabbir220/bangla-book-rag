"""FAISS vector index build, persist, and load.

Responsibilities (to be implemented later):
- Take chunk texts + metadata + embeddings.
- Store them in a local FAISS index under data/vectorstore/.
- Reload the index at query time without re-crawling or re-embedding.
"""

from __future__ import annotations

from typing import Any


def build_index(chunks: list[dict[str, Any]], persist_dir: str | None = None) -> Any:
    """Create a FAISS index from prepared chunks and save it to disk.

    Args:
        chunks: Chunk dicts containing text and citation metadata.
        persist_dir: Directory under data/vectorstore/ for the saved index.

    Returns:
        The in-memory vector store / FAISS wrapper.
    """
    raise NotImplementedError("FAISS index building is not implemented yet.")


def load_index(persist_dir: str | None = None) -> Any:
    """Load a previously saved FAISS index from disk.

    Args:
        persist_dir: Directory containing the persisted index files.

    Returns:
        The loaded vector store / FAISS wrapper.
    """
    raise NotImplementedError("FAISS index loading is not implemented yet.")
