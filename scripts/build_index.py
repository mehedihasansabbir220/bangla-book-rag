"""CLI entry point: clean, chunk, embed, and build the FAISS index.

Run later with:
    python scripts/build_index.py

Pipeline (to be implemented):
    data/raw/ -> text_processing -> embeddings -> indexer -> data/vectorstore/
"""

from __future__ import annotations


def main() -> None:
    """Build the local FAISS index (not implemented yet)."""
    raise NotImplementedError("build_index.py is a placeholder; indexing is not implemented yet.")


if __name__ == "__main__":
    main()
