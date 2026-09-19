"""CLI: raw pages → preprocess/chunk → embed → FAISS.

Usage:
    python scripts/build_index.py

Reads data/raw/book_pages.json and writes:
    data/processed/chunks.json
    data/vectorstore/index.faiss
    data/vectorstore/index.pkl
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    EMBEDDING_MODEL,
    RAW_DIR,
    VECTORSTORE_DIR,
)
from src.indexer import build_index, processed_chunks_path, save_chunks  # noqa: E402
from src.text_processing import chunk_pages  # noqa: E402

RAW_PAGES_PATH = RAW_DIR / "book_pages.json"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if not RAW_PAGES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {RAW_PAGES_PATH}. Run: python scripts/crawl_book.py"
        )

    pages = json.loads(RAW_PAGES_PATH.read_text(encoding="utf-8"))
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"No pages found in {RAW_PAGES_PATH}")

    print("Building FAISS index")
    print(f"  Raw pages:        {RAW_PAGES_PATH}")
    print(f"  Embedding model:  {EMBEDDING_MODEL}")
    print(f"  Vector store:     {VECTORSTORE_DIR}")
    print()

    chunks = chunk_pages(pages)
    chunks_path = save_chunks(chunks)

    build_index(chunks, VECTORSTORE_DIR)

    print("Index build complete.")
    print(f"  number of pages:       {len(pages)}")
    print(f"  number of chunks:      {len(chunks)}")
    print(f"  embedding model:       {EMBEDDING_MODEL}")
    print(f"  vector store location: {VECTORSTORE_DIR}")
    print(f"  processed chunks:      {chunks_path}")


if __name__ == "__main__":
    main()
