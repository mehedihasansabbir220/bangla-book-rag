"""FAISS vector index: build, persist, and load locally.

Converts processed chunks into LangChain Documents, embeds them with the
local E5 model, and stores the vectors under data/vectorstore/.

The chatbot should call `get_vectorstore()` / `load_index()` so an existing
index is reused and not rebuilt on every startup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import PROCESSED_DIR, VECTORSTORE_DIR
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

CHUNKS_FILENAME = "chunks.json"
INDEX_NAME = "index"

METADATA_FIELDS = ("book", "chapter", "section", "source_url", "chunk_id")


def processed_chunks_path() -> Path:
    return PROCESSED_DIR / CHUNKS_FILENAME


def resolve_index_dir(persist_dir: str | Path | None = None) -> Path:
    return Path(persist_dir) if persist_dir is not None else VECTORSTORE_DIR


def index_exists(persist_dir: str | Path | None = None) -> bool:
    """True if a previously saved FAISS index is on disk."""
    folder = resolve_index_dir(persist_dir)
    return (folder / f"{INDEX_NAME}.faiss").is_file() and (
        folder / f"{INDEX_NAME}.pkl"
    ).is_file()


def save_chunks(chunks: list[dict[str, Any]], path: str | Path | None = None) -> Path:
    """Write processed chunks to data/processed/chunks.json."""
    out = Path(path) if path is not None else processed_chunks_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s chunks to %s", len(chunks), out)
    return out


def load_chunks(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load processed chunks from data/processed/chunks.json."""
    src = Path(path) if path is not None else processed_chunks_path()
    if not src.is_file():
        raise FileNotFoundError(f"Processed chunks not found: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"No chunks in {src}")
    return data


def chunks_to_documents(chunks: list[dict[str, Any]]) -> list[Document]:
    """Convert chunk dicts into LangChain Documents with citation metadata."""
    documents: list[Document] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        metadata = {field: chunk.get(field) or "" for field in METADATA_FIELDS}
        documents.append(Document(page_content=text, metadata=metadata))
    if not documents:
        raise ValueError("No non-empty chunks to convert into Documents.")
    return documents


def build_index(
    chunks: list[dict[str, Any]],
    persist_dir: str | Path | None = None,
) -> FAISS:
    """Embed chunks with the local E5 model, build FAISS, and save it.

    Args:
        chunks: Chunk dicts with text plus citation metadata.
        persist_dir: Directory for index.faiss / index.pkl.

    Returns:
        The in-memory FAISS vector store.
    """
    documents = chunks_to_documents(chunks)
    embeddings = get_embeddings()
    logger.info("Embedding %s documents with the local model...", len(documents))
    vectorstore = FAISS.from_documents(documents, embeddings)

    folder = resolve_index_dir(persist_dir)
    folder.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(folder), index_name=INDEX_NAME)
    logger.info("Saved FAISS index to %s", folder)
    return vectorstore


def load_index(persist_dir: str | Path | None = None) -> FAISS:
    """Load a previously saved FAISS index from disk.

    Uses the same local embedding model so queries are encoded consistently.
    Does not re-embed the book.

    `allow_dangerous_deserialization` is required by LangChain for the local
    pickle sidecar; this is safe because we wrote the files ourselves.
    """
    folder = resolve_index_dir(persist_dir)
    if not index_exists(folder):
        raise FileNotFoundError(
            f"FAISS index not found in {folder}. Run: python scripts/build_index.py"
        )
    embeddings = get_embeddings()
    logger.info("Loading existing FAISS index from %s", folder)
    return FAISS.load_local(
        str(folder),
        embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )


def get_vectorstore(
    persist_dir: str | Path | None = None,
    chunks: list[dict[str, Any]] | None = None,
    *,
    rebuild: bool = False,
) -> FAISS:
    """Return the FAISS store.

    Default (`rebuild=False`) only loads an existing index. The chatbot and
    retriever must not embed the book on startup. Rebuild from
    `scripts/build_index.py` (or pass rebuild=True).
    """
    if rebuild:
        if chunks is None:
            chunks = load_chunks()
        return build_index(chunks, persist_dir)
    return load_index(persist_dir)
