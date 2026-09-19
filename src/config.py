"""Project configuration loaded from environment variables.

Copy `.env.example` to `.env` to override defaults. Values are read once at
import time via python-dotenv.

This project uses only local models:
- embeddings: Hugging Face Sentence Transformers
- LLM: Ollama

No paid cloud LLM API keys are used or required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths (not env-configurable; derived from the repository layout)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# E5 embedding prefixes are required by intfloat/multilingual-e5-small.
# They are model constraints, not user settings.
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "

# Load `.env` from the project root so scripts work from any working directory.
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    """Return a stripped environment string, or default if missing/blank."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    """Return an integer environment value, or default if missing/blank."""
    return int(_env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    """Return a float environment value, or default if missing/blank."""
    return float(_env_str(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for crawling, indexing, and RAG."""

    book_title: str
    book_url: str
    embedding_model: str
    embedding_device: str
    ollama_model: str
    ollama_base_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    generation_k: int
    retrieval_threshold: float

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from process env, falling back to assignment defaults."""
        return cls(
            book_title=_env_str("BOOK_TITLE", "রাজর্ষি"),
            book_url=_env_str(
                "BOOK_URL",
                "https://bn.wikisource.org/wiki/রাজর্ষি_(১৯৬১)",
            ),
            embedding_model=_env_str(
                "EMBEDDING_MODEL",
                "intfloat/multilingual-e5-small",
            ),
            embedding_device=_env_str("EMBEDDING_DEVICE", "cpu"),
            ollama_model=_env_str("OLLAMA_MODEL", "qwen2.5:3b"),
            ollama_base_url=_env_str(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ),
            chunk_size=_env_int("CHUNK_SIZE", 500),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
            top_k=_env_int("TOP_K", 5),
            generation_k=_env_int("GENERATION_K", 3),
            # FAISS IndexFlatL2: lower score = closer. 0.25 was too strict
            # (in-book L2 ≈ 0.22–0.30, out-of-book ≈ 0.48).
            retrieval_threshold=_env_float("RETRIEVAL_THRESHOLD", 0.40),
        )


settings = Settings.from_env()

# Module-level names matching the .env variables, for convenient imports:
#     from src.config import BOOK_URL, CHUNK_SIZE
BOOK_TITLE = settings.book_title
BOOK_URL = settings.book_url
EMBEDDING_MODEL = settings.embedding_model
EMBEDDING_DEVICE = settings.embedding_device
OLLAMA_MODEL = settings.ollama_model
OLLAMA_BASE_URL = settings.ollama_base_url
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
TOP_K = settings.top_k
GENERATION_K = settings.generation_k
RETRIEVAL_THRESHOLD = settings.retrieval_threshold
