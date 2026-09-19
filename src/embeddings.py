"""Local multilingual embeddings for Bengali RAG.

Uses LangChain's HuggingFaceEmbeddings with intfloat/multilingual-e5-small.
The model is downloaded once from Hugging Face and then runs entirely on
the local CPU. No OpenAI, Gemini, or other embedding API key is used.

E5 requires asymmetric prefixes:
- documents / chunks: "passage: "
- user questions:     "query: "
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_PASSAGE_PREFIX,
    EMBEDDING_QUERY_PREFIX,
)

_embeddings: Embeddings | None = None


def apply_e5_prefix(text: str, prefix: str) -> str:
    """Add an E5 prefix unless the text already has query:/passage:."""
    stripped = (text or "").strip()
    if stripped.startswith("query:") or stripped.startswith("passage:"):
        return stripped
    return f"{prefix}{stripped}"


class PrefixedE5Embeddings(Embeddings):
    """LangChain embeddings wrapper that applies E5 passage/query prefixes.

    FAISS and retrievers can use this object directly later. Prefixing is
    done here so callers can pass raw chunk/question strings.
    """

    def __init__(
        self,
        inner: HuggingFaceEmbeddings,
        *,
        query_prefix: str = EMBEDDING_QUERY_PREFIX,
        passage_prefix: str = EMBEDDING_PASSAGE_PREFIX,
    ) -> None:
        self._inner = inner
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [apply_e5_prefix(text, self.passage_prefix) for text in texts]
        return self._inner.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(apply_e5_prefix(text, self.query_prefix))


def get_embeddings() -> Embeddings:
    """Return a cached local E5 embedding model (CPU by default).

    The first call loads Sentence Transformers weights from the Hugging Face
    cache (or downloads them once). Later calls reuse the same instance.
    """
    global _embeddings
    if _embeddings is None:
        inner = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={"normalize_embeddings": True},
        )
        _embeddings = PrefixedE5Embeddings(inner)
    return _embeddings


def load_embedding_model(model_name: str | None = None) -> Embeddings:
    """Load (or reuse) the local embedding model.

    Args:
        model_name: Optional override. If it differs from the cached model,
            a new instance is created. Defaults to config EMBEDDING_MODEL.
    """
    if model_name is None or model_name == EMBEDDING_MODEL:
        return get_embeddings()
    inner = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
    )
    return PrefixedE5Embeddings(inner)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing (adds the passage: prefix)."""
    return get_embeddings().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """Embed a user question for retrieval (adds the query: prefix)."""
    return get_embeddings().embed_query(query)
