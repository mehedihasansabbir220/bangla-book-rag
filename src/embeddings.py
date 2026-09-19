"""Local multilingual embeddings for Bengali text.

Responsibilities (to be implemented later):
- Load intfloat/multilingual-e5-small via Sentence Transformers (no paid API).
- Prefix passages with "passage: " and queries with "query: ".
- Encode texts into L2-normalized vectors for FAISS cosine / inner-product search.
"""

from __future__ import annotations

from typing import Any


def load_embedding_model(model_name: str | None = None) -> Any:
    """Load the local Sentence Transformers embedding model.

    Args:
        model_name: Hugging Face model id. Defaults to the configured E5 model.

    Returns:
        A loaded embedding model instance.
    """
    raise NotImplementedError("Embedding model loading is not implemented yet.")


def embed_passages(texts: list[str]) -> Any:
    """Embed document chunks for indexing.

    Args:
        texts: Chunk strings (without the E5 prefix; this function should add it).

    Returns:
        Matrix of embedding vectors.
    """
    raise NotImplementedError("Passage embedding is not implemented yet.")


def embed_query(query: str) -> Any:
    """Embed a user question for retrieval.

    Args:
        query: Natural-language question (Bengali or English).

    Returns:
        Single embedding vector.
    """
    raise NotImplementedError("Query embedding is not implemented yet.")
