"""LangChain retriever over the local FAISS index.

Responsibilities (to be implemented later):
- Wrap the FAISS store as a LangChain retriever.
- Return the top-k most similar chunks for a user query.
- Expose similarity scores so the RAG layer can reject weak matches.
"""

from __future__ import annotations

from typing import Any


def get_retriever(vectorstore: Any | None = None, k: int | None = None) -> Any:
    """Build a LangChain retriever from the FAISS vector store.

    Args:
        vectorstore: Loaded FAISS index. Loaded from disk if omitted.
        k: Number of chunks to retrieve. Uses config default if None.

    Returns:
        A LangChain retriever instance.
    """
    raise NotImplementedError("Retriever construction is not implemented yet.")


def retrieve(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve the most relevant book chunks for a query.

    Args:
        query: User question.
        k: Number of chunks to return.

    Returns:
        Ranked list of chunks with text, score, and citation metadata.
    """
    raise NotImplementedError("Chunk retrieval is not implemented yet.")
