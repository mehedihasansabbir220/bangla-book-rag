"""Retrieval over the persisted FAISS index.

Loads `data/vectorstore/` (does not rebuild). Encodes the user question with
the local E5 model, which adds the `query:` prefix inside get_embeddings().

This project's index is LangChain's default **IndexFlatL2**
(`DistanceStrategy.EUCLIDEAN_DISTANCE`). `similarity_search_with_score`
therefore returns **L2 distance: lower score = more similar**.

Do not treat RETRIEVAL_THRESHOLD as cosine similarity. The live cutoff is a
maximum L2 distance (default 0.40 after measuring in-book vs out-of-book
scores). A hit passes when score <= threshold.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import RETRIEVAL_THRESHOLD, TOP_K
from src.indexer import get_vectorstore

logger = logging.getLogger(__name__)

# Inspected from the live index (LangChain FAISS.from_documents default):
#   index type          = faiss.IndexFlatL2
#   distance_strategy   = DistanceStrategy.EUCLIDEAN_DISTANCE
#   stored vector L2    = 1.0  (E5 normalize_embeddings=True)
SCORE_METRIC = "l2_distance"
HIGHER_SCORE_IS_BETTER = False

_vectorstore: FAISS | None = None


def _get_store(vectorstore: FAISS | None = None) -> FAISS:
    global _vectorstore
    if vectorstore is not None:
        return vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(rebuild=False)
    return _vectorstore


def l2_to_cosine(l2_distance: float) -> float:
    """Convert L2 distance to cosine similarity for unit-normalized vectors.

    For ||u||=||v||=1:  L2^2 = 2(1 - cosine)  ⇒  cosine = 1 - L2^2 / 2.
    This is a convenience for tuning, not the raw FAISS score.
    """
    cosine = 1.0 - (float(l2_distance) ** 2) / 2.0
    return max(-1.0, min(1.0, cosine))


def passes_threshold(
    score: float,
    score_threshold: float | None = None,
) -> bool:
    """True if a raw FAISS L2 score is at least as similar as the threshold.

    Because the metric is L2 distance, a hit passes when score <= threshold.
    """
    limit = RETRIEVAL_THRESHOLD if score_threshold is None else score_threshold
    return float(score) <= float(limit)


def get_retriever(vectorstore: FAISS | None = None, k: int | None = None) -> Any:
    """LangChain retriever over the persisted FAISS store (no rebuild)."""
    store = _get_store(vectorstore)
    return store.as_retriever(search_kwargs={"k": TOP_K if k is None else k})


def _hit_from_pair(document: Document, score: float, score_threshold: float) -> dict[str, Any]:
    metadata = dict(document.metadata or {})
    return {
        "document": document,
        "text": document.page_content,
        "score": float(score),
        "score_metric": SCORE_METRIC,
        "higher_is_better": HIGHER_SCORE_IS_BETTER,
        "cosine_similarity": l2_to_cosine(score),
        "passes_threshold": passes_threshold(score, score_threshold),
        "metadata": metadata,
        "book": metadata.get("book", ""),
        "chapter": metadata.get("chapter", ""),
        "section": metadata.get("section", ""),
        "source_url": metadata.get("source_url", ""),
        "chunk_id": metadata.get("chunk_id", ""),
    }


def retrieve_documents(
    question: str,
    k: int | None = None,
    score_threshold: float | None = None,
    *,
    vectorstore: FAISS | None = None,
    only_passing: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve TOP_K book chunks for a Bengali (or English) question.

    Args:
        question: User question. The E5 `query:` prefix is applied by the
            embedding wrapper, not by this function.
        k: How many neighbours to fetch. Defaults to config TOP_K.
        score_threshold: Maximum L2 distance to count as relevant. Defaults
            to config RETRIEVAL_THRESHOLD (measured default 0.40).
        vectorstore: Optional preloaded FAISS store.
        only_passing: If True, drop hits whose L2 score is worse than the
            threshold. Default False so scores can be inspected for tuning.

    Returns:
        Ranked list of dicts with `document`, raw `score` (L2 distance),
        `cosine_similarity` (derived), `passes_threshold`, and metadata.
    """
    query = (question or "").strip()
    if not query:
        return []

    top_k = TOP_K if k is None else k
    threshold = RETRIEVAL_THRESHOLD if score_threshold is None else score_threshold
    store = _get_store(vectorstore)

    pairs = store.similarity_search_with_score(query, k=top_k)
    hits = [_hit_from_pair(doc, score, threshold) for doc, score in pairs]

    logger.info(
        "Retrieved %s hits for %r (metric=%s, lower_is_better, threshold=%s)",
        len(hits),
        query[:80],
        SCORE_METRIC,
        threshold,
    )
    for hit in hits:
        logger.info(
            "  L2=%.4f cosine=%.4f pass=%s %s %s",
            hit["score"],
            hit["cosine_similarity"],
            hit["passes_threshold"],
            hit["chunk_id"],
            hit["chapter"],
        )

    if only_passing:
        hits = [hit for hit in hits if hit["passes_threshold"]]
        if not hits:
            logger.info("No hits passed L2 threshold %s", threshold)
    return hits


def retrieve(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Alias for retrieve_documents (kept for the original module stub)."""
    return retrieve_documents(query, k=k)
