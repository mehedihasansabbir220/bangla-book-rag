"""RAG pipeline: retrieve from the book, then generate a grounded answer.

Responsibilities (to be implemented later):
- Accept a user question.
- Retrieve relevant chunks from the selected book only.
- Ask the local LLM to answer strictly from those chunks.
- Attach chapter/section/source citations.
- If the book does not contain the answer, return a clear no-answer message.
"""

from __future__ import annotations

from typing import Any


def answer_question(question: str) -> dict[str, Any]:
    """Run the full RAG pipeline for one question.

    Args:
        question: User question about the selected book.

    Returns:
        Dict with keys such as answer, citations, retrieved_chunks, and
        is_grounded (whether the answer came from the book).
    """
    raise NotImplementedError("RAG pipeline is not implemented yet.")
