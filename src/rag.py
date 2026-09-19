"""Grounded RAG pipeline over the selected Bengali book.

Flow:
    User question
    → E5 query embedding (via retriever)
    → FAISS top-k
    → inspect scores / apply L2 threshold
    → if best hit fails the threshold: no LLM, no-answer
    → else local Ollama, using the best GENERATION_K passing chunks
    → answer + citations from metadata

The LLM is never asked to invent sources. Citations are copied from
retrieved chunk metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.citations import collect_source_records
from src.config import BOOK_TITLE, GENERATION_K, RETRIEVAL_THRESHOLD
from src.llm import get_llm
from src.retriever import retrieve_documents

logger = logging.getLogger(__name__)

NO_ANSWER = "এই তথ্যটি নির্বাচিত বইয়ের পাঠ্যে পাওয়া যায়নি।"

SYSTEM_PROMPT = f"""তুমি একটি বাংলা জ্ঞানভাণ্ডার চ্যাটবট। তোমার একমাত্র জ্ঞানের উৎস হলো ব্যবহারকারীর বার্তায় দেওয়া “প্রসঙ্গ”, যা নির্বাচিত বই “{BOOK_TITLE}” থেকে উদ্ধৃত।

কঠোর নিয়ম:
1. শুধু প্রসঙ্গে লেখা তথ্য দিয়ে উত্তর দাও।
2. নিজের সাধারণ জ্ঞান, ইতিহাস পাঠ, ইন্টারনেট, বা অনুমান ব্যবহার করো না।
3. নাম, তারিখ, স্থান, ঘটনা বা সংলাপ উদ্ভাবন করো না।
4. প্রসঙ্গে যা স্পষ্ট নয়, তা সত্য বলে উপস্থাপন করো না।
5. উত্তর বাংলায় লেখো; সংক্ষিপ্ত, স্পষ্ট ও প্রসঙ্গভিত্তিক রাখো।
6. উৎস, অধ্যায় নম্বর বা URL নিজে যোগ করো না — সেটি সিস্টেম আলাদাভাবে যোগ করবে।
7. যদি প্রসঙ্গে প্রশ্নের উত্তর না থাকে, অন্য কোনো বাক্য না লিখে কেবল এই একটি বাক্য লেখো:
{NO_ANSWER}
"""


def _empty_result(answer: str = NO_ANSWER) -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": [],
        "found": False,
    }


def _build_user_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        chapter = hit.get("chapter") or "অজানা অধ্যায়"
        section = hit.get("section") or ""
        location = chapter if not section else f"{chapter} / {section}"
        url = hit.get("source_url") or ""
        text = (hit.get("text") or "").strip()
        blocks.append(
            f"[{index}] অধ্যায়: {location}\nউৎস: {url}\nপাঠ্য:\n{text}"
        )
    context = "\n\n".join(blocks)
    return (
        "প্রসঙ্গ (শুধু এই বইয়ের উদ্ধৃতি):\n"
        f"{context}\n\n"
        f"প্রশ্ন: {question}\n\n"
        "নিয়ম মনে রেখে শুধু প্রসঙ্গ থেকে উত্তর দাও।"
    )


def _llm_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts).strip()
    return str(content).strip()


def _is_no_answer(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    return NO_ANSWER in stripped


def answer_question(question: str) -> dict[str, Any]:
    """Run retrieve → threshold gate → grounded Ollama generation.

    Returns:
        {"answer": str, "sources": [citation dicts], "found": bool}
    """
    query = (question or "").strip()
    if not query:
        return _empty_result()

    hits = retrieve_documents(query)
    logger.info("RAG inspect: %s retrieved hit(s) for %r", len(hits), query[:80])

    if not hits:
        logger.info("RAG gate: no retrieval hits; skipping LLM")
        return _empty_result()

    best = hits[0]
    logger.info(
        "RAG inspect best: L2=%.4f cosine=%.4f pass=%s chapter=%s chunk=%s",
        best["score"],
        best["cosine_similarity"],
        best["passes_threshold"],
        best.get("chapter"),
        best.get("chunk_id"),
    )

    # L2 distance: a hit fails when score is *worse* (larger) than the threshold.
    if not best["passes_threshold"]:
        logger.info(
            "RAG gate: best L2 %.4f exceeds threshold %s; returning no-answer without LLM",
            best["score"],
            RETRIEVAL_THRESHOLD,
        )
        return _empty_result()

    grounded_hits = [hit for hit in hits if hit["passes_threshold"]]
    if not grounded_hits:
        return _empty_result()

    # Retrieval keeps TOP_K for evaluation; generation uses a short context
    # so the 3B model is less likely to answer from a weakly related chapter.
    gen_k = max(1, GENERATION_K)
    generation_hits = grounded_hits[:gen_k]
    logger.info(
        "RAG generation context: %s of %s passing hit(s) (GENERATION_K=%s)",
        len(generation_hits),
        len(grounded_hits),
        gen_k,
    )

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(query, generation_hits)),
    ]
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        raise ConnectionError(
            "Local Ollama did not generate an answer. "
            "Start Ollama and run: ollama pull qwen2.5:3b"
        ) from exc

    answer = _llm_content(response)
    if _is_no_answer(answer):
        logger.info("RAG: LLM reported no answer in context")
        return _empty_result(NO_ANSWER)

    sources = collect_source_records(generation_hits)
    return {
        "answer": answer,
        "sources": sources,
        "found": True,
    }
