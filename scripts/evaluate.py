"""Retrieval hit-rate evaluation on tests/test_questions.json.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --compare

For each *answerable* question, a hit is counted if the expected chapter
appears in any of the TOP_K retrieved chunks. Unanswerable questions are
shown but not included in the hit-rate denominator.

Also prints live L2 scores and whether the best hit would pass the RAG
threshold gate. This script does not hard-code percentages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR, RETRIEVAL_THRESHOLD, TOP_K  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.indexer import chunks_to_documents, index_exists, load_index  # noqa: E402
from src.retriever import retrieve_documents  # noqa: E402
from src.text_processing import chunk_pages  # noqa: E402

QUESTIONS_PATH = PROJECT_ROOT / "tests" / "test_questions.json"


def _nfc(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "").strip()


def _is_hit(expected_chapter: str, retrieved_chapters: list[str]) -> bool:
    want = _nfc(expected_chapter)
    return any(_nfc(chapter) == want for chapter in retrieved_chapters)


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def evaluate_retrieval(
    questions: list[dict],
    k: int | None = None,
    *,
    vectorstore=None,
) -> dict:
    """Return per-question rows and hit-rate over answerable items only."""
    top_k = TOP_K if k is None else k
    rows: list[dict] = []
    answerable = 0
    hits = 0
    threshold_passes = 0
    unanswerable = 0
    unanswerable_gated = 0

    for item in questions:
        question = (item.get("question") or "").strip()
        qtype = (item.get("type") or "").strip().lower()
        expected = item.get("expected_chapter")
        retrieved = retrieve_documents(question, k=top_k, vectorstore=vectorstore)
        chapters = [hit.get("chapter") or "" for hit in retrieved]
        scores = [hit.get("score") for hit in retrieved]
        best_l2 = scores[0] if scores else None
        passes_threshold = bool(retrieved[0]["passes_threshold"]) if retrieved else False

        row = {
            "question": question,
            "type": qtype,
            "expected_chapter": expected,
            "expected_answer": item.get("expected_answer"),
            "retrieved_chapters": chapters,
            "scores": scores,
            "best_l2": best_l2,
            "passes_threshold": passes_threshold,
            "hit": None,
        }

        if qtype == "answerable":
            answerable += 1
            hit = _is_hit(str(expected or ""), chapters)
            row["hit"] = hit
            if hit:
                hits += 1
            if passes_threshold:
                threshold_passes += 1
        elif qtype == "unanswerable":
            unanswerable += 1
            if not passes_threshold:
                unanswerable_gated += 1
        rows.append(row)

    hit_rate = (hits / answerable) if answerable else None
    threshold_pass_rate = (threshold_passes / answerable) if answerable else None
    return {
        "k": top_k,
        "threshold": RETRIEVAL_THRESHOLD,
        "answerable": answerable,
        "hits": hits,
        "hit_rate": hit_rate,
        "threshold_passes": threshold_passes,
        "threshold_pass_rate": threshold_pass_rate,
        "unanswerable": unanswerable,
        "unanswerable_gated": unanswerable_gated,
        "rows": rows,
    }


def _fmt_l2(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


# Bonus: two word-window sizes. Same book, same E5 model, same TOP_K / hit rule.
CHUNKING_STRATEGIES = (
    {
        "id": "A",
        "label": "500-word chunks, 100-word overlap (production)",
        "chunk_size": 500,
        "overlap": 100,
        "use_persisted": True,
    },
    {
        "id": "B",
        "label": "300-word chunks, 50-word overlap",
        "chunk_size": 300,
        "overlap": 50,
        "use_persisted": False,
    },
)


def _load_raw_pages() -> list[dict]:
    path = RAW_DIR / "book_pages.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/crawl_book.py")
    pages = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"No pages in {path}")
    return pages


def compare_chunking_strategies(questions: list[dict], k: int | None = None) -> list[dict]:
    """Rebuild (or reuse) indexes for two chunk sizes and score live hit rate.

    Strategy A reuses the persisted production index when it exists so the
    bonus run does not overwrite data/vectorstore/. Strategy B is built in
    memory only.
    """
    from langchain_community.vectorstores import FAISS

    pages = _load_raw_pages()
    embeddings = get_embeddings()
    rows: list[dict] = []

    for strategy in CHUNKING_STRATEGIES:
        if strategy["use_persisted"] and index_exists():
            store = load_index()
            n_chunks = store.index.ntotal
            source = "persisted FAISS index"
        else:
            chunks = chunk_pages(
                pages,
                chunk_size=strategy["chunk_size"],
                overlap=strategy["overlap"],
            )
            store = FAISS.from_documents(chunks_to_documents(chunks), embeddings)
            n_chunks = len(chunks)
            source = "in-memory FAISS (not saved)"

        scored = evaluate_retrieval(questions, k=k, vectorstore=store)
        rows.append(
            {
                "id": strategy["id"],
                "label": strategy["label"],
                "chunk_size": strategy["chunk_size"],
                "overlap": strategy["overlap"],
                "n_chunks": n_chunks,
                "source": source,
                "hits": scored["hits"],
                "answerable": scored["answerable"],
                "hit_rate": scored["hit_rate"],
            }
        )
    return rows


def _print_default_report(result: dict) -> None:
    for index, row in enumerate(result["rows"], start=1):
        print(f"{index:02d}. [{row['type']}] {row['question']}")
        gate = "PASS" if row["passes_threshold"] else "FAIL"
        print(
            f"    best L2: {_fmt_l2(row['best_l2'])}  "
            f"threshold gate: {gate}"
        )
        if row["type"] == "answerable":
            mark = "HIT" if row["hit"] else "MISS"
            print(f"    expected chapter: {row['expected_chapter']}")
            if row.get("expected_answer"):
                print(f"    expected answer: {row['expected_answer']}")
            print(f"    retrieved: {row['retrieved_chapters']}")
            print(f"    chapter hit: {mark}")
        else:
            print("    expected: null (unanswerable; not in hit-rate)")
            if row.get("expected_answer"):
                print(f"    expected answer: {row['expected_answer']}")
            print(f"    retrieved: {row['retrieved_chapters']}")
            print(
                "    RAG would skip LLM"
                if not row["passes_threshold"]
                else "    RAG would still call LLM (threshold too loose)"
            )
        print()

    rate = result["hit_rate"]
    gate_rate = result["threshold_pass_rate"]
    print("Answerable questions:", result["answerable"])
    print("Chapter hits (expected chapter in TOP_K):", result["hits"])
    if rate is None:
        print("Hit rate: n/a (no answerable questions)")
    else:
        print(
            f"Hit rate: {result['hits']}/{result['answerable']} = {rate:.2%}"
        )
    print(
        "Answerable threshold passes "
        f"(best L2 <= {result['threshold']}): {result['threshold_passes']}"
    )
    if gate_rate is None:
        print("Threshold pass rate: n/a")
    else:
        print(
            "Threshold pass rate: "
            f"{result['threshold_passes']}/{result['answerable']} = {gate_rate:.2%}"
        )
    print(
        "Unanswerable gated "
        f"(best L2 > {result['threshold']}): "
        f"{result['unanswerable_gated']}/{result['unanswerable']}"
    )


def _print_compare_report(rows: list[dict]) -> None:
    print("Bonus comparison: two chunking strategies")
    print("Same book, same embedding model (multilingual-e5-small), same TOP_K.")
    print("Hit = expected chapter appears in the TOP_K retrieved chunks.")
    print("Percentages are computed live; they are not hard-coded.")
    print()
    print(f"{'Approach':<48} {'Chunks':>7} {'Hits':>8} {'Hit rate':>10}")
    ranked = sorted(
        rows,
        key=lambda row: (-1 if row["hit_rate"] is None else -row["hit_rate"], row["id"]),
    )
    for row in rows:
        rate = row["hit_rate"]
        rate_s = "n/a" if rate is None else f"{rate:.2%}"
        hits = f"{row['hits']}/{row['answerable']}"
        print(f"{row['label']:<48} {row['n_chunks']:>7} {hits:>8} {rate_s:>10}")
        print(f"    index: {row['source']}")
    print()
    rates = [row["hit_rate"] for row in rows]
    if len(rates) >= 2 and rates[0] == rates[1]:
        print(
            "Hit rates are equal on this 9-question set. "
            f"Production keeps {rows[0]['id']} "
            f"({rows[0]['n_chunks']} chunks vs {rows[1]['n_chunks']})."
        )
    else:
        best = ranked[0]
        print(
            f"Higher chapter hit rate: {best['id']} — {best['label']} "
            f"({best['hits']}/{best['answerable']})."
        )


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Bonus: compare two chunking strategies with live hit rate",
    )
    args = parser.parse_args()

    questions = load_questions()
    print(f"Evaluating retrieval on {QUESTIONS_PATH}")
    print(f"TOP_K = {TOP_K}")
    print(f"RETRIEVAL_THRESHOLD (max L2) = {RETRIEVAL_THRESHOLD}")
    print()

    if args.compare:
        _print_compare_report(compare_chunking_strategies(questions))
        return

    result = evaluate_retrieval(questions)
    _print_default_report(result)


if __name__ == "__main__":
    main()
