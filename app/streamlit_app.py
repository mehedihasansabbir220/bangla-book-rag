"""Streamlit chatbot UI for the Bangla Book RAG system.

Talks only to the local RAG pipeline (FAISS + Ollama). No paid APIs and no
API keys are used or displayed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st

from src.config import BOOK_TITLE, BOOK_URL, OLLAMA_BASE_URL, OLLAMA_MODEL
from src.indexer import index_exists
from src.rag import NO_ANSWER, answer_question

PAGE_TITLE = f"{BOOK_TITLE} — জ্ঞানভাণ্ডার চ্যাটবট"
CHAT_PLACEHOLDER = "বই সম্পর্কে আপনার প্রশ্ন বাংলায় লিখুন…"
SPINNER_TEXT = "উত্তর খোঁজা হচ্ছে… অনুগ্রহ করে অপেক্ষা করুন।"

FAISS_MISSING_MESSAGE = (
    "FAISS index was not found. Build it before chatting:\n\n"
    "`python scripts/build_index.py`"
)
OLLAMA_DOWN_MESSAGE = (
    f"Ollama is not running at `{OLLAMA_BASE_URL}`.\n\n"
    "Install and start Ollama, then pull the local model:\n\n"
    f"`ollama pull {OLLAMA_MODEL}`"
)
OLLAMA_MODEL_MISSING_MESSAGE = (
    f"Ollama is running, but the model `{OLLAMA_MODEL}` is not installed.\n\n"
    f"`ollama pull {OLLAMA_MODEL}`"
)


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts).lower()


def classify_runtime_error(exc: BaseException) -> str:
    """Map pipeline failures to a short, actionable message."""
    blob = _exception_text(exc)
    if isinstance(exc, FileNotFoundError) or "faiss index not found" in blob:
        return FAISS_MISSING_MESSAGE
    model_missing = (
        "not found" in blob
        and ("model" in blob or OLLAMA_MODEL.lower() in blob)
    ) or "try pulling it first" in blob
    if model_missing:
        return OLLAMA_MODEL_MISSING_MESSAGE
    if any(
        token in blob
        for token in (
            "connection refused",
            "failed to connect",
            "connecterror",
            "connectionerror",
            "max retries exceeded",
            "nodename nor servname",
        )
    ):
        return OLLAMA_DOWN_MESSAGE
    return f"Could not answer this question.\n\n{exc}"


def _ollama_model_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
    return names


def _model_is_installed(names: list[str], wanted: str) -> bool:
    wanted = wanted.strip()
    aliases = {wanted}
    if ":" not in wanted:
        aliases.add(f"{wanted}:latest")
    return any(name in aliases for name in names)


def probe_ollama() -> str | None:
    """Return an error message if Ollama or the configured model is unavailable."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        payload = response.json() if response.content else {}
    except requests.RequestException:
        return OLLAMA_DOWN_MESSAGE
    if not isinstance(payload, dict):
        return OLLAMA_DOWN_MESSAGE
    if not _model_is_installed(_ollama_model_names(payload), OLLAMA_MODEL):
        return OLLAMA_MODEL_MISSING_MESSAGE
    return None


def readiness_errors() -> list[str]:
    errors: list[str] = []
    if not index_exists():
        errors.append(FAISS_MISSING_MESSAGE)
    ollama_error = probe_ollama()
    if ollama_error:
        errors.append(ollama_error)
    return errors


def render_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return
    st.markdown("**উৎস**")
    for index, source in enumerate(sources, start=1):
        chapter = (source.get("chapter") or "").strip()
        section = (source.get("section") or "").strip()
        url = (source.get("source_url") or "").strip()
        location = " / ".join(part for part in (chapter, section) if part) or "—"
        st.markdown(f"{index}. অধ্যায়/অনুচ্ছেদ: **{location}**")
        if url:
            st.markdown(f"   উৎস URL: {unquote(url)}")


def render_assistant_payload(payload: dict[str, Any]) -> None:
    error = payload.get("error")
    if error:
        st.error(error)
        return

    answer = (payload.get("content") or "").strip() or NO_ANSWER
    found = bool(payload.get("found"))
    if not found or answer == NO_ANSWER:
        st.warning(NO_ANSWER)
        return

    st.markdown(answer)
    render_sources(payload.get("sources") or [])


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="📖",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    init_session()

    st.title(BOOK_TITLE)
    st.caption(
        "নির্বাচিত বইয়ের পাঠ্য থেকে উত্তর। "
        f"[উইকিসংকলন]({BOOK_URL}) · স্থানীয় FAISS ও Ollama · কোনো ক্লাউড API নয়।"
    )

    with st.sidebar:
        st.subheader("অধিবেশন")
        st.write(f"**বই:** {BOOK_TITLE}")
        if st.button("নতুন আলাপ", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for error in readiness_errors():
        st.error(error)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_payload(message)

    question = st.chat_input(CHAT_PLACEHOLDER)
    if not question:
        return

    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    assistant_payload: dict[str, Any]
    with st.chat_message("assistant"):
        with st.spinner(SPINNER_TEXT):
            if not index_exists():
                assistant_payload = {
                    "role": "assistant",
                    "content": "",
                    "found": False,
                    "sources": [],
                    "error": FAISS_MISSING_MESSAGE,
                }
            else:
                try:
                    result = answer_question(question)
                    assistant_payload = {
                        "role": "assistant",
                        "content": result.get("answer") or NO_ANSWER,
                        "found": bool(result.get("found")),
                        "sources": result.get("sources") or [],
                        "error": None,
                    }
                except Exception as exc:  # noqa: BLE001 - show a useful UI error
                    assistant_payload = {
                        "role": "assistant",
                        "content": "",
                        "found": False,
                        "sources": [],
                        "error": classify_runtime_error(exc),
                    }
        render_assistant_payload(assistant_payload)

    st.session_state.messages.append(assistant_payload)


if __name__ == "__main__":
    main()
