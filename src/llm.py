"""Local LLM client via Ollama (no paid APIs).

Talks to a locally running Ollama HTTP server, defaulting to:

- model: qwen2.5:3b
- base_url: http://localhost:11434

LangChain's ChatOllama is used so a later RAG chain can call `.invoke()`
the same way. This module does not retrieve book chunks or build a prompt
template; it only constructs the LLM.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Deterministic settings for knowledge-base QA (not creative writing).
LLM_TEMPERATURE = 0.0
LLM_SEED = 0

_llm: BaseChatModel | None = None


def get_llm(
    model_name: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Return a LangChain chat model bound to local Ollama.

    Args:
        model_name: Ollama model tag. Defaults to config OLLAMA_MODEL.
        base_url: Ollama server URL. Defaults to config OLLAMA_BASE_URL.

    Returns:
        A ChatOllama instance (temperature=0, fixed seed). No API key is used.
    """
    global _llm
    model = model_name or OLLAMA_MODEL
    url = base_url or OLLAMA_BASE_URL

    if (
        _llm is not None
        and model_name is None
        and base_url is None
    ):
        return _llm

    llm = ChatOllama(
        model=model,
        base_url=url,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
    )
    if model_name is None and base_url is None:
        _llm = llm
    return llm


def generate(prompt: str) -> str:
    """Generate a completion from the local Ollama model.

    This is a thin helper for smoke tests. The RAG pipeline will call
    `get_llm()` directly instead of this function.

    Raises:
        ConnectionError: If the Ollama server is not running or the model
            is not pulled.
    """
    llm = get_llm()
    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - map any transport/model error
        raise ConnectionError(
            "Could not reach the local Ollama server at "
            f"{OLLAMA_BASE_URL} with model {OLLAMA_MODEL}. "
            "Start Ollama, then run: ollama pull qwen2.5:3b"
        ) from exc
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)
