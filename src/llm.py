"""Local LLM client via Ollama.

Responsibilities (to be implemented later):
- Talk to a locally running Ollama server (default model: qwen2.5:3b).
- Provide a LangChain-compatible chat/LLM wrapper for the RAG chain.
- Never call OpenAI, Anthropic, Gemini, or any other paid cloud LLM API.
"""

from __future__ import annotations

from typing import Any


def get_llm(model_name: str | None = None, base_url: str | None = None) -> Any:
    """Return a local Ollama LLM instance.

    Args:
        model_name: Ollama model tag. Defaults to the configured local model.
        base_url: Ollama server URL. Defaults to http://localhost:11434.

    Returns:
        A LangChain LLM / chat model bound to Ollama.
    """
    raise NotImplementedError("Local Ollama LLM wiring is not implemented yet.")


def generate(prompt: str) -> str:
    """Generate a completion from the local LLM.

    Args:
        prompt: Fully formatted RAG prompt.

    Returns:
        Model text output.
    """
    raise NotImplementedError("Local LLM generation is not implemented yet.")
