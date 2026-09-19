"""Bangla Book RAG package.

Local pipeline: Wikisource crawler → Bengali preprocessing/chunking →
multilingual E5 embeddings → FAISS → Ollama generation → Streamlit UI.

No paid cloud LLM APIs are used.
"""
