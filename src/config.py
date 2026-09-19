"""Central configuration for the Bangla Book RAG project.

Holds paths, model names, chunking defaults, and Wikisource book metadata.
Later phases may load overrides from environment variables (.env).

No paid cloud APIs are used. The LLM runs locally via Ollama and embeddings
run locally via Hugging Face Sentence Transformers.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Selected book (Bengali Wikisource, public domain)
BOOK_TITLE = "কপালকুণ্ডলা"
BOOK_AUTHOR = "বঙ্কিমচন্দ্র চট্টোপাধ্যায়"
WIKISOURCE_BOOK_URL = (
    "https://bn.wikisource.org/wiki/"
    "%E0%A6%95%E0%A6%AA%E0%A6%BE%E0%A6%B2%E0%A6%95%E0%A7%81%E0%A6%A3%E0%A7%8D%E0%A6%A1%E0%A6%B2%E0%A6%BE"
    "_(%E0%A6%AC%E0%A6%99%E0%A7%8D%E0%A6%95%E0%A6%BF%E0%A6%AE%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0"
    "_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC"
    ",_%E0%A7%A7%E0%A7%AE%E0%A7%AD%E0%A7%A6)"
)

# Chunking (configurable)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Local embedding model (Bengali-capable, runs on CPU)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "

# Local LLM via Ollama (no paid API)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

# Retrieval
TOP_K = 4
SIMILARITY_THRESHOLD = 0.5
