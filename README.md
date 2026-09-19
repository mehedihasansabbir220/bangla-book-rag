# Knowledge Base Chatbot with Vector DB

University assignment: a **local** RAG chatbot over one complete Bengali prose book from [Bengali Wikisource](https://bn.wikisource.org/).

Selected book: **রাজর্ষি** from [Bengali Wikisource](https://bn.wikisource.org/wiki/রাজর্ষি) (public domain).

The Wikisource crawler is implemented. Embeddings, FAISS, RAG, and UI logic are not implemented yet.

## Constraints

- No OpenAI, Anthropic, Gemini, or other paid cloud LLM APIs
- LLM: local **Ollama** (`qwen2.5:3b`)
- Embeddings: local **Sentence Transformers** (`intfloat/multilingual-e5-small`)
- Vector store: **FAISS**
- Orchestration: **LangChain**
- UI: **Streamlit**
- Crawling: **requests** + **BeautifulSoup**

## Planned layout

```
bangla-book-rag/
├── app/streamlit_app.py      # Chat UI
├── src/                      # Library code (crawler → RAG)
├── scripts/                  # Crawl, index, evaluate CLIs
├── data/                     # Raw pages, processed chunks, FAISS index
├── tests/test_questions.json # 10 evaluation questions (later)
├── requirements.txt
└── README.md
```

## Setup

macOS does not provide a `python` command by default. Use `python3`, or activate the virtualenv (that creates a local `python`).

1. Create a virtualenv and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   After `source .venv/bin/activate`, the prompt should show `(.venv)` and both `python` and `pip` work.

2. Install [Ollama](https://ollama.com/) and pull the local model (needed later for the chatbot, not for crawling):

   ```bash
   ollama pull qwen2.5:3b
   ```

3. Copy environment defaults:

   ```bash
   cp .env.example .env
   ```

4. Crawl the book (index and chat come later):

   ```bash
   source .venv/bin/activate
   python scripts/crawl_book.py
   ```

   Without activating the venv:

   ```bash
   .venv/bin/python scripts/crawl_book.py
   ```

   Output: `data/raw/book_pages.json` (one object per chapter).

## Status

Crawler works. Do not expect embeddings, retrieval, or chat to run yet.
