# Knowledge Base Chatbot with Vector DB

University assignment: a **local** RAG chatbot over one complete Bengali prose book from [Bengali Wikisource](https://bn.wikisource.org/).

Selected book: **কপালকুণ্ডলা** by বঙ্কিমচন্দ্র চট্টোপাধ্যায় (public domain).

This repository is currently a **project skeleton**. Crawler, embeddings, FAISS, RAG, and UI logic are not implemented yet.

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

## Setup (later, after logic is implemented)

1. Create a virtualenv and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Install [Ollama](https://ollama.com/) and pull the local model:

   ```bash
   ollama pull qwen2.5:3b
   ```

3. Copy environment defaults:

   ```bash
   cp .env.example .env
   ```

4. Crawl, index, then chat (commands will work once implemented):

   ```bash
   python scripts/crawl_book.py
   python scripts/build_index.py
   streamlit run app/streamlit_app.py
   ```

## Status

Placeholder modules and folders only. Do not expect crawling, retrieval, or chat to run yet.
# bangla-book-rag
