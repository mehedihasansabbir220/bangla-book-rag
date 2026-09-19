# Knowledge Base Chatbot with Vector DB

A local Retrieval-Augmented Generation (RAG) chatbot that answers questions about one Bengali prose book. Answers are grounded in crawled Wikisource text, cited by chapter/section, and refused when the book does not contain the fact.

**No paid AI API key is required.** Embeddings run on the local CPU. Generation uses a local Ollama model.

**Demo video:** [demo/Demo-video.mov](demo/Demo-video.mov) (pipeline, in-book questions, and a no-answer case).

---

## 1. Project Overview

This repository implements a university assignment: a knowledge-base chatbot over a single Bangla book using:

- a Wikisource crawler limited to that book’s chapter hierarchy
- Bengali-aware cleaning and word-based chunking
- a multilingual embedding model (`intfloat/multilingual-e5-small`)
- a local FAISS vector index
- LangChain retrieval plus a local Ollama LLM (`qwen2.5:3b`)
- a Streamlit chat UI

The core rule is: **every answer must come only from the selected book**. If retrieval is too weak or the context does not support an answer, the system returns a fixed no-answer sentence instead of guessing.

---

## 2. Selected Book

| Field | Value |
| --- | --- |
| **Title** | রাজর্ষি |
| **Author** | রবীন্দ্রনাথ ঠাকুর (Rabindranath Tagore) |
| **Bengali Wikisource URL** | [https://bn.wikisource.org/wiki/রাজর্ষি_(১৯৬১)](https://bn.wikisource.org/wiki/রাজর্ষি_(১৯৬১)) |
| **Short description** | A public-domain historical novel about Tripura’s king গোবিন্দমাণিক্য, the priest রঘুপতি, the children হাসি and তাতা, and the conflict over temple sacrifice, exile, and the throne. The 1961 Wikisource edition is complete digital prose (not poetry, no OCR). The short title URL `…/wiki/রাজর্ষি` is a missing page; ingestion uses the 1961 edition root and its chapter subpages. |

---

## 3. Problem Statement

A chatbot that relies on a general-purpose LLM will mix training-data “knowledge” with the book and will not cite chapters. This project instead:

1. Ingests the **complete** book from Bengali Wikisource.
2. Indexes chapter-bounded chunks in FAISS.
3. Retrieves the nearest passages for each question.
4. Generates an answer **only** from those passages, with metadata citations.
5. Says clearly when the information is **not** in the book.

The interface is Bengali-first. There is no cloud LLM and no embedding API.

---

## 4. RAG Architecture

```
Wikisource
    → Crawling (chapter/subpage URLs only)
    → Cleaning (NFC Bengali, boilerplate removed)
    → Chunking (word windows + metadata)
    → Multilingual embedding (E5, local CPU)
    → FAISS (IndexFlatL2)
    → Retrieval (query embedding + top-k + L2 gate)
    → Local Ollama LLM (qwen2.5:3b)
    → Answer + citation (chapter / section / source URL)
```

Scripts:

- `scripts/crawl_book.py` — crawl and write `data/raw/book_pages.json`
- `scripts/build_index.py` — chunk, embed, persist FAISS
- `app/streamlit_app.py` — chat over the **existing** index (does not rebuild it)

---

## 5. Why Multilingual Embeddings

The book and the questions are in Bangla. An English-only MiniLM-style default would map Bengali poorly and retrieval would fail.

This project uses **`intfloat/multilingual-e5-small`**:

- **Bengali support** — E5-small is a multilingual retrieval model trained across many scripts, including Indic text. It is used here as a dense retriever, not as a translator.
- **Asymmetric prefixes** — documents are embedded with `passage: ` and questions with `query: `, as required by E5.
- **Local execution** — Sentence Transformers loads weights from the Hugging Face cache and runs on **CPU** (`EMBEDDING_DEVICE=cpu`).
- **No embedding API** — there is no OpenAI, Cohere, or Gemini embeddings client. Nothing in `requirements.txt` is a paid LLM/embedding SDK.

Vectors are L2-normalized (384 dimensions). FAISS then uses Euclidean distance on those unit vectors.

---

## 6. Chunking Strategy

Chunk size and overlap are measured in **words**, not characters (Bengali orthography makes character windows uneven).

| Setting | Production value |
| --- | --- |
| Chunk size | **500 words** |
| Overlap | **100 words** |
| Boundaries | One chapter at a time (chunks never mix two অধ্যায়) |

**Why 500 / 100**

- 500 words is large enough to keep a scene (names, dialogue, place) in one window for a novel chapter of a few thousand characters.
- 100-word overlap is **strictly smaller** than the chunk size, so windows advance and names that sit on a boundary can still appear in two chunks.
- Per-chapter splitting preserves citation metadata: every chunk copies `book`, `chapter`, `section`, and `source_url`.

Defaults live in `.env.example` (`CHUNK_SIZE`, `CHUNK_OVERLAP`). A second size (300 / 50) is used only in the bonus comparison; it is not the production index.

---

## 7. Vector Database

Embeddings are stored in **FAISS** (`faiss-cpu`) via LangChain:

- Files: `data/vectorstore/index.faiss` and `data/vectorstore/index.pkl`
- Index type: **IndexFlatL2** (exact Euclidean nearest neighbours)
- Score semantics: **`similarity_search_with_score` returns L2 distance. Lower is closer.** It is not cosine similarity and not “higher is better.”

The chatbot calls `load_index()` / `get_vectorstore(rebuild=False)`. If the index is missing, it errors and asks you to run `scripts/build_index.py`. It does **not** silently re-embed the book on Streamlit startup.

---

## 8. LLM

Generation uses **local Ollama**:

- Model: **`qwen2.5:3b`**
- Server: `http://localhost:11434` (`OLLAMA_BASE_URL`)
- Client: LangChain `ChatOllama` (temperature 0)

**No paid AI API key is required.** The LLM never talks to OpenAI, Anthropic, or Gemini. If Ollama is down or the model is not pulled, the UI shows an actionable error (`ollama serve`, `ollama pull qwen2.5:3b`).

---

## 9. Installation

Python **3.10–3.12** is preferred. macOS does not ship a `python` command; use `python3` or activate the virtualenv.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Install [Ollama](https://ollama.com/), then:

```bash
ollama serve
ollama pull qwen2.5:3b
```

Keep `ollama serve` running while you chat. If the `ollama` command is not found on this machine:

```bash
export PATH="$HOME/bin:$PATH"
```

Restart Streamlit after editing `.env` (settings are read at import time).

---

## 10. Book Ingestion

`python scripts/crawl_book.py` starts at `BOOK_URL` and discovers **all chapter/subpage URLs** that belong to this book’s hierarchy (PrefixIndex + table-of-contents order). It does **not** ingest only the main page and does **not** crawl the rest of Wikisource.

Each downloaded page is stored in `data/raw/book_pages.json` with:

- `book`, `chapter`, `section`
- `source_url`, `wikisource_title`
- `text`, `char_count`

Empty or missing pages are skipped and counted. A successful crawl of the 1961 edition yields one record per chapter from প্রথম পরিচ্ছেদ through উপসংহার.

---

## 11. Running the chatbot

After crawl + index exist:

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**.

If `data/raw/book_pages.json` or `data/vectorstore/index.faiss` is missing (fresh clone; those paths are gitignored):

```bash
python scripts/crawl_book.py
python scripts/build_index.py
streamlit run app/streamlit_app.py
```

---

## 12. RAG Pipeline

1. The user submits a Bengali question in Streamlit.
2. The question is embedded with the local E5 model (`query: ` prefix).
3. FAISS returns `TOP_K` (default **5**) nearest chunks and their **L2** scores.
4. If the **best** L2 score is worse than `RETRIEVAL_THRESHOLD` (default **0.40**, a maximum distance), generation is skipped and the no-answer sentence is returned.
5. Otherwise the best `GENERATION_K` (default **3**) passing chunks are sent to Ollama with a grounded system prompt.
6. The UI shows the model answer plus citations copied from chunk metadata.

Retrieval uses a LangChain FAISS retriever. Generation uses LangChain `ChatOllama`. The LLM is instructed not to use general knowledge and not to invent chapter numbers or URLs.

---

## 13. Citation Strategy

Citations are **not** generated by the LLM. They are copied from each retrieved chunk’s metadata:

| Field | Source |
| --- | --- |
| Book | `book` (selected title) |
| Chapter | `chapter` (Wikisource subpage name, e.g. `তৃতীয় পরিচ্ছেদ`) |
| Section | `section` (empty when the book has no nested sections) |
| Source URL | `source_url` (the Wikisource chapter URL) |

The RAG module deduplicates by `(chapter, source_url)` and Streamlit lists অধ্যায়/অনুচ্ছেদ plus the URL under the answer.

---

## 14. No-answer / hallucination protection

Three layers:

1. **Retrieval threshold (L2).** FAISS always returns *some* neighbour. A hit **passes** only when `score <= RETRIEVAL_THRESHOLD`. If the best neighbour fails, Ollama is not called. Default `0.40` is a measured maximum L2 (not cosine). Treating `0.25` as “similarity” on this index is incorrect and blocks most in-book questions.
2. **Grounded prompt.** The system prompt forbids outside knowledge, invention of names/dates, and adding citations. If the passages do not contain the answer, the model must output only:

   `এই তথ্যটি নির্বাচিত বইয়ের পাঠ্যে পাওয়া যায়নি।`

3. **UI.** That sentence is shown as a distinct warning. Citations are omitted when `found` is false.

This reduces fabrication; it does not make a 3B model perfect (see Limitations).

---

## 15. Test Questions

`tests/test_questions.json` contains **10** items:

- **9 answerable** questions spanning different chapters (প্রথম, দ্বিতীয়, তৃতীয়, চতুর্থ, দশম, ঊনবিংশ, পঞ্চবিংশ, সপ্তত্রিংশ, দ্বাচত্বারিংশ পরিচ্ছেদ). Each has `expected_chapter` and `expected_answer` taken from the crawled book text (not invented).
- **1 unanswerable** question (year of Bangladesh’s independence war), with `expected_chapter: null` and the no-answer sentence as `expected_answer`.

These files are the assignment’s required question set and expected answers.

---

## 16. Evaluation

Retrieval **hit rate** is computed live by:

```bash
python scripts/evaluate.py
```

Definition:

- For each **answerable** question, retrieve `TOP_K` chunks.
- **Hit** if `expected_chapter` equals any retrieved chunk’s `chapter` (NFC-normalized).
- **Hit rate** = hits / number of answerable questions.
- Unanswerable items are printed (best L2, whether they would pass the RAG gate) but are **not** in the denominator.

The script does **not** hard-code percentages. It also prints best L2 and threshold pass/fail so retrieval quality is not confused with LLM answer quality. Re-run after changing the index or `TOP_K`.

---

## 17. Bonus Experiment

Compare two **chunking** strategies with the same book, same E5 model, and the same hit-rate rule:

| Approach | Chunk size | Overlap | Role |
| --- | --- | --- | --- |
| A | 500 words | 100 words | Production index |
| B | 300 words | 50 words | In-memory rebuild only |

```bash
python scripts/evaluate.py --compare
```

Strategy A reuses `data/vectorstore/` (does not overwrite it). Strategy B embeds 300/50 chunks in memory. Hit = gold chapter in `TOP_K`.

**Do not copy a percentage into this README.** Run `--compare` on your machine and report the printed table in the demo or report. The script computes rates at runtime.

---

## 18. Limitations

- **Local 3B generation quality.** `qwen2.5:3b` is small. It can hedge, miss a name that is in the retrieved passage, or blend neighbouring chapters. Grounding is enforced by prompt + threshold, not by a guarantee of extractive accuracy.
- **Retrieval threshold.** `0.40` L2 was tuned on this book and this embedding. A different book, chunk size, or model needs new in-book vs out-of-book measurements. Too low → false no-answer; too high → LLM on irrelevant chunks.
- **Bengali generation.** The model is multilingual, not Bengali-specialized. Answers may be awkward, overly long, or mixed-register even when retrieval is correct.
- **FAISS L2 vs cosine.** Using a cosine-style cutoff (e.g. 0.25 as “minimum similarity”) on IndexFlatL2 is a metric error.
- **Clone is not runnable until crawl + index.** Generated `data/` files are gitignored.
- **Ollama must stay running.** The UI cannot generate if port 11434 is down.

---

## 19. Project Structure

```
bangla-book-rag/
├── app/
│   └── streamlit_app.py          # Chat UI
├── src/
│   ├── config.py                 # .env settings
│   ├── crawler.py                # Wikisource crawl
│   ├── text_processing.py        # Clean + chunk
│   ├── embeddings.py             # Local E5
│   ├── indexer.py                # FAISS build / load
│   ├── retriever.py              # Top-k + L2 scores
│   ├── llm.py                    # ChatOllama
│   ├── citations.py              # Metadata citations
│   ├── rag.py                    # Retrieve → gate → generate
│   └── __init__.py
├── scripts/
│   ├── crawl_book.py
│   ├── build_index.py
│   └── evaluate.py               # Hit rate + --compare
├── data/                         # Local artifacts (gitignored)
│   ├── raw/                      # book_pages.json
│   ├── processed/                # chunks.json
│   └── vectorstore/              # index.faiss, index.pkl
├── tests/
│   └── test_questions.json       # 10 questions + expected answers
├── demo/
│   └── Demo-video.mov            # Assignment demo (3–5 min)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
