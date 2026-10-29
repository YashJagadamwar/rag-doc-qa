# Generative AI-Powered Document Q&A System (RAG + LLM)

Upload PDF/DOCX/TXT files, ask questions in natural language, and get
answers grounded in your documents with inline citations back to the
exact source chunk — instead of the model guessing from general
knowledge.

**100% free and local.** No API key, no billing, no paid service of any
kind, and no data leaves your machine — embeddings and the LLM both run
on your own computer.

**Stack:** Python · LangChain · ChromaDB · Streamlit · sentence-transformers · Ollama

---

## 1. Architecture

```
 Upload (PDF/DOCX/TXT)
        │
        ▼
 ingestion.py   → load text, split into overlapping chunks
        │
        ▼
 vectorstore.py → embed chunks locally (MiniLM) → store in ChromaDB
        │
        ▼
   User question
        │
        ▼
 vectorstore.py → similarity search → top-k relevant chunks
        │
        ▼
 qa_chain.py    → local LLM (Ollama) answers using ONLY those chunks,
                  cites [1][2]...
        │
        ▼
 app.py (Streamlit) → shows answer + expandable source excerpts
```

**Key design decisions (good to mention in interviews):**
- **Fully local, zero cost.** Both the embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`) and the LLM (via Ollama) run
  on the local machine — no API key, no per-token billing, and documents
  never leave the device. This is a meaningful design trade-off to be
  able to discuss: local models are free and private but need a
  reasonably capable machine and are usually a notch below the largest
  hosted models in raw quality.
- **Grounded, citation-forced answers.** The system prompt instructs the
  LLM to answer only from the numbered context and to cite sources
  inline, which reduces hallucination and makes answers auditable.
- **Tunable chunking.** Chunk size/overlap and retrieval `k` are exposed
  as sliders because they are a real precision/context trade-off, not a
  fixed constant.
- **A new, isolated collection per upload.** Rather than reusing one
  ChromaDB collection (which can collide across sessions) or wiping and
  rebuilding the database on every upload (which can hit file-lock
  issues, especially on Windows), each "Process documents" click creates
  its own uniquely named collection. This keeps the app reliable to
  re-run without restarting it.
- **A live status check, not a hard failure.** The sidebar shows whether
  Ollama is actually running and whether the selected model is pulled,
  so setup problems show up as a clear one-line message instead of a
  Python traceback.
- **Lightweight evaluation harness** (`src/evaluation.py`): retrieval
  hit-rate (did we fetch the right chunk?) and answer-similarity (cosine
  similarity to a reference answer) without needing a heavy framework
  like RAGAS — a reasonable v1 that's easy to explain and extend.

## 2. Setup

### Step 1 — Install Ollama (the free local LLM engine)

Download and install it from **https://ollama.com** (Windows, Mac, and
Linux are all supported). Once installed, the Ollama app runs quietly in
the background — you don't need to open anything else for it.

### Step 2 — Pull a model

Open a terminal and run:

```bash
ollama pull llama3.2
```

This downloads a free, open-source model (a few GB) once. It's cached
locally after that. `llama3.2` is a good default; `llama3.2:1b` is a
smaller/faster option if your machine is limited on RAM.

### Step 3 — Install the Python project

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 — Run it

```bash
streamlit run app.py
```

It opens automatically in your browser (usually `http://localhost:8501`).
The sidebar will show a green "Ollama is running" message once
everything is set up correctly — if it shows red instead, it tells you
exactly what to fix (usually just "open the Ollama app" or "pull the
model").

The first run also downloads the MiniLM embedding model (~90 MB) once;
after that it's cached locally too.

## 3. Using it

1. Upload one or more PDF/DOCX/TXT files and click **Process documents**.
2. Ask a question in plain English. The answer will cite `[1]`, `[2]`,
   etc. — expand **"View source excerpt(s)"** to see exactly which
   chunk backs each citation.
3. Optional: open **"Run a quick evaluation"**, add a few test questions
   (`question | expected_keyword | reference_answer`), and get a
   retrieval hit-rate and answer-similarity score.

## 4. Project structure

```
rag-doc-qa/
├── app.py                 # Streamlit UI
├── src/
│   ├── ingestion.py        # File loading + chunking
│   ├── vectorstore.py      # Embeddings + ChromaDB
│   ├── qa_chain.py         # Retrieval + local LLM answer with citations
│   └── evaluation.py       # Retrieval hit-rate + answer similarity
├── requirements.txt
└── README.md
```

## 5. Troubleshooting

- **Sidebar shows "Ollama isn't running"** — open the Ollama app (it
  should be in your system tray/menu bar once installed) and refresh
  the page.
- **Sidebar shows "model isn't pulled yet"** — run
  `ollama pull <model name>` in a terminal, then refresh the page.
- **Answers are slow or low quality** — try a smaller model
  (`llama3.2:1b`) if your machine is limited on RAM, or a larger one
  (`mistral`) if you have the resources and want higher-quality answers.
- **"No such table" / database errors** — these were caused by an older
  ChromaDB version's in-memory mode; this project uses ChromaDB's
  on-disk persistent client with a fresh collection per upload
  specifically to avoid this class of bug.

## 6. Possible extensions

- Swap in RAGAS or an LLM-as-judge for deeper evaluation (faithfulness,
  answer relevance, context precision/recall).
- Add hybrid search (BM25 + embeddings) for better recall on exact terms
  like names, IDs, or numbers that embeddings alone can miss.
- Add a "use a hosted model instead" toggle (OpenAI/Anthropic) for
  comparison against the free local model.
- Add conversation memory so follow-up questions can reference earlier
  answers.
- Periodically clean up old ChromaDB collections left behind by past
  sessions.
