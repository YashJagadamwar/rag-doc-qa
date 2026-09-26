"""
app.py
------
Streamlit front-end for the Generative AI-Powered Document Q&A System.

Flow:
  1. User uploads PDF/DOCX/TXT files.
  2. Files are chunked and embedded into a local ChromaDB vector store.
  3. User asks a question -> top-k relevant chunks are retrieved ->
     a local Ollama LLM answers using only those chunks, citing them
     by number.
  4. Retrieved chunks are shown so the user can verify each citation.

Everything here runs locally and for free: embeddings via
sentence-transformers, the LLM via Ollama, storage via ChromaDB. No API
key and no internet connection is required once Ollama's model has been
pulled once.

Setup (see README.md for full detail):
  1. Install Ollama from https://ollama.com and make sure it's running.
  2. Pull a model, e.g.:  ollama pull llama3.2
  3. pip install -r requirements.txt
  4. streamlit run app.py

Run with:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from src.ingestion import load_files, chunk_documents
from src.vectorstore import build_vectorstore, get_retriever
from src.qa_chain import answer_question, check_ollama_status
from src.evaluation import TestCase, run_evaluation, summarize

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")

# ---------- session state ----------
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, RagAnswer)
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Ollama model",
        ["llama3.2", "llama3.2:1b", "mistral", "qwen2.5:3b"],
        help="Must already be pulled locally, e.g. run: ollama pull llama3.2",
    )

    ollama_ready, ollama_message = check_ollama_status(model_name)
    if ollama_ready:
        st.success(ollama_message)
    else:
        st.error(ollama_message)

    st.caption(
        "This app is 100% free and local: embeddings run via "
        "sentence-transformers, the LLM runs via Ollama. No API key, "
        "no cost, no data leaves your machine."
    )

    st.divider()
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 120, step=20)
    top_k = st.slider("Chunks retrieved per question (k)", 1, 10, 4)

    st.divider()
    st.caption(
        "Architecture: PDF/DOCX/TXT → chunk → local embeddings "
        "(MiniLM) → ChromaDB → similarity search → local LLM answer "
        "with citations."
    )

# ---------- main: document upload ----------
st.title("📄 Generative AI-Powered Document Q&A System")
st.caption("RAG + LLM · Python · LangChain · ChromaDB · Streamlit · 100% free & local")

uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 3])
with col1:
    process_clicked = st.button("Process documents", type="primary", use_container_width=True)

if process_clicked:
    if not uploaded_files:
        st.warning("Upload at least one file first.")
    else:
        with st.spinner("Chunking and embedding documents..."):
            tmp_paths = []
            for uf in uploaded_files:
                suffix = os.path.splitext(uf.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_paths.append(tmp.name)

            loaded = load_files(tmp_paths)
            # restore original filenames for nicer citations
            for lf, uf in zip(loaded, uploaded_files):
                lf.filename = uf.name

            chunks = chunk_documents(loaded, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vs = build_vectorstore(chunks)

            st.session_state.retriever = get_retriever(vs, k=top_k)
            st.session_state.processed_files = [uf.name for uf in uploaded_files]
            st.session_state.chat_history = []

        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

if st.session_state.processed_files:
    st.info("Indexed: " + ", ".join(st.session_state.processed_files))

st.divider()

# ---------- main: Q&A ----------
st.subheader("Ask a question")

question = st.text_input("Your question", placeholder="e.g. What are the key findings in section 2?")
ask_clicked = st.button("Ask")

if ask_clicked:
    if not st.session_state.retriever:
        st.warning("Process at least one document first.")
    elif not question.strip():
        st.warning("Type a question first.")
    elif not ollama_ready:
        st.error(ollama_message)
    else:
        with st.spinner("Retrieving context and generating answer..."):
            rag_answer = answer_question(
                question=question,
                retriever=st.session_state.retriever,
                model_name=model_name,
            )
        st.session_state.chat_history.insert(0, (question, rag_answer))

for q, rag_answer in st.session_state.chat_history:
    with st.container(border=True):
        st.markdown(f"**Q: {q}**")
        st.markdown(rag_answer.answer)
        if rag_answer.sources:
            with st.expander(f"View {len(rag_answer.sources)} source excerpt(s)"):
                for i, doc in enumerate(rag_answer.sources, start=1):
                    st.markdown(
                        f"**[{i}] {doc.metadata.get('source')} "
                        f"(chunk {doc.metadata.get('chunk_id')})**"
                    )
                    st.text(doc.page_content[:500])

st.divider()

# ---------- main: evaluation ----------
with st.expander("🧪 Run a quick evaluation (optional)"):
    st.caption(
        "Provide test questions with an expected keyword and (optionally) a "
        "reference answer to measure retrieval hit-rate and answer similarity."
    )
    eval_input = st.text_area(
        "One test case per line: question | expected_keyword | reference_answer (optional)",
        height=120,
        placeholder="What is the refund policy? | refund | Refunds are issued within 30 days.",
    )
    run_eval_clicked = st.button("Run evaluation")

    if run_eval_clicked:
        if not st.session_state.retriever:
            st.warning("Process at least one document first.")
        elif not ollama_ready:
            st.error(ollama_message)
        else:
            cases = []
            for line in eval_input.strip().splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    cases.append(
                        TestCase(
                            question=parts[0],
                            expected_keyword=parts[1],
                            reference_answer=parts[2] if len(parts) > 2 and parts[2] else None,
                        )
                    )
            if not cases:
                st.warning("Add at least one valid test case line.")
            else:
                with st.spinner("Running evaluation..."):
                    results = run_evaluation(
                        cases, st.session_state.retriever, model_name
                    )
                    summary = summarize(results)

                st.json(summary)
                for r in results:
                    st.write(
                        f"- **{r.question}** — retrieval hit: {r.retrieval_hit}, "
                        f"similarity: {r.answer_similarity}"
                    )
