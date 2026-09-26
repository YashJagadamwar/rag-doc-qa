"""
qa_chain.py
-----------
Builds the retrieval-augmented generation chain:
  question -> retrieve top-k chunks -> prompt LLM with numbered sources
  -> answer that must cite [1], [2], ... back to those chunks.

Returning the raw source chunks alongside the answer (not just the text)
is what lets the UI show verifiable citations instead of a bare answer.

The LLM is Ollama, which runs entirely on the local machine for free --
no API key, no per-token cost, no internet required once the model is
pulled. This requires the separate Ollama app (https://ollama.com) to be
installed, running, and have the chosen model pulled ahead of time, e.g.:
    ollama pull llama3.2

check_ollama_status() below exists specifically so failures show up as a
plain, actionable message in the UI instead of a raw Python traceback --
by far the most common setup issue is simply "Ollama isn't running yet."
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

DEFAULT_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question
using ONLY the numbered context excerpts below. If the answer is not contained
in the excerpts, say you don't have enough information -- never guess or use
outside knowledge.

When you use information from an excerpt, cite it inline like [1], [2], etc.,
matching the excerpt number. Use multiple citations if multiple excerpts
support a statement.

Context excerpts:
{context}
"""


@dataclass
class RagAnswer:
    answer: str
    sources: List[Document]


def check_ollama_status(
    model_name: str, base_url: str = DEFAULT_BASE_URL
) -> Tuple[bool, str]:
    """
    Checks whether Ollama is running and whether the requested model has
    been pulled. Returns (is_ready, message) so the caller can show a
    plain-English status instead of waiting for a chain call to fail.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False, (
            "Ollama isn't running. Install it from https://ollama.com, "
            "open the app, and try again."
        )

    available = [m.get("name", "") for m in data.get("models", [])]
    # Ollama tags models like "llama3.2:latest" -- compare on the base
    # name so "llama3.2" correctly matches "llama3.2:latest".
    wanted_base = model_name.split(":")[0]
    if any(m.split(":")[0] == wanted_base for m in available):
        return True, "Ollama is running and the model is ready."

    return False, (
        f"Ollama is running, but the model '{model_name}' isn't pulled yet. "
        f"Run this in a terminal, then try again: ollama pull {model_name}"
    )


def _format_context(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        chunk_id = d.metadata.get("chunk_id", "?")
        lines.append(f"[{i}] (source: {src}, chunk {chunk_id})\n{d.page_content}")
    return "\n\n".join(lines)


def answer_question(
    question: str,
    retriever,
    model_name: str = "llama3.2",
    temperature: float = 0.0,
    base_url: str = DEFAULT_BASE_URL,
) -> RagAnswer:
    docs = retriever.invoke(question)

    if not docs:
        return RagAnswer(
            answer="No relevant content was found in the uploaded documents.",
            sources=[],
        )

    context = _format_context(docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
    )

    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question})

    return RagAnswer(answer=result.content, sources=docs)
