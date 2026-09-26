"""
vectorstore.py
--------------
Wraps ChromaDB behind a small interface.

Embeddings run locally via sentence-transformers (free, no API key, no
internet needed after the first download). The LLM used for generation
(see qa_chain.py) is also local and free, via Ollama -- this project has
no paid dependency anywhere in the pipeline.

Design note -- why a new collection per upload instead of reusing one:
Deleting and recreating the on-disk database on every "Process documents"
click sounds simpler, but on Windows a previous ChromaDB client can still
hold a file lock on the old database, causing the delete to fail
intermittently. Instead, we keep ONE ChromaDB client alive for the life of
the app process and give every processed batch of documents its own
uniquely named collection. This never touches or deletes another
collection's files, so there is nothing to lock or conflict with -- it is
the simplest option that is also reliable on Windows.
"""

import uuid
from typing import List

import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_store"
COLLECTION_PREFIX = "documents"

# Module-level caches so we build the embedding model and the ChromaDB
# client only once per process, not once per button click.
_embedding_cache = {}
_client_cache = {}


def get_embedding_model() -> HuggingFaceEmbeddings:
    if "model" not in _embedding_cache:
        _embedding_cache["model"] = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_cache["model"]


def _get_client() -> chromadb.ClientAPI:
    if "client" not in _client_cache:
        _client_cache["client"] = chromadb.PersistentClient(path=PERSIST_DIR)
    return _client_cache["client"]


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Embeds the given chunks and stores them in a brand-new, uniquely
    named Chroma collection. Call this once per "Process documents"
    click; each call is fully independent of any previous one.
    """
    embeddings = get_embedding_model()
    client = _get_client()
    collection_name = f"{COLLECTION_PREFIX}_{uuid.uuid4().hex[:8]}"

    vs = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    vs.add_documents(chunks)
    return vs


def get_retriever(vs: Chroma, k: int = 4):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
