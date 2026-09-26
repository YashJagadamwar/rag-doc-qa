"""
evaluation.py
-------------
A lightweight evaluation harness for the RAG pipeline -- no extra heavy
dependencies (no RAGAS/OpenAI-judge required), so it runs anywhere the
rest of the app runs. Two metrics are computed per test question:

1. Retrieval hit rate: does at least one retrieved chunk contain an
   expected keyword/phrase? (proxy for "did we find the right context")
2. Answer similarity: cosine similarity between the generated answer's
   embedding and a reference answer's embedding, using the same
   sentence-transformers model as the vector store (proxy for "is the
   answer semantically close to a known-good answer").

This is intentionally simple -- in an interview you can describe it as
a v1 evaluation approach and mention RAGAS / human eval / LLM-as-judge
as natural next steps.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.vectorstore import get_embedding_model
from src.qa_chain import answer_question


@dataclass
class TestCase:
    question: str
    expected_keyword: str
    reference_answer: Optional[str] = None


@dataclass
class EvalResult:
    question: str
    generated_answer: str
    retrieval_hit: bool
    answer_similarity: Optional[float]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_evaluation(
    test_cases: List[TestCase],
    retriever,
    model_name: str = "llama3.2",
) -> List[EvalResult]:
    embedder = get_embedding_model()
    results = []

    for case in test_cases:
        docs = retriever.invoke(case.question)
        retrieval_hit = any(
            case.expected_keyword.lower() in d.page_content.lower() for d in docs
        )

        rag_answer = answer_question(
            question=case.question,
            retriever=retriever,
            model_name=model_name,
        )

        similarity = None
        if case.reference_answer:
            vecs = embedder.embed_documents(
                [rag_answer.answer, case.reference_answer]
            )
            similarity = _cosine_sim(np.array(vecs[0]), np.array(vecs[1]))

        results.append(
            EvalResult(
                question=case.question,
                generated_answer=rag_answer.answer,
                retrieval_hit=retrieval_hit,
                answer_similarity=similarity,
            )
        )

    return results


def summarize(results: List[EvalResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    hit_rate = sum(r.retrieval_hit for r in results) / n
    sims = [r.answer_similarity for r in results if r.answer_similarity is not None]
    avg_sim = sum(sims) / len(sims) if sims else None
    return {
        "num_questions": n,
        "retrieval_hit_rate": round(hit_rate, 3),
        "avg_answer_similarity": round(avg_sim, 3) if avg_sim is not None else None,
    }
