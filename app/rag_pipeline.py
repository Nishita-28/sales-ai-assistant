#python -m app.rag_pipeline "Can we detect hydrogen?"
"""Top-level orchestration: question -> intent classification -> retrieval
-> answer generation -> claim checking -> structured result. Just wires
the other components together; no logic of its own.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.intent import Intent, classify_intent
from app.response_generator import generate_answer
from app.retriever import retrieve


def answer_question(question: str, want_customer_wording: bool = False) -> dict[str, Any]:
    """Runs one question through the full pipeline and returns a structured
    result dict. Raises RetrieverError or ResponseGeneratorError on failure."""
    intent: Intent = classify_intent(question)
    retrieval = retrieve(question)
    result = generate_answer(question, retrieval, intent, want_customer_wording)

    return {
        "question": question,
        "intent": intent,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "risk_flag": result.risk,
        "customer_wording": result.customer_wording,
    }


# ---------------------------------------------------------------------------
# CLI: python -m app.rag_pipeline --reindex
#      python -m app.rag_pipeline "question" [--customer-wording]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if "--reindex" in sys.argv:
        from app.retriever import RetrieverError, build_index, load_and_chunk_approved_docs

        docs_dir = Path("data/approved_docs")
        print(f"Loading and chunking documents from {docs_dir} ...")
        chunk_dicts = load_and_chunk_approved_docs(docs_dir)
        print(f"Produced {len(chunk_dicts)} chunks.")
        try:
            count = build_index(chunk_dicts)
            print(f"Indexed {count} chunks from {docs_dir}.")
        except RetrieverError as e:
            print(f"Index build failed: {e}", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)

    want_customer_wording = "--customer-wording" in sys.argv
    question_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    question = " ".join(question_args) if question_args else "Can we detect hydrogen?"

    result = answer_question(question, want_customer_wording=want_customer_wording)

    print(f"Question: {result['question']}")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk flag: {result['risk_flag']}")
    print(f"Answer: {result['answer']}")
    print("Sources:")
    for name, label in result["sources"]:
        print(f"  - {name} ({label})")
    if result["customer_wording"]:
        print(f"Customer-facing wording: {result['customer_wording']}")
    else:
        print("Customer-facing wording: (blocked or not requested)")
