#python -m app.rag_pipeline "Can we detect hydrogen?"
"""Top-level orchestration: question -> intent classification -> retrieval
-> answer generation -> claim checking -> structured result. Just wires
the other components together; no logic of its own.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

from app.intent import Intent, classify_intent
from app.response_generator import (
    finalize_answer,
    finalize_customer_wording,
    generate_answer,
    generate_customer_wording,
    stream_answer,
    stream_customer_wording,
)
from app.retriever import retrieve


def answer_question(question: str) -> dict[str, Any]:
    """Runs one question through the full pipeline and returns a structured
    result dict. Raises RetrieverError or ResponseGeneratorError on failure.
    Non-streaming -- for CLI/scripted use. The UI uses stream_answer_question()
    instead so the answer can display live as it's generated. Does not
    generate customer-facing wording -- call get_customer_wording()
    separately, on demand, once a rep actually asks for one."""
    intent: Intent = classify_intent(question)
    retrieval = retrieve(question)
    result = generate_answer(question, retrieval, intent)

    return {
        "question": question,
        "intent": intent,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "risk_flag": result.risk,
        "customer_wording_blocked": result.customer_wording_blocked,
    }


def stream_answer_question(question: str) -> tuple[dict[str, Any], Iterator[str]]:
    """Streaming counterpart to answer_question(). Returns (retrieval, text_stream)
    -- feed text_stream to something like st.write_stream() for live display,
    then pass retrieval and the full text it returns to finalize_streamed_answer()
    to get the rest of the structured result (sources, confidence, risk,
    whether customer-facing wording is available)."""
    intent: Intent = classify_intent(question)
    retrieval = retrieve(question)
    return retrieval, stream_answer(question, retrieval, intent)


def finalize_streamed_answer(question: str, retrieval: dict[str, Any], answer_text: str) -> dict[str, Any]:
    """Completes a streamed answer -- call once the text_stream from
    stream_answer_question() is fully consumed, with the full text it
    produced."""
    result = finalize_answer(question, retrieval, answer_text)
    return {
        "question": question,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "risk_flag": result.risk,
        "customer_wording_blocked": result.customer_wording_blocked,
    }


def get_customer_wording(question: str, answer_text: str) -> str | None:
    """Generates a plain-language customer-facing rewrite of a previously
    generated answer, on demand. Raises ResponseGeneratorError on failure."""
    return generate_customer_wording(question, answer_text)


def stream_customer_wording_question(question: str, answer_text: str) -> Iterator[str]:
    """Streaming counterpart to get_customer_wording(). Feed to something
    like st.write_stream(), then pass the full text it returns to
    finalize_customer_wording_question()."""
    return stream_customer_wording(question, answer_text)


def finalize_customer_wording_question(raw_text: str) -> str | None:
    return finalize_customer_wording(raw_text)


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

    result = answer_question(question)

    print(f"Question: {result['question']}")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk flag: {result['risk_flag']}")
    print(f"Answer: {result['answer']}")
    print("Sources:")
    for name, label, text in result["sources"]:
        print(f"  - {name} ({label})")
        if text:
            print(f"      \"{text}\"")
    if result["customer_wording_blocked"]:
        print("Customer-facing wording: (blocked)")
    elif want_customer_wording:
        print(f"Customer-facing wording: {get_customer_wording(question, result['answer'])}")
    else:
        print("Customer-facing wording: (not requested -- pass --customer-wording to generate one)")
