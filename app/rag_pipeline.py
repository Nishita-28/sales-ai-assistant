#python -m app.rag_pipeline "Can we detect hydrogen?"
"""Top-level orchestration: question -> intent classification -> retrieval
-> answer generation -> claim checking -> structured result. Just wires
the other components together; no logic of its own.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

from app.dispatcher import dispatch
from app.intent import Intent, classify_intent
from app.response_generator import (
    finalize_answer,
    finalize_customer_wording,
    generate_answer,
    generate_customer_wording,
    stream_answer,
    stream_customer_wording,
)
from app.retriever import main_assistant_excluded_document_names, retrieve


def _apply_nomenclature_notes(retrieval: dict[str, Any], result) -> dict[str, Any]:
    """Appends every scoped product's complete ordering-configuration
    table (see DispatchResult.nomenclature_notes) as high-confidence
    matches, so generation always has the real, deterministic codes
    available regardless of whether ordinary semantic retrieval happened
    to rank that chunk highly for this question's specific wording."""
    if result is None or not result.nomenclature_notes:
        return retrieval
    for note in result.nomenclature_notes:
        retrieval["matches"].append({
            "text": note["text"],
            "similarity": 1.0,
            "metadata": {"document_name": note["document_name"], "product_name": note["product_name"]},
        })
    return retrieval


def answer_question(question: str) -> dict[str, Any]:
    """Runs one question through the full pipeline and returns a structured
    result dict. Raises RetrieverError or ResponseGeneratorError on
    failure. Non-streaming -- the UI uses stream_answer_question() instead
    so the answer displays live. Does not generate customer-facing
    wording; call get_customer_wording() separately, on demand.

    Checks the dispatcher first: a catalog/filter question or nomenclature
    code gets answered directly from the Product Index, no LLM involved;
    a single resolved product scopes retrieval to its own catalogue;
    anything else falls through to the original, unscoped flow."""
    intent: Intent = classify_intent(question)
    result = dispatch(question)

    if result is not None and result.kind == "direct":
        retrieval = result.retrieval
        generated = finalize_answer(question, retrieval, result.answer_text)
    else:
        # main_assistant_excluded_document_names() is merged in
        # unconditionally, on top of whatever dispatch() decided -- these
        # internal-strategy documents must never reach the Assistant,
        # since nothing here gates a chunk before it reaches a rep.
        exclude = (result.exclude_document_names if result is not None else set()) | main_assistant_excluded_document_names()
        mentioned = result.scoped_product_names if result is not None else None
        retrieval = retrieve(question, exclude_document_names=exclude, mentioned_products=mentioned)
        retrieval = _apply_nomenclature_notes(retrieval, result)
        # dispatch() already resolved the product reference deterministically
        # for a "scoped" result -- skip response_generator's own, weaker
        # check, which would otherwise override a correct answer with a
        # generic "which product?" message.
        skip_ambiguity_check = result is not None and result.kind == "scoped"
        generated = generate_answer(question, retrieval, intent, skip_ambiguity_check)

    return {
        "question": question,
        "intent": intent,
        "answer": generated.answer,
        "sources": generated.sources,
        "confidence": generated.confidence,
        "risk_flag": generated.risk,
        "customer_wording_blocked": generated.customer_wording_blocked,
    }


def stream_answer_question(question: str) -> tuple[dict[str, Any], Iterator[str]]:
    """Streaming counterpart to answer_question(). Returns (retrieval, text_stream)
    -- feed text_stream to something like st.write_stream() for live display,
    then pass retrieval and the full text it returns to finalize_streamed_answer()
    to get the rest of the structured result (sources, confidence, risk,
    whether customer-facing wording is available). Same dispatcher check as
    answer_question() -- see its docstring."""
    intent: Intent = classify_intent(question)
    result = dispatch(question)

    if result is not None and result.kind == "direct":
        answer_text = result.answer_text
        return result.retrieval, iter([answer_text])

    exclude = (result.exclude_document_names if result is not None else set()) | main_assistant_excluded_document_names()
    mentioned = result.scoped_product_names if result is not None else None
    retrieval = retrieve(question, exclude_document_names=exclude, mentioned_products=mentioned)
    retrieval = _apply_nomenclature_notes(retrieval, result)
    skip_ambiguity_check = result is not None and result.kind == "scoped"
    return retrieval, stream_answer(question, retrieval, intent, skip_ambiguity_check)


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
#      python -m app.rag_pipeline --add-doc <path-to-file-already-in-data/approved_docs>
#      python -m app.rag_pipeline --remove-doc "<document name as indexed>"
#      python -m app.rag_pipeline "question" [--customer-wording]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if "--reindex" in sys.argv:
        from app.registry_builder import rebuild_product_registry
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
        registry_count = rebuild_product_registry(docs_dir)
        print(f"Rebuilt Product Registry: {registry_count} products.")
        raise SystemExit(0)

    # --add-doc / --remove-doc are the CLI counterparts of what the Admin
    # page's Documents tab already does on upload/remove (add_document_to_
    # index / remove_document_from_index in retriever.py) -- they touch only
    # the one document's chunks instead of re-embedding the whole corpus via
    # --reindex. The Product Registry rebuild that follows stays a full
    # rebuild either way (see rebuild_product_registry's docstring: it's
    # cheap -- a handful of catalogues, not thousands of chunks -- and needs
    # to see every catalogue together to keep product-name collision
    # avoidance consistent), so only the expensive embedding step is skipped.
    if "--add-doc" in sys.argv:
        from app.registry_builder import rebuild_product_registry
        from app.retriever import RetrieverError, add_document_to_index

        idx = sys.argv.index("--add-doc")
        if idx + 1 >= len(sys.argv):
            print("Usage: python -m app.rag_pipeline --add-doc <path-to-file-in-data/approved_docs>", file=sys.stderr)
            raise SystemExit(1)
        doc_path = Path(sys.argv[idx + 1])
        try:
            count, warnings = add_document_to_index(doc_path)
        except RetrieverError as e:
            print(f"Indexing failed: {e}", file=sys.stderr)
            raise SystemExit(1)
        print(f"Indexed {count} chunks from {doc_path.name} (other documents untouched).")
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)
        registry_count = rebuild_product_registry(doc_path.parent)
        print(f"Rebuilt Product Registry: {registry_count} products.")
        raise SystemExit(0)

    if "--remove-doc" in sys.argv:
        from app.registry_builder import rebuild_product_registry
        from app.retriever import RetrieverError, remove_document_from_index

        idx = sys.argv.index("--remove-doc")
        if idx + 1 >= len(sys.argv):
            print('Usage: python -m app.rag_pipeline --remove-doc "<document name as indexed>"', file=sys.stderr)
            raise SystemExit(1)
        doc_name = sys.argv[idx + 1]
        try:
            remove_document_from_index(doc_name)
        except RetrieverError as e:
            print(f"Removal failed: {e}", file=sys.stderr)
            raise SystemExit(1)
        print(f"Removed all chunks for '{doc_name}' from the index (other documents untouched).")
        registry_count = rebuild_product_registry(Path("data/approved_docs"))
        print(f"Rebuilt Product Registry: {registry_count} products.")
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
