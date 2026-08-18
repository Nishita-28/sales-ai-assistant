#python -m app.sales_aid_generator "multiple potential leak spots in close vicinity, 100% H2 contained in pipelines" "Metal Oxide Semiconductor Sensors"
"""Generates a short, customer-facing Sales Aid: a comparison document for
a specific use case that a customer's technical champion can circulate
internally. Reuses the same retrieval/LLM plumbing as the discovery flow --
this is a distinct deliverable (a document to hand to a customer), so it's
its own module rather than folded into discovery_generator.py.

Since this output is meant to leave the building, it goes through the same
claim-checking guardrail and the same ALLOW_CUSTOMER_FACING_OUTPUT gate as
the main assistant's customer-facing wording -- an unresolved risk flag or
a disabled gate means "internal draft, needs review," never "ready to
send."
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from app.claim_checker import check_restricted_claims, guardrail_source_text
from app.generation_helpers import (
    build_context_block,
    dedupe_sources,
    extract_lines,
    extract_list_items,
    split_sections,
)
from app.response_generator import ALLOW_CUSTOMER_FACING_OUTPUT, ResponseGeneratorError, _call_llm_stream

PROMPTS_DIR = Path(__file__).parent / "prompts"
SALES_AID_PROMPT_PATH = PROMPTS_DIR / "sales_aid_prompt.md"

_prompt_cache: dict[Path, str] = {}


def _load_prompt(path: Path) -> str:
    if path not in _prompt_cache:
        try:
            _prompt_cache[path] = path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise ResponseGeneratorError(f"Failed to read prompt file: {e}") from e
    return _prompt_cache[path]


@dataclass
class SalesAidResult:
    """customer_priorities/title/use_case_framing/comparison/customer_summary
    come from the LLM's structured reply. comparison is a markdown table
    (topic, MNST, competitor, and why each topic matters), scoped to only
    the topics relevant to customer_priorities -- not every retrieved spec.
    customer_summary is a separate, plain-prose paragraph meant to be
    pasted into an email as-is, not a rendering of the table. ready_for_
    customer is only True when the output is both enabled by config and
    cleared by the claim-checking guardrail -- otherwise this is an
    internal draft that needs human review before it can be shared."""

    customer_priorities: list[str]
    title: str
    use_case_framing: str
    comparison: list[str]
    customer_summary: str
    ready_for_customer: bool
    risk: str
    sources: list[tuple[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_priorities": self.customer_priorities,
            "title": self.title,
            "use_case_framing": self.use_case_framing,
            "comparison": self.comparison,
            "customer_summary": self.customer_summary,
            "ready_for_customer": self.ready_for_customer,
            "risk": self.risk,
            "sources": self.sources,
        }


def _parse_reply(raw_text: str) -> tuple[list[str], str, str, list[str], str]:
    """Splits the structured reply into (customer_priorities, title,
    use_case_framing, comparison, customer_summary). comparison is the raw
    lines of the markdown table (header, separator, and data rows) in
    order, so they can be rendered as one table block rather than as
    separate bullets."""
    sections = split_sections(
        raw_text,
        ["Customer Priorities", "Title", "Use Case Framing", "Comparison", "Customer-Ready Summary"],
    )

    customer_priorities = extract_list_items(sections.get("customer priorities", ""))
    title = sections.get("title", "").strip()
    use_case_framing = sections.get("use case framing", "").strip()
    comparison = extract_lines(sections.get("comparison", ""))
    customer_summary = sections.get("customer-ready summary", "").strip()

    return customer_priorities, title, use_case_framing, comparison, customer_summary


def stream_sales_aid(
    use_case_description: str,
    compare_against: str = "",
    top_k: int = 12,
) -> tuple[list[dict[str, Any]], Iterator[str]]:
    """Streaming counterpart to generate_sales_aid(). Returns (matches,
    text_stream) -- stream text_stream to the UI (e.g. via st.write_stream)
    for live display of the raw reply as it's generated, then pass matches
    and the full text it returns to finalize_sales_aid(). The raw reply is
    structured (priorities/title/framing/comparison table/summary), so
    what streams live is that raw text, not the final rendered
    layout -- the properly parsed sections render once finalize_ runs,
    same pattern as the Assistant page's streamed answer."""
    from app.retriever import all_document_names, retrieve

    query = use_case_description
    if compare_against.strip():
        query = f"{use_case_description} compared to {compare_against.strip()}"

    # A single unscoped retrieve() lets one side's ranking dominate the whole
    # top_k -- confirmed by testing: a query naming a competitor technology
    # pulled almost entirely from the competitor-comparison document, leaving
    # 1-2 MNST chunks, and vice versa when the query leaned MNST. A real
    # comparison needs guaranteed room for both sides, so retrieve them
    # separately (each excluding the other's documents) and merge, the same
    # split-retrieval fix already proven for discovery_generator.py's
    # spreadsheet-crowding bug.
    all_names = all_document_names()
    competitor_docs = {n for n in all_names if "competitor" in n.lower() or "comparison" in n.lower()}
    mnst_docs = set(all_names) - competitor_docs

    half = max(top_k // 2, 4)
    mnst_retrieval = retrieve(query, top_k=half, exclude_document_names=competitor_docs, scope_to_products=False)
    competitor_retrieval = retrieve(query, top_k=half, exclude_document_names=mnst_docs, scope_to_products=False)
    matches = (mnst_retrieval.get("matches") or []) + (competitor_retrieval.get("matches") or [])

    compare_line = (
        f"Compare specifically against: {compare_against.strip()}"
        if compare_against.strip()
        else "No specific competitor named -- use whichever competing technology/product in the "
        "excerpts is most relevant to this use case."
    )

    system_prompt = _load_prompt(SALES_AID_PROMPT_PATH)
    user_message = (
        f"Approved knowledge base excerpts:\n{build_context_block(matches)}\n\n"
        f"{compare_line}\n\n"
        f"Customer use case: {use_case_description}"
    )

    def _gen() -> Iterator[str]:
        try:
            yield from _call_llm_stream(system_prompt, user_message)
        except ResponseGeneratorError:
            raise
        except Exception as e:
            raise ResponseGeneratorError(f"Sales aid generation failed: {e}") from e

    return matches, _gen()


def finalize_sales_aid(
    use_case_description: str, matches: list[dict[str, Any]], raw_reply: str
) -> SalesAidResult:
    """Builds the final SalesAidResult from a completed raw reply -- call
    with whatever stream_sales_aid() produced, once it's fully streamed."""
    customer_priorities, title, use_case_framing, comparison, customer_summary = _parse_reply(raw_reply)

    full_text = "\n".join([title, use_case_framing, *comparison, customer_summary])
    claim_result = check_restricted_claims(use_case_description, full_text, guardrail_source_text(matches))

    return SalesAidResult(
        customer_priorities=customer_priorities,
        title=title,
        use_case_framing=use_case_framing,
        comparison=comparison,
        customer_summary=customer_summary,
        ready_for_customer=ALLOW_CUSTOMER_FACING_OUTPUT and not claim_result.is_blocked,
        risk=claim_result.category,
        sources=dedupe_sources(matches),
    )


def generate_sales_aid(
    use_case_description: str,
    compare_against: str = "",
    top_k: int = 12,
) -> SalesAidResult:
    """Non-streaming convenience wrapper around stream_sales_aid() +
    finalize_sales_aid(), for CLI/scripted callers with no UI to stream
    into."""
    matches, text_stream = stream_sales_aid(use_case_description, compare_against, top_k)
    raw_reply = "".join(text_stream)
    return finalize_sales_aid(use_case_description, matches, raw_reply)


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    demo_use_case = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Multiple potential leak spots in close vicinity, 100% H2 contained in pipelines"
    )
    demo_compare_against = sys.argv[2] if len(sys.argv) > 2 else ""

    result = generate_sales_aid(demo_use_case, demo_compare_against)

    print(f"Use case: {demo_use_case}")
    if demo_compare_against:
        print(f"Compare against: {demo_compare_against}")
    print("\nCustomer Priorities:")
    for p in result.customer_priorities:
        print(f"  - {p}")
    print(f"\nTitle: {result.title}")
    print(f"\nUse Case Framing: {result.use_case_framing}")
    print("\nComparison:")
    for c in result.comparison:
        print(f"  {c}")
    print(f"\nCustomer-Ready Summary: {result.customer_summary}")
    print(f"\nReady for customer: {result.ready_for_customer}")
    print(f"Risk: {result.risk}")
    print("Drawn from:")
    for name, _ in result.sources:
        print(f"  - {name}")
