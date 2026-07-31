#python -m app.discovery_generator "Customer wants to monitor a lead-acid battery room for hydrogen buildup"
"""Pre-call discovery questions, then (once the rep has captured the
customer's answers) a grounded product recommendation -- one workflow, two
stages, both reusing the same retrieval and LLM plumbing as
answer_question(). Stage 1 identifies potential "Right to Win" points --
documented product differentiators relevant to the use case -- each with
its own supporting evidence and exploratory discovery questions (no claim-
checking -- no product is recommended yet, just questions to ask). Stage 2
matches the customer's actual answers against the knowledge base and
reuses claim_checker, since a recommendation is exactly the kind of
product claim that guardrail exists for."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.claim_checker import check_restricted_claims
from app.generation_helpers import build_context_block, dedupe_sources, extract_list_items, split_sections
from app.response_generator import ResponseGeneratorError, _call_llm

PROMPTS_DIR = Path(__file__).parent / "prompts"
DISCOVERY_PROMPT_PATH = PROMPTS_DIR / "discovery_questions_prompt.md"
RECOMMENDATION_PROMPT_PATH = PROMPTS_DIR / "product_recommendation_prompt.md"

# A recommendation needs real evidence, not just a couple of quick answers --
# this is a deterministic floor; the LLM applies its own judgment on top of it.
MIN_ANSWERS_FOR_RECOMMENDATION = 2

_prompt_cache: dict[str, str] = {}


def _load_prompt(path: Path) -> str:
    if path not in _prompt_cache:
        try:
            _prompt_cache[path] = path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise ResponseGeneratorError(f"Failed to read prompt file: {e}") from e
    return _prompt_cache[path]


@dataclass
class RightToWinPoint:
    """One documented differentiator: a short title, the evidence from the
    retrieved excerpts that supports it, and the exploratory questions that
    would confirm whether it actually matters to this customer."""

    title: str
    evidence: str
    questions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "evidence": self.evidence, "questions": self.questions}


@dataclass
class DiscoveryResult:
    """right_to_win comes from the LLM's structured reply, ranked highest
    relevance first; empty if no documented advantage was found for this
    use case. sources are the retrieved KB documents that grounded it,
    independent of whatever the LLM said."""

    right_to_win: list[RightToWinPoint]
    sources: list[tuple[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "right_to_win": [p.to_dict() for p in self.right_to_win],
            "sources": self.sources,
        }


@dataclass
class RecommendationResult:
    """Outcome of matching the customer's captured discovery answers
    against the knowledge base. outcome is one of "Recommend",
    "Trade-offs", "Insufficient"."""

    confirmed_priorities: list[str]
    outcome: str
    recommendation: str
    missing_information: str
    sources: list[tuple[str, str]]
    risk: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed_priorities": self.confirmed_priorities,
            "outcome": self.outcome,
            "recommendation": self.recommendation,
            "missing_information": self.missing_information,
            "sources": self.sources,
            "risk": self.risk,
        }


# Each Right-to-Win block starts with a "Right to Win: <title>" line --
# unlike split_sections' single-occurrence headers, this one repeats once
# per point, so it needs its own block-boundary scan.
_RIGHT_TO_WIN_TITLE_RE = re.compile(r"^Right to Win:\s*(.+)$", re.MULTILINE)


def _split_block_sections_inline(block: str, headers: list[str]) -> dict[str, str]:
    """Like split_sections, but also accepts content written inline right
    after the header's colon ("Supporting Evidence: some text"), not just
    content starting on the following line -- despite the prompt's own
    header-then-newline example, the model often writes short sections
    inline anyway, and a strict "header alone on its own line" match would
    silently drop that content instead of just reading it."""
    pattern = re.compile(
        r"^(" + "|".join(re.escape(h) for h in headers) + r"):[ \t]*(.*)$",
        re.MULTILINE,
    )
    sections: dict[str, str] = {}
    matches = list(pattern.finditer(block))
    for i, match in enumerate(matches):
        header = match.group(1).strip().lower()
        inline = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        rest = block[start:end].strip()
        sections[header] = (inline + ("\n" + rest if rest else "")).strip()
    return sections


def _parse_discovery_reply(raw_text: str) -> list[RightToWinPoint]:
    """Splits the structured reply into one RightToWinPoint per block, in
    the ranked order the LLM produced them. Returns [] if the LLM reported
    no documented Right to Win (or didn't follow the format at all --
    treated the same as "nothing to show" rather than guessing)."""
    title_matches = list(_RIGHT_TO_WIN_TITLE_RE.finditer(raw_text))

    points = []
    for i, match in enumerate(title_matches):
        title = match.group(1).strip()
        start = match.end()
        end = title_matches[i + 1].start() if i + 1 < len(title_matches) else len(raw_text)
        block = raw_text[start:end]

        sections = _split_block_sections_inline(block, ["Supporting Evidence", "Discovery Questions"])
        evidence = sections.get("supporting evidence", "").strip()
        questions = extract_list_items(sections.get("discovery questions", ""))

        points.append(RightToWinPoint(title=title, evidence=evidence, questions=questions))

    return points


def generate_discovery_questions(use_case_description: str, top_k: int = 8) -> DiscoveryResult:
    """Retrieves relevant KB excerpts for the described use case and asks
    the LLM to identify documented Right-to-Win points, each with its own
    supporting evidence and exploratory discovery questions."""
    from app.retriever import retrieve

    retrieval = retrieve(use_case_description, top_k=top_k)
    matches = retrieval.get("matches") or []

    system_prompt = _load_prompt(DISCOVERY_PROMPT_PATH)
    user_message = (
        f"Approved knowledge base excerpts:\n{build_context_block(matches)}\n\n"
        f"Customer use case: {use_case_description}"
    )

    try:
        raw_reply = _call_llm(system_prompt, user_message)
    except ResponseGeneratorError:
        raise
    except Exception as e:
        raise ResponseGeneratorError(f"Discovery question generation failed: {e}") from e

    right_to_win = _parse_discovery_reply(raw_reply)

    return DiscoveryResult(
        right_to_win=right_to_win,
        sources=dedupe_sources(matches),
    )


def _parse_recommendation_reply(raw_text: str) -> tuple[list[str], str, str, str]:
    """Splits the structured reply into (confirmed_priorities, outcome,
    recommendation, missing_information)."""
    sections = split_sections(
        raw_text, ["Confirmed Priorities", "Outcome", "Recommendation", "Missing Information"]
    )

    confirmed_priorities = extract_list_items(sections.get("confirmed priorities", ""))
    outcome_raw = sections.get("outcome", "").strip().lower()
    recommendation = sections.get("recommendation", "").strip()
    missing_information = sections.get("missing information", "").strip()

    if "trade" in outcome_raw:
        outcome = "Trade-offs"
    elif "recommend" in outcome_raw:
        outcome = "Recommend"
    else:
        # Fail safe: anything unclear is treated as insufficient rather
        # than risking an unwarranted recommendation.
        outcome = "Insufficient"

    return confirmed_priorities, outcome, recommendation, missing_information


def _is_recommendable_product_document(document_name: str) -> bool:
    """Sales-history/customer-record spreadsheets describe past deals, not
    a purchasable product -- they're valid supporting evidence in stage 1's
    Right-to-Win discovery (real deployment history is good social proof),
    but a spreadsheet must never be named as "the product" to recommend
    here. Proven necessary by testing: for some use cases, spreadsheet rows
    dominate the raw vector search (their prose reads a lot like a
    discovery-call description) and crowd out actual product documents
    entirely, and no prompt instruction alone reliably stopped the LLM
    from citing the spreadsheet as if it were a product."""
    return not document_name.lower().endswith(".xlsx")


def generate_recommendation(
    use_case_description: str,
    qa_pairs: list[tuple[str, str]],
    top_k: int = 15,
) -> RecommendationResult:
    """Matches the customer's captured discovery answers against the
    knowledge base and returns a recommendation, a trade-off comparison, or
    an explicit "not enough evidence yet" result. qa_pairs is the full list
    of (question, answer) pairs the rep recorded; unanswered ones (blank
    answer) are ignored."""
    from app.retriever import retrieve

    answered = [(q, a.strip()) for q, a in qa_pairs if a and a.strip()]

    if len(answered) < MIN_ANSWERS_FOR_RECOMMENDATION:
        return RecommendationResult(
            confirmed_priorities=[],
            outcome="Insufficient",
            recommendation="Not enough discovery answers have been captured yet to make a recommendation.",
            missing_information=(
                f"Record answers for at least {MIN_ANSWERS_FOR_RECOMMENDATION} discovery questions "
                "(currently have "
                f"{len(answered)}) before requesting a recommendation."
            ),
            sources=[],
            risk="None",
        )

    query = use_case_description + " " + " ".join(a for _, a in answered)
    # Over-fetch, then drop non-product documents (e.g. the sales-history
    # spreadsheet) before truncating to top_k -- otherwise filtering could
    # leave fewer than top_k usable matches even when real product
    # documents would have scored well enough to be included.
    retrieval = retrieve(query, top_k=top_k * 2)
    matches = [
        m for m in (retrieval.get("matches") or [])
        if _is_recommendable_product_document((m.get("metadata") or {}).get("document_name", ""))
    ][:top_k]

    qa_block = "\n".join(f"Q: {q}\nA: {a}" for q, a in answered)
    system_prompt = _load_prompt(RECOMMENDATION_PROMPT_PATH)
    user_message = (
        f"Approved knowledge base excerpts:\n{build_context_block(matches)}\n\n"
        f"Original use case: {use_case_description}\n\n"
        f"Customer's discovery call answers:\n{qa_block}"
    )

    try:
        raw_reply = _call_llm(system_prompt, user_message)
    except ResponseGeneratorError:
        raise
    except Exception as e:
        raise ResponseGeneratorError(f"Recommendation generation failed: {e}") from e

    confirmed_priorities, outcome, recommendation, missing_information = _parse_recommendation_reply(raw_reply)

    source_text = " ".join(m.get("text", "") for m in matches)
    claim_result = check_restricted_claims(qa_block, recommendation, source_text)

    return RecommendationResult(
        confirmed_priorities=confirmed_priorities,
        outcome=outcome,
        recommendation=recommendation,
        missing_information=missing_information,
        sources=dedupe_sources(matches),
        risk=claim_result.category,
    )


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    demo_use_case = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Customer wants to monitor a lead-acid battery room for hydrogen buildup"
    )
    result = generate_discovery_questions(demo_use_case)

    print(f"Use case: {demo_use_case}")
    if not result.right_to_win:
        print("\nNo documented Right to Win identified from the available evidence.")
    for i, point in enumerate(result.right_to_win, start=1):
        print(f"\n{i}. Right to Win: {point.title}")
        print(f"   Supporting Evidence: {point.evidence}")
        print("   Discovery Questions:")
        for q in point.questions:
            print(f"     - {q}")
    print("\nDrawn from:")
    for name, _ in result.sources:
        print(f"  - {name}")
