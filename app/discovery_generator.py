#python -m app.discovery_generator "Customer wants to monitor a lead-acid battery room for hydrogen buildup"
"""Pre-call discovery questions, then (once the rep has captured the
customer's answers) a grounded product recommendation -- one workflow, two
stages, both reusing the retrieval/LLM plumbing answer_question() uses.
Stage 1 identifies "Right to Win" points -- documented differentiators
with supporting evidence and exploratory questions, no claim-checking
since no product is recommended yet. Stage 2 matches the customer's
answers against the knowledge base and reuses claim_checker, since a
recommendation is exactly the kind of claim that guardrail exists for."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from app.claim_checker import check_restricted_claims, guardrail_source_text
from app.generation_helpers import build_context_block, dedupe_sources, extract_list_items, split_sections
from app.response_generator import ResponseGeneratorError, _call_llm_stream

PROMPTS_DIR = Path(__file__).parent / "prompts"
DISCOVERY_PROMPT_PATH = PROMPTS_DIR / "discovery_questions_prompt.md"
RECOMMENDATION_PROMPT_PATH = PROMPTS_DIR / "product_recommendation_prompt.md"
QUALIFICATION_PROMPT_PATH = PROMPTS_DIR / "qualification_questions_prompt.md"

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
    """One documented differentiator: a short title, a one-sentence
    "why it matters to this customer" line (read first -- see UI),
    the detailed evidence from the retrieved excerpts that backs it
    (shown collapsed by default, since it can run to a full paragraph),
    and the exploratory questions that would confirm whether it
    actually matters to this customer."""

    title: str
    why_it_matters: str
    evidence: str
    questions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "why_it_matters": self.why_it_matters,
            "evidence": self.evidence,
            "questions": self.questions,
        }


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
    "Trade-offs", "Insufficient". product is the single recommended name
    (blank for Trade-offs/Insufficient); candidate_products is the 2+
    names being weighed for a Trade-offs outcome -- both kept separate
    from the recommendation prose so the UI can show them prominently."""

    confirmed_priorities: list[str]
    outcome: str
    product: str
    candidate_products: list[str]
    recommendation: str
    missing_information: str
    sources: list[tuple[str, str]]
    risk: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed_priorities": self.confirmed_priorities,
            "outcome": self.outcome,
            "product": self.product,
            "candidate_products": self.candidate_products,
            "recommendation": self.recommendation,
            "missing_information": self.missing_information,
            "sources": self.sources,
            "risk": self.risk,
        }


# Each Right-to-Win block starts with a "Right to Win: <title>" line --
# unlike split_sections' single-occurrence headers, this one repeats once
# per point, so it needs its own block-boundary scan.
_RIGHT_TO_WIN_TITLE_RE = re.compile(r"^Right to Win:\s*(.+)$", re.MULTILINE)

# Single source of truth for both parse sites below (_parse_discovery_reply
# and stream_discovery_points), so the two can't drift out of sync.
_RIGHT_TO_WIN_SECTION_HEADERS = ["Why It Matters", "Supporting Evidence", "Discovery Questions"]


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

        sections = _split_block_sections_inline(block, _RIGHT_TO_WIN_SECTION_HEADERS)
        why_it_matters = sections.get("why it matters", "").strip()
        evidence = sections.get("supporting evidence", "").strip()
        questions = extract_list_items(sections.get("discovery questions", ""))

        points.append(
            RightToWinPoint(title=title, why_it_matters=why_it_matters, evidence=evidence, questions=questions)
        )

    return points


def stream_discovery_questions(
    use_case_description: str, top_k: int = 8
) -> tuple[list[dict[str, Any]], Iterator[str]]:
    """Streaming counterpart to generate_discovery_questions(). Returns
    (matches, text_stream) -- stream text_stream to the UI for live
    display of the raw reply, then pass matches and the full text to
    finalize_discovery_questions() for the parsed, ranked Right-to-Win
    points. What streams live is the raw structured text, not pre-rendered
    cards -- those render once finalize_ runs."""
    from app.retriever import retrieve

    retrieval = retrieve(use_case_description, top_k=top_k)
    matches = retrieval.get("matches") or []

    system_prompt = _load_prompt(DISCOVERY_PROMPT_PATH)
    user_message = (
        f"Approved knowledge base excerpts:\n{build_context_block(matches)}\n\n"
        f"Customer use case: {use_case_description}"
    )

    def _gen() -> Iterator[str]:
        try:
            yield from _call_llm_stream(system_prompt, user_message)
        except ResponseGeneratorError:
            raise
        except Exception as e:
            raise ResponseGeneratorError(f"Discovery question generation failed: {e}") from e

    return matches, _gen()


def stream_discovery_points(
    text_stream: Iterator[str],
) -> Iterator[tuple[list[RightToWinPoint], str, str]]:
    """Consumes a raw text_stream chunk by chunk, yielding
    (newly_completed_points, live_tail_text, full_text_so_far) after each
    chunk. A point only counts as complete once a further "Right to Win:"
    title appears after its block -- proof the model finished it, not
    still mid-question. live_tail_text is the in-progress point's text so
    far, for live display.

    Lets a UI render each Right-to-Win card as soon as it's done, rather
    than waiting for the whole multi-point reply -- a rep can start on
    point 1's questions while point 4 is still generating.

    The last point is never provably complete until the stream ends --
    callers parse it via finalize_discovery_questions(matches,
    full_text_so_far) once exhausted."""
    accumulated = ""
    known_complete = 0
    for chunk in text_stream:
        accumulated += chunk
        title_matches = list(_RIGHT_TO_WIN_TITLE_RE.finditer(accumulated))
        complete_count = max(len(title_matches) - 1, 0)

        new_points: list[RightToWinPoint] = []
        while known_complete < complete_count:
            start = title_matches[known_complete].end()
            end = title_matches[known_complete + 1].start()
            title = title_matches[known_complete].group(1).strip()
            block = accumulated[start:end]

            sections = _split_block_sections_inline(block, _RIGHT_TO_WIN_SECTION_HEADERS)
            why_it_matters = sections.get("why it matters", "").strip()
            evidence = sections.get("supporting evidence", "").strip()
            questions = extract_list_items(sections.get("discovery questions", ""))
            new_points.append(
                RightToWinPoint(title=title, why_it_matters=why_it_matters, evidence=evidence, questions=questions)
            )
            known_complete += 1

        tail_start = title_matches[known_complete].start() if known_complete < len(title_matches) else len(accumulated)
        yield new_points, accumulated[tail_start:], accumulated


def finalize_discovery_questions(matches: list[dict[str, Any]], raw_reply: str) -> DiscoveryResult:
    """Builds the final DiscoveryResult from a completed raw reply -- call
    with whatever stream_discovery_questions() produced, once it's fully
    streamed."""
    right_to_win = _parse_discovery_reply(raw_reply)
    return DiscoveryResult(
        right_to_win=right_to_win,
        sources=dedupe_sources(matches),
    )


def generate_discovery_questions(use_case_description: str, top_k: int = 8) -> DiscoveryResult:
    """Non-streaming convenience wrapper around stream_discovery_questions()
    + finalize_discovery_questions(), for CLI/scripted callers with no UI
    to stream into."""
    matches, text_stream = stream_discovery_questions(use_case_description, top_k)
    raw_reply = "".join(text_stream)
    return finalize_discovery_questions(matches, raw_reply)


# Qualification questions -- which of a deal's 8 MEDDPICC fields are still
# blank, and what to ask about them next. See app/deals_store.py for the
# field list/schema; a rep's answer is saved via deals_store.
# update_deal_fields(). Deliberately not a framework-selection system (no
# BANT/SPIN/etc.) -- just the one MEDDPICC framework for now.

# Which field is missing is deterministic (deals_store.missing_fields); so
# is the order to ask them in -- conversational priority informed by the
# methodology document's own guidance (ask Identify Pain/Metrics early;
# Decision Process/Paper Process only once engaged), not MEDDPICC_FIELDS'
# plain definitional order.
_QUALIFICATION_PRIORITY = [
    "identify_pain",
    "metrics",
    "economic_buyer",
    "champion",
    "competition",
    "decision_criteria",
    "decision_process",
    "paper_process",
]


@dataclass
class QualificationQuestion:
    """One still-missing MEDDPICC field for a specific deal -- a question
    phrased for that deal's use case, and a short rationale grounded in
    the retrieved methodology definition. field_key matches
    deals_store.MEDDPICC_FIELD_KEYS -- saving a rep's answer is
    deals_store.update_deal_fields(deal_id, **{field_key: answer})."""

    field_key: str
    field_label: str
    question: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_key": self.field_key,
            "field_label": self.field_label,
            "question": self.question,
            "rationale": self.rationale,
        }


def next_missing_fields(deal, limit: int = 3) -> list[tuple[str, str, str]]:
    """The top `limit` still-blank MEDDPICC fields for this deal, in
    conversational priority order -- (field_key, field_label,
    default_question) tuples. deal is a deals_store row (or any mapping
    with the 8 MEDDPICC field keys)."""
    from app.deals_store import MEDDPICC_FIELDS, missing_fields

    missing_keys = {key for key, _, _ in missing_fields(deal)}
    field_map = {key: (label, question) for key, label, question in MEDDPICC_FIELDS}
    ordered_keys = [k for k in _QUALIFICATION_PRIORITY if k in missing_keys]
    return [(key, *field_map[key]) for key in ordered_keys[:limit]]


def _fetch_methodology_matches(query: str, top_k: int) -> list[dict[str, Any]]:
    """Retrieves only from documents classified as Golden Frameworks &
    Sales Tactics -- scoping retrieval the same way
    _fetch_recommendable_matches (below) scopes to Product Catalogue
    only, so a qualification-question call doesn't get crowded out by or
    blended with unrelated product/use-case chunks."""
    from app.document_types import SALES_METHODOLOGY_REFERENCE, get_document_type
    from app.retriever import all_document_names, retrieve

    all_names = all_document_names()
    exclude = {name for name in all_names if get_document_type(name) != SALES_METHODOLOGY_REFERENCE}
    retrieval = retrieve(query, top_k=top_k, exclude_document_names=exclude, scope_to_products=False)
    return retrieval.get("matches") or []


def stream_qualification_questions(
    deal, limit: int = 3, top_k: int = 6
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]], Optional[Iterator[str]]]:
    """Streaming counterpart to generate_qualification_questions(). deal
    is a deals_store row. Returns (missing, matches, text_stream) -- missing
    is precomputed since finalize_qualification_questions() needs it as
    the deterministic fallback for any field the reply doesn't parse.
    Returns (missing, [], None), no LLM call, if nothing is left to ask."""
    missing = next_missing_fields(deal, limit)
    if not missing:
        return [], [], None

    query = "MEDDPICC " + " ".join(label for _, label, _ in missing)
    matches = _fetch_methodology_matches(query, top_k)

    fields_block = "\n".join(
        f'- {key}: {label} (default question: "{question}")' for key, label, question in missing
    )
    system_prompt = _load_prompt(QUALIFICATION_PROMPT_PATH)
    user_message = (
        f"Retrieved Sales Methodology excerpts:\n{build_context_block(matches)}\n\n"
        f"Customer use case: {deal['use_case']}\n\n"
        f"Missing MEDDPICC fields to cover:\n{fields_block}"
    )

    def _gen() -> Iterator[str]:
        try:
            yield from _call_llm_stream(system_prompt, user_message)
        except ResponseGeneratorError:
            raise
        except Exception as e:
            raise ResponseGeneratorError(f"Qualification question generation failed: {e}") from e

    return missing, matches, _gen()


_QUALIFICATION_FIELD_RE = re.compile(r"^Field:\s*(\S+)\s*$", re.MULTILINE)


def finalize_qualification_questions(
    missing: list[tuple[str, str, str]], raw_reply: str
) -> list[QualificationQuestion]:
    """Parses the LLM's per-field blocks. Any missing field the reply
    didn't produce a usable block for falls back to its deterministic
    default question -- this feature must never come back empty (or
    silently drop a field) just because a reply didn't parse perfectly."""
    field_matches = list(_QUALIFICATION_FIELD_RE.finditer(raw_reply))
    parsed: dict[str, tuple[str, str]] = {}
    for i, m in enumerate(field_matches):
        key = m.group(1).strip()
        start = m.end()
        end = field_matches[i + 1].start() if i + 1 < len(field_matches) else len(raw_reply)
        block = raw_reply[start:end]
        sections = _split_block_sections_inline(block, ["Question", "Rationale"])
        question = sections.get("question", "").strip()
        rationale = sections.get("rationale", "").strip()
        if question:
            parsed[key] = (question, rationale)

    results = []
    for key, label, default_question in missing:
        if key in parsed:
            question, rationale = parsed[key]
        else:
            question = default_question
            rationale = "Standard MEDDPICC qualification question -- not yet asked for this deal."
        results.append(QualificationQuestion(field_key=key, field_label=label, question=question, rationale=rationale))
    return results


def generate_qualification_questions(deal, limit: int = 3, top_k: int = 6) -> list[QualificationQuestion]:
    """Non-streaming convenience wrapper, for CLI/scripted callers."""
    missing, matches, text_stream = stream_qualification_questions(deal, limit, top_k)
    if text_stream is None:
        return []
    raw_reply = "".join(text_stream)
    return finalize_qualification_questions(missing, raw_reply)


def _parse_recommendation_reply(raw_text: str) -> tuple[list[str], str, str, list[str], str, str]:
    """Splits the structured reply into (confirmed_priorities, outcome,
    product, candidate_products, recommendation, missing_information)."""
    sections = split_sections(
        raw_text,
        [
            "Confirmed Priorities",
            "Outcome",
            "Recommended Product",
            "Candidate Products",
            "Recommendation",
            "Missing Information",
        ],
    )

    confirmed_priorities = extract_list_items(sections.get("confirmed priorities", ""))
    outcome_raw = sections.get("outcome", "").strip().lower()
    product = sections.get("recommended product", "").strip()
    candidate_products_raw = sections.get("candidate products", "").strip()
    recommendation = sections.get("recommendation", "").strip()
    missing_information = sections.get("missing information", "").strip()

    if "trade" in outcome_raw:
        outcome = "Trade-offs"
    elif "recommend" in outcome_raw:
        outcome = "Recommend"
    else:
        # Fail safe: anything unclear is Insufficient, not an unwarranted
        # recommendation.
        outcome = "Insufficient"

    # Only Recommend ever has a real product name -- discard anything
    # written for another outcome so the UI never shows a stray name.
    if outcome != "Recommend" or product.lower() == "none":
        product = ""

    # Same guardrail for the plural case -- only Trade-offs ever gets a
    # real candidate list.
    if outcome != "Trade-offs" or candidate_products_raw.lower() == "none":
        candidate_products = []
    else:
        candidate_products = [p.strip() for p in candidate_products_raw.split(",") if p.strip()]

    return confirmed_priorities, outcome, product, candidate_products, recommendation, missing_information


def _is_recommendable_product_document(document_name: str) -> bool:
    """Sales-history/customer-record spreadsheets and other reference
    documents describe past deals, not a purchasable product -- valid
    evidence in stage 1's Right-to-Win discovery, but must never be named
    as "the product" to recommend, since spreadsheet rows can read like a
    discovery-call description and dominate raw vector search. Driven by
    the Admin-assigned document type, not a filename guess."""
    from app.document_types import is_product_catalogue

    return is_product_catalogue(document_name)


def _fetch_recommendable_matches(query: str, top_k: int) -> list[dict[str, Any]]:
    """Retrieves real product documents only, excluding non-product
    documents from the vector search itself rather than filtering results
    afterward -- a dominant non-product document could otherwise occupy
    the entire top-k regardless of how wide it's fetched."""
    from app.retriever import all_document_names, retrieve

    exclude = {name for name in all_document_names() if not _is_recommendable_product_document(name)}
    retrieval = retrieve(query, top_k=top_k, exclude_document_names=exclude)
    return retrieval.get("matches") or []


def insufficient_recommendation(answered_count: int) -> RecommendationResult:
    return RecommendationResult(
        confirmed_priorities=[],
        outcome="Insufficient",
        product="",
        candidate_products=[],
        recommendation="Not enough discovery answers have been captured yet to make a recommendation.",
        missing_information=(
            f"Record answers for at least {MIN_ANSWERS_FOR_RECOMMENDATION} discovery questions "
            f"(currently have {answered_count}) before requesting a recommendation."
        ),
        sources=[],
        risk="None",
    )


def stream_recommendation(
    use_case_description: str,
    qa_pairs: list[tuple[str, str]],
    top_k: int = 15,
) -> tuple[Optional[list[dict[str, Any]]], Optional[str], Optional[Iterator[str]]]:
    """Streaming counterpart to generate_recommendation(). qa_pairs is the
    full list of (question, answer) pairs recorded; unanswered ones are
    ignored. Returns (None, None, None), no LLM call, if there isn't
    enough evidence yet -- caller should render
    insufficient_recommendation(answered_count) instead. Otherwise returns
    (matches, qa_block, text_stream) for finalize_recommendation()."""
    answered = [(q, a.strip()) for q, a in qa_pairs if a and a.strip()]
    if len(answered) < MIN_ANSWERS_FOR_RECOMMENDATION:
        return None, None, None

    query = use_case_description + " " + " ".join(a for _, a in answered)
    matches = _fetch_recommendable_matches(query, top_k)

    qa_block = "\n".join(f"Q: {q}\nA: {a}" for q, a in answered)
    system_prompt = _load_prompt(RECOMMENDATION_PROMPT_PATH)
    user_message = (
        f"Approved knowledge base excerpts:\n{build_context_block(matches)}\n\n"
        f"Original use case: {use_case_description}\n\n"
        f"Customer's discovery call answers:\n{qa_block}"
    )

    def _gen() -> Iterator[str]:
        try:
            yield from _call_llm_stream(system_prompt, user_message)
        except ResponseGeneratorError:
            raise
        except Exception as e:
            raise ResponseGeneratorError(f"Recommendation generation failed: {e}") from e

    return matches, qa_block, _gen()


def finalize_recommendation(
    matches: list[dict[str, Any]], qa_block: str, raw_reply: str
) -> RecommendationResult:
    """Builds the final RecommendationResult from a completed raw reply --
    call with whatever stream_recommendation() produced, once it's fully
    streamed."""
    confirmed_priorities, outcome, product, candidate_products, recommendation, missing_information = (
        _parse_recommendation_reply(raw_reply)
    )

    claim_result = check_restricted_claims(qa_block, recommendation, guardrail_source_text(matches))

    return RecommendationResult(
        confirmed_priorities=confirmed_priorities,
        outcome=outcome,
        product=product,
        candidate_products=candidate_products,
        recommendation=recommendation,
        missing_information=missing_information,
        sources=dedupe_sources(matches),
        risk=claim_result.category,
    )


def generate_recommendation(
    use_case_description: str,
    qa_pairs: list[tuple[str, str]],
    top_k: int = 15,
) -> RecommendationResult:
    """Non-streaming convenience wrapper around stream_recommendation() +
    finalize_recommendation(), for CLI/scripted callers with no UI to
    stream into."""
    answered_count = sum(1 for _, a in qa_pairs if a and a.strip())
    matches, qa_block, text_stream = stream_recommendation(use_case_description, qa_pairs, top_k)
    if text_stream is None:
        return insufficient_recommendation(answered_count)
    raw_reply = "".join(text_stream)
    return finalize_recommendation(matches, qa_block, raw_reply)


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
        print(f"   Why It Matters: {point.why_it_matters}")
        print(f"   Supporting Evidence: {point.evidence}")
        print("   Discovery Questions:")
        for q in point.questions:
            print(f"     - {q}")
    print("\nDrawn from:")
    for name, _ in result.sources:
        print(f"  - {name}")
