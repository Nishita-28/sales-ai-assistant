#python -m app.response_generator "Can we detect hydrogen?"

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from app.claim_checker import check_restricted_claims
from app.intent import Intent, UNKNOWN_INTENT

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.md"
ANSWER_FORMAT_PATH = PROMPTS_DIR / "answer_format.md"


LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")


def _env_flag(name: str, default: bool) -> bool:
    """Missing -> default; "false"/"0"/"no"/"" -> False; else True."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("false", "0", "no", "")


# If true, skip the LLM call entirely when no relevant source was found.
REQUIRE_SOURCES = _env_flag("REQUIRE_SOURCES", True)

# Off by default -- gates customer_wording regardless of what's requested.
ALLOW_CUSTOMER_FACING_OUTPUT = _env_flag("ALLOW_CUSTOMER_FACING_OUTPUT", False)

NO_SOURCE_MESSAGE = (
    "I do not have approved information for this. Please check with the "
    "technical or management team."
)


class ResponseGeneratorError(RuntimeError):
    """Wraps prompt-loading/LLM failures so callers can catch one specific
    type instead of a raw SDK/IO exception."""


@dataclass
class GeneratedAnswer:
    """The answer returned to the UI, plus its sources, confidence, risk
    category, and optional customer-facing draft."""

    answer: str
    sources: list[tuple[str, str, str]]
    confidence: str  # "High" or "Low"
    risk: str  # risk category, or "None"
    customer_wording: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "risk": self.risk,
            "customer_wording": self.customer_wording,
        }


def _unsupported_answer(question: str) -> GeneratedAnswer:
    # Still scan the question itself for restricted terms so the risk badge
    # stays meaningful even with no draft answer or sources.
    claim_result = check_restricted_claims(question, "", source_text="")
    return GeneratedAnswer(
        answer=NO_SOURCE_MESSAGE,
        sources=[],
        confidence="Low",
        risk=claim_result.category,
        customer_wording=None,
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
_system_prompt_cache: Optional[str] = None


def _load_system_prompt() -> str:
    """Loads and caches the combined system prompt + answer format."""
    global _system_prompt_cache
    if _system_prompt_cache is None:
        try:
            base = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
            answer_format = ANSWER_FORMAT_PATH.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise ResponseGeneratorError(f"Failed to read prompt file: {e}") from e
        _system_prompt_cache = f"{base}\n\nRespond using exactly this format:\n\n{answer_format}"
    return _system_prompt_cache


# ---------------------------------------------------------------------------
# Building the LLM request
# ---------------------------------------------------------------------------
def _format_source(metadata: dict[str, Any]) -> tuple[str, str]:
    """Turns one match's Chroma metadata into a (document_name,
    section-or-page label) pair."""
    document_name = metadata.get("document_name", "unknown")
    parts: list[str] = []
    if metadata.get("section_heading"):
        parts.append(str(metadata["section_heading"]))
    if metadata.get("page_number") is not None:
        parts.append(f"page {metadata['page_number']}")
    if metadata.get("slide_number") is not None:
        parts.append(f"slide {metadata['slide_number']}")
    label = ", ".join(parts) if parts else "General"
    return document_name, label


def _dedupe_sources(matches: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """A (document, section, extracted text) triple can appear on more than
    one retrieved chunk; list it once, in first-seen order. The extracted
    text is the exact chunk wording retrieved from the document -- shown in
    the UI so a rep can see what grounded the answer, not just its source."""
    seen: set[tuple[str, str, str]] = set()
    sources: list[tuple[str, str, str]] = []
    for match in matches:
        document_name, label = _format_source(match.get("metadata") or {})
        text = (match.get("text") or "").strip()
        source = (document_name, label, text)
        if source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def _build_context_block(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "(no approved excerpts were retrieved for this question)"
    blocks = []
    for i, match in enumerate(matches, start=1):
        document_name, label = _format_source(match.get("metadata") or {})
        blocks.append(f"[{i}] {document_name} ({label}):\n{match.get('text', '')}")
    return "\n\n".join(blocks)


def _build_user_message(
    question: str,
    matches: list[dict[str, Any]],
    intent: str,
    want_customer_wording: bool,
) -> str:
    wording_line = (
        "A customer-facing wording is requested -- include that section."
        if want_customer_wording
        else "No customer-facing wording is requested -- omit that section."
    )
    return (
        f"Question intent: {intent}\n"
        f"{wording_line}\n\n"
        f"Approved knowledge base excerpts:\n{_build_context_block(matches)}\n\n"
        f"Question: {question}"
    )


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
_chat_client = None  # OpenAI or AzureOpenAI client, whichever LLM_PROVIDER selects


def _get_openai_chat_client():
    global _chat_client
    if _chat_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ResponseGeneratorError("OPENAI_API_KEY is not set -- required when LLM_PROVIDER=openai.")
        try:
            from openai import OpenAI

            _chat_client = OpenAI(api_key=api_key)
        except Exception as e:
            raise ResponseGeneratorError(f"Failed to initialize the OpenAI client: {e}") from e
    return _chat_client


def _get_azure_openai_chat_client():
    global _chat_client
    if _chat_client is None:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key or not AZURE_OPENAI_ENDPOINT:
            raise ResponseGeneratorError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required when LLM_PROVIDER=azure_openai."
            )
        try:
            from openai import AzureOpenAI

            _chat_client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
            )
        except Exception as e:
            raise ResponseGeneratorError(f"Failed to initialize the Azure OpenAI client: {e}") from e
    return _chat_client


def warm_up() -> None:
    """Eagerly loads the system prompt and makes a throwaway chat completion
    call, so the network connection is already warm before the first
    question -- constructing the client alone doesn't touch the slow part
    (the TLS handshake on the first real request). max_tokens=1 keeps this
    as cheap as a chat completion call can be."""
    _load_system_prompt()
    if LLM_PROVIDER in ("openai", "azure_openai"):
        _call_llm("Hi", "Hi", max_tokens=1)


def _call_offline_mock(user_message: str) -> str:
    """A fixed, well-formed reply so the pipeline is testable without a
    real API key."""
    return (
        "Short answer:\n"
        "[offline_mock] Placeholder answer -- no real LLM was called. "
        "Set LLM_PROVIDER=openai (or azure_openai) with a valid key to get a real answer.\n\n"
        "Sources:\n- (see excerpts)\n\n"
        "Confidence:\nMedium\n\n"
        "Risk flag:\nUnknown\n\n"
        "Customer-facing wording:\n[offline_mock] Placeholder customer-facing wording.\n"
    )


def _call_llm(system_prompt: str, user_message: str, max_tokens: Optional[int] = None) -> str:
    """Single dispatch point for LLM_PROVIDER. max_tokens is only for
    callers (like warm_up) that want to cap response length -- real answer
    generation leaves it unset."""
    if LLM_PROVIDER == "offline_mock":
        return _call_offline_mock(user_message)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    extra = {"max_tokens": max_tokens} if max_tokens is not None else {}

    if LLM_PROVIDER == "azure_openai":
        client = _get_azure_openai_chat_client()
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT, messages=messages, temperature=0.2, **extra
        )
        return response.choices[0].message.content or ""

    if LLM_PROVIDER == "openai":
        client = _get_openai_chat_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, temperature=0.2, **extra
        )
        return response.choices[0].message.content or ""

    raise ResponseGeneratorError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' -- expected 'openai', 'azure_openai', or 'offline_mock'."
    )


# ---------------------------------------------------------------------------
# Parsing the LLM's reply
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r"^(Short answer|Sources|Confidence|Risk flag|Customer-facing wording):\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_llm_response(raw_text: str) -> dict[str, str]:
    """Splits the reply into its labeled sections, keyed by lowercase
    header. A skipped section is simply absent from the result."""
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(raw_text))
    for i, match in enumerate(matches):
        header = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        sections[header] = raw_text[start:end].strip()
    return sections


def _extract_customer_wording(sections: dict[str, str]) -> Optional[str]:
    text = sections.get("customer_facing_wording")
    if not text:
        return None
    if text.lower().startswith(("not requested", "n/a", "none", "omit")):
        return None
    return text


# Confidence is tied to retrieval quality, not the LLM's own self-assessment
# -- this just relabels it for display, with no thresholds of its own.
_CONFIDENCE_LABELS = {"high": "High", "low": "Low"}


def _display_confidence(retrieval_confidence: str) -> str:
    return _CONFIDENCE_LABELS.get(retrieval_confidence, "Low")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_answer(
    question: str,
    retrieval: dict[str, Any],
    intent: Intent = UNKNOWN_INTENT,
    want_customer_wording: bool = False,
) -> GeneratedAnswer:
    """Generates the internal answer (and, if requested and enabled, a
    customer-facing draft) for one question. Returns NO_SOURCE_MESSAGE
    instead of calling the LLM if retrieval found nothing usable."""
    matches = retrieval.get("matches") or []
    retrieval_confidence = retrieval.get("confidence", "none")

    if REQUIRE_SOURCES and (not matches or retrieval_confidence == "none"):
        return _unsupported_answer(question)

    allow_customer_wording = want_customer_wording and ALLOW_CUSTOMER_FACING_OUTPUT
    system_prompt = _load_system_prompt()
    user_message = _build_user_message(question, matches, intent, allow_customer_wording)

    try:
        raw_reply = _call_llm(system_prompt, user_message)
    except ResponseGeneratorError:
        raise
    except Exception as e:
        raise ResponseGeneratorError(f"Answer generation failed: {e}") from e

    sections = _parse_llm_response(raw_reply)
    answer_text = sections.get("short_answer") or raw_reply.strip() or NO_SOURCE_MESSAGE
    confidence = _display_confidence(retrieval_confidence)

    # Block customer_wording unless the source text itself backs every
    # restricted term matched -- a confident retrieval can't launder an
    # answer that goes beyond what the source actually says.
    source_text = " ".join(m.get("text", "") for m in matches)
    claim_result = check_restricted_claims(question, answer_text, source_text)

    return GeneratedAnswer(
        answer=answer_text,
        sources=_dedupe_sources(matches),
        confidence=confidence,
        risk=claim_result.category,
        customer_wording=(
            _extract_customer_wording(sections)
            if allow_customer_wording and not claim_result.is_blocked
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from app.intent import classify_intent
    from app.retriever import retrieve

    demo_question = sys.argv[1] if len(sys.argv) > 1 else "Can we detect hydrogen?"
    demo_intent = classify_intent(demo_question)
    demo_retrieval = retrieve(demo_question)

    result = generate_answer(demo_question, demo_retrieval, demo_intent, want_customer_wording=True)

    print(f"Question: {demo_question}")
    print(f"Intent: {demo_intent}")
    print(f"Confidence: {result.confidence}")
    print(f"Risk: {result.risk}")
    print(f"Answer: {result.answer}")
    print("Sources:")
    for name, label, text in result.sources:
        print(f"  - {name} ({label})")
        if text:
            print(f"      \"{text}\"")
    if result.customer_wording:
        print(f"Customer-facing wording: {result.customer_wording}")
    else:
        print("Customer-facing wording: (blocked or not requested)")
