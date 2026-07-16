#python -m app.response_generator "Can we detect hydrogen?"

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

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


# Per spec 12.2: "Do not answer if no relevant source is found."
REQUIRE_SOURCES = _env_flag("REQUIRE_SOURCES", True)

# Per spec 6.7.6: default false; gates customer_wording regardless of request.
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
    """Shaped to match what streamlit_app.py's UI already expects."""

    answer: str
    sources: list[tuple[str, str]]
    confidence: str  # "High" | "Low" -- see _display_confidence (retriever.py has no "Medium" tier)
    risk: str  # always "Unknown" here -- claim_checker.py decides the real value
    customer_wording: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "risk": self.risk,
            "customer_wording": self.customer_wording,
        }


def _unsupported_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=NO_SOURCE_MESSAGE,
        sources=[],
        confidence="Low",
        risk="Unknown",
        customer_wording=None,
    )


# ---------------------------------------------------------------------------
# Prompt loading (spec 9: "stored as editable files, not hardcoded")
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
# Building the LLM request from retriever.py's output
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


def _dedupe_sources(matches: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """A (document, section) pair can appear on more than one retrieved
    chunk; list it once, in first-seen order."""
    seen: set[tuple[str, str]] = set()
    sources: list[tuple[str, str]] = []
    for match in matches:
        source = _format_source(match.get("metadata") or {})
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


def _call_offline_mock(user_message: str) -> str:
    """No network call -- a fixed, well-formed reply so the rest of the
    pipeline is exercisable without a real API key."""
    return (
        "Short answer:\n"
        "[offline_mock] Placeholder answer -- no real LLM was called. "
        "Set LLM_PROVIDER=openai (or azure_openai) with a valid key to get a real answer.\n\n"
        "Sources:\n- (see excerpts)\n\n"
        "Confidence:\nMedium\n\n"
        "Risk flag:\nUnknown\n\n"
        "Customer-facing wording:\n[offline_mock] Placeholder customer-facing wording.\n"
    )


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Single dispatch point for LLM_PROVIDER."""
    if LLM_PROVIDER == "offline_mock":
        return _call_offline_mock(user_message)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    if LLM_PROVIDER == "azure_openai":
        client = _get_azure_openai_chat_client()
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT, messages=messages, temperature=0.2
        )
        return response.choices[0].message.content or ""

    if LLM_PROVIDER == "openai":
        client = _get_openai_chat_client()
        response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.2)
        return response.choices[0].message.content or ""

    raise ResponseGeneratorError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' -- expected 'openai', 'azure_openai', or 'offline_mock'."
    )


# ---------------------------------------------------------------------------
# Parsing the LLM's reply (see app/prompts/answer_format.md)
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r"^(Short answer|Sources|Confidence|Risk flag|Customer-facing wording):\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_llm_response(raw_text: str) -> dict[str, str]:
    """Splits the reply into answer_format.md's sections, keyed by
    lowercase header. A skipped section is simply absent from the result."""
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


# Confidence is tied to retrieval quality (spec 12.2), not the LLM's own
# self-assessment. retrieve()'s confidence field is retriever.py's alone --
# this just relabels it for display, with no thresholds of its own.
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
    customer-facing draft) for one question, given retriever.retrieve()'s
    output.

    No LLM call is made -- NO_SOURCE_MESSAGE is returned instead -- when
    REQUIRE_SOURCES is true and retrieval found nothing usable.
    customer_wording is None unless both want_customer_wording and
    ALLOW_CUSTOMER_FACING_OUTPUT are true.

    Raises ResponseGeneratorError if prompt loading or the LLM call fails.
    """
    matches = retrieval.get("matches") or []
    retrieval_confidence = retrieval.get("confidence", "none")

    if REQUIRE_SOURCES and (not matches or retrieval_confidence == "none"):
        return _unsupported_answer()

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

    return GeneratedAnswer(
        answer=answer_text,
        sources=_dedupe_sources(matches),
        confidence=_display_confidence(retrieval_confidence),
        # claim_checker.py owns real risk detection; never derived here.
        risk="Unknown",
        customer_wording=_extract_customer_wording(sections) if allow_customer_wording else None,
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
    for name, label in result.sources:
        print(f"  - {name} ({label})")
    if result.customer_wording:
        print(f"Customer-facing wording: {result.customer_wording}")
    else:
        print("Customer-facing wording: (blocked or not requested)")
