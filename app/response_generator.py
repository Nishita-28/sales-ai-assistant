#python -m app.response_generator "Can we detect hydrogen?"

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from dotenv import load_dotenv

from app.claim_checker import check_restricted_claims, guardrail_source_text
from app.concentration import conversion_grounding_note, detect_conversion_request
from app.intent import Intent, UNKNOWN_INTENT
from app.retriever import is_ambiguous_product_reference, is_self_referential_without_own_products

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.md"
ANSWER_FORMAT_PATH = PROMPTS_DIR / "answer_format.md"
CUSTOMER_WORDING_PROMPT_PATH = PROMPTS_DIR / "customer_wording_prompt.md"


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

AMBIGUOUS_PRODUCT_MESSAGE = (
    "Could you specify which product you are referring to? Multiple products in the "
    "documentation may apply."
)


class ResponseGeneratorError(RuntimeError):
    """Wraps prompt-loading/LLM failures so callers can catch one specific
    type instead of a raw SDK/IO exception."""


@dataclass
class GeneratedAnswer:
    """The answer returned to the UI, plus its sources, confidence, and risk
    category. Customer-facing wording is NOT generated here -- it's a
    separate, on-demand call (see generate_customer_wording()) made only
    when a rep actually asks for it, since most answers never need one and
    generating it eagerly on every question roughly doubled completion
    time for no benefit. customer_wording_blocked tells the UI upfront
    whether that option should even be offered, without needing an LLM
    call to find out."""

    answer: str
    sources: list[tuple[str, str, str]]
    confidence: str  # "High" or "Low"
    risk: str  # risk category, or "None"
    customer_wording_blocked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "risk": self.risk,
            "customer_wording_blocked": self.customer_wording_blocked,
        }


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
_system_prompt_cache: Optional[str] = None
_customer_wording_prompt_cache: Optional[str] = None


def _load_system_prompt() -> str:
    """Loads and caches the combined system prompt + answer format."""
    global _system_prompt_cache
    if _system_prompt_cache is None:
        try:
            base = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
            answer_format = ANSWER_FORMAT_PATH.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise ResponseGeneratorError(f"Failed to read prompt file: {e}") from e
        _system_prompt_cache = f"{base}\n\nFormatting:\n\n{answer_format}"
    return _system_prompt_cache


def _load_customer_wording_prompt() -> str:
    global _customer_wording_prompt_cache
    if _customer_wording_prompt_cache is None:
        try:
            _customer_wording_prompt_cache = CUSTOMER_WORDING_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise ResponseGeneratorError(f"Failed to read prompt file: {e}") from e
    return _customer_wording_prompt_cache


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


def _build_user_message(question: str, matches: list[dict[str, Any]], intent: str) -> str:
    return (
        f"Question intent: {intent}\n\n"
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
        "[offline_mock] Placeholder answer -- no real LLM was called. "
        "Set LLM_PROVIDER=openai (or azure_openai) with a valid key to get a real answer."
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


def _call_llm_stream_raw(system_prompt: str, user_message: str) -> Iterator[str]:
    """Yields text deltas exactly as the provider sends them -- for OpenAI/
    Azure this is roughly token-by-token, which redraws the UI so often on
    a short answer that it reads as a flicker rather than a smooth
    "typing" effect. _call_llm_stream() wraps this with batching before
    handing it to callers; nothing outside this module should call the
    raw version directly."""
    if LLM_PROVIDER == "offline_mock":
        # Yields a few words at a time so the offline/dev path exercises
        # the same streaming UI code as a real provider.
        words = _call_offline_mock(user_message).split(" ")
        for i in range(0, len(words), 3):
            yield " ".join(words[i : i + 3]) + (" " if i + 3 < len(words) else "")
        return

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    if LLM_PROVIDER == "azure_openai":
        client = _get_azure_openai_chat_client()
        stream = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT, messages=messages, temperature=0.2, stream=True
        )
    elif LLM_PROVIDER == "openai":
        client = _get_openai_chat_client()
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, temperature=0.2, stream=True
        )
    else:
        raise ResponseGeneratorError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' -- expected 'openai', 'azure_openai', or 'offline_mock'."
        )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# Below this many buffered characters, a batch isn't flushed on size alone
# -- avoids a redraw every couple of characters on a fast stream.
_STREAM_BATCH_MIN_CHARS = 20

# Above this many seconds since the last flush, the buffer is flushed
# regardless of size -- keeps the display moving during a slower stretch
# instead of sitting on a half-formed word.
_STREAM_BATCH_MAX_WAIT_SECONDS = 0.1


def _batch_text_stream(chunks: Iterator[str]) -> Iterator[str]:
    """Buffers small text deltas and re-yields them in fewer, larger
    pieces, so the UI redraws in smooth steps instead of on every raw
    delta from the API."""
    buffer = ""
    last_flush = time.monotonic()
    for chunk in chunks:
        buffer += chunk
        now = time.monotonic()
        if len(buffer) >= _STREAM_BATCH_MIN_CHARS or (now - last_flush) >= _STREAM_BATCH_MAX_WAIT_SECONDS:
            yield buffer
            buffer = ""
            last_flush = now
    if buffer:
        yield buffer


def _full_catalog_coverage_note(matches: list[dict[str, Any]]) -> Optional[str]:
    """When retrieval was deliberately balanced across every currently
    approved product (see dispatcher.py's "no specific product named"
    fallback -- it forces exactly this), tells the model explicitly which
    products were checked. Proven necessary by testing: asked "how much
    time does each device take to charge," retrieval correctly covered
    all 6 products, but the model found charging-relevant content for
    only 2 of them and silently said nothing about the other 4 (fixed,
    wall-powered products with no battery/charging concept at all) --
    reading as if only those 2 were ever considered, not as a complete,
    deliberate check that correctly found nothing relevant for the rest."""
    try:
        from app.product_index import load_product_index
        products, _ = load_product_index()
    except (FileNotFoundError, OSError):
        return None
    all_names = {p.product_name for p in products}
    if len(all_names) < 2:
        return None
    matched_names = {m["metadata"].get("product_name") for m in matches if m.get("metadata")}
    if matched_names != all_names:
        return None
    names_list = ", ".join(sorted(all_names))
    return (
        f"This search deliberately checked every currently approved product: {names_list}. "
        "For each one, either state what its documents say about this question, or explicitly "
        "note that it isn't documented/applicable for that specific product -- never silently "
        "omit a product from the answer without saying so."
    )


def _call_llm_stream(system_prompt: str, user_message: str) -> Iterator[str]:
    """Streaming counterpart to _call_llm(): yields text in smooth,
    batched pieces as the LLM generates them, instead of returning the
    full reply only once it's complete. Nothing in this generator's body
    runs until it's first iterated (standard Python generator semantics),
    so callers can wrap the call in a try/except around iteration to
    catch request errors the same way as a non-streaming call."""
    yield from _batch_text_stream(_call_llm_stream_raw(system_prompt, user_message))


# Confidence is tied to retrieval quality, not the LLM's own self-assessment
# -- this just relabels it for display, with no thresholds of its own.
_CONFIDENCE_LABELS = {"high": "High", "low": "Low"}


def _display_confidence(retrieval_confidence: str) -> str:
    return _CONFIDENCE_LABELS.get(retrieval_confidence, "Low")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def stream_answer(
    question: str,
    retrieval: dict[str, Any],
    intent: Intent = UNKNOWN_INTENT,
    skip_ambiguity_check: bool = False,
) -> Iterator[str]:
    """Yields the internal answer's text in chunks as the LLM generates it
    -- feed this straight to st.write_stream() (or similar) for live
    display. Yields NO_SOURCE_MESSAGE once, without calling the LLM at
    all, if retrieval found nothing usable, or AMBIGUOUS_PRODUCT_MESSAGE
    once, also without calling the LLM, if the question refers to "this"/
    "it"/"the sensor" without naming a real product -- a deterministic
    check, not a prompt instruction, since testing showed the model
    reliably invents a product to answer about rather than asking which
    one was meant (retrieval finding topically-similar chunks isn't the
    same as the user having named a product), or NO_SOURCE_MESSAGE again
    if the question is self-referential ("our"/"we"/"us"/"the company")
    but every retrieved match came from a reference document rather than
    an actual product catalogue -- retrieving topically-related
    competitor content isn't the same as having this company's own
    answer. Once the caller has the full text (e.g. st.write_stream()'s
    return value), pass it to
    finalize_answer() to get sources, confidence, risk, and whether
    customer-facing wording is available -- that can't be known until the
    full answer exists.

    skip_ambiguity_check=True skips is_ambiguous_product_reference() --
    for a caller (rag_pipeline.py) whose dispatcher already resolved this
    question's product reference deterministically (kind="scoped"), which
    this older, independent, weaker check knows nothing about. Proven
    necessary by testing: asked "what is the warranty of the mEMS
    device," dispatcher.py correctly resolved "MEMS" to VISION H2 LD XX
    (the only current MEMS product) and scoped retrieval to just its
    catalogue -- but this function still re-ran its own ambiguity check
    on the raw question text regardless, which doesn't know about
    technology-alias resolution and produced the generic "Could you
    specify which product?" message anyway, discarding a correct answer
    dispatch() had already found."""
    matches = retrieval.get("matches") or []
    retrieval_confidence = retrieval.get("confidence", "none")

    if REQUIRE_SOURCES and (not matches or retrieval_confidence == "none"):
        yield NO_SOURCE_MESSAGE
        return

    if not skip_ambiguity_check and is_ambiguous_product_reference(question):
        yield AMBIGUOUS_PRODUCT_MESSAGE
        return

    if is_self_referential_without_own_products(question, matches):
        yield NO_SOURCE_MESSAGE
        return

    system_prompt = _load_system_prompt()
    user_message = _build_user_message(question, matches, intent)

    coverage_note = _full_catalog_coverage_note(matches)
    if coverage_note:
        user_message += "\n\n" + coverage_note

    # Hydrogen concentration unit conversion is arithmetic, not something to
    # trust the model with -- proven necessary by testing: asked to convert
    # 15,000 ppm to %LEL, the model computed 6% / 150% LEL instead of the
    # correct 1.5% / 37.5% LEL, while stating it with High confidence. The
    # conversion ratio (100% LEL = 4% H2 v/v = 40,000 ppm) is computed here
    # in code and handed to the model as a fact to state, not a calculation
    # to perform.
    conversion = detect_conversion_request(question)
    if conversion is not None:
        concentration, target_unit = conversion
        user_message += "\n\n" + conversion_grounding_note(concentration, target_unit)

    try:
        yield from _call_llm_stream(system_prompt, user_message)
    except ResponseGeneratorError:
        raise
    except Exception as e:
        raise ResponseGeneratorError(f"Answer generation failed: {e}") from e


_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def _contains_ungrounded_email(answer_text: str, source_text: str) -> bool:
    """True if the answer states an email address that doesn't literally
    appear anywhere in the retrieved source text. Proven necessary by
    testing: asked "Can I get a demo unit?", the model answered with a
    specific, real-looking contact email and phone number attributed to
    a named product -- reproducibly -- even though neither appeared in
    the excerpts actually retrieved for that question. Since MNST is a
    real company, the model can recall genuine contact details from its
    own general knowledge rather than the given context, which is
    exactly the kind of claim "answer only from the approved excerpts"
    is meant to rule out but doesn't reliably on its own (same class of
    problem as the ambiguous-product-reference check -- a deterministic
    check on the finished answer, not another prompt instruction)."""
    for email in _EMAIL_RE.findall(answer_text):
        if email.lower() not in source_text.lower():
            return True
    return False


def finalize_answer(question: str, retrieval: dict[str, Any], answer_text: str) -> GeneratedAnswer:
    """Builds the final GeneratedAnswer -- sources, confidence, risk, and
    whether customer-facing wording is available -- from a completed
    answer_text. Call this with whatever stream_answer() produced, once
    it's fully streamed (or with NO_SOURCE_MESSAGE if retrieval found
    nothing, same as stream_answer() would have yielded)."""
    matches = retrieval.get("matches") or []
    retrieval_confidence = retrieval.get("confidence", "none")
    answer_text = answer_text.strip() or NO_SOURCE_MESSAGE
    source_text = " ".join(m.get("text", "") for m in matches)

    if _contains_ungrounded_email(answer_text, source_text):
        answer_text = NO_SOURCE_MESSAGE

    # A "not documented" or "please clarify" answer showing as high
    # confidence reads as a contradiction to a rep -- confidence below
    # comes from retrieval similarity, which can be high even when the
    # model correctly declines to answer (the fact asked about isn't in
    # the topically-close excerpts it found) or a product reference is
    # ambiguous (the excerpts are a strong topical match, just not
    # confirmation of which product was meant). Force it to Low for
    # either fixed non-answer, so the badge reflects what the rep
    # actually got: nothing.
    if answer_text in (NO_SOURCE_MESSAGE, AMBIGUOUS_PRODUCT_MESSAGE):
        confidence = "Low"
    else:
        confidence = _display_confidence(retrieval_confidence)

    # A customer-facing rewrite is blocked unless the source text itself
    # backs every restricted term matched -- a confident retrieval can't
    # launder an answer that goes beyond what the source actually says.
    claim_result = check_restricted_claims(question, answer_text, guardrail_source_text(matches))

    return GeneratedAnswer(
        answer=answer_text,
        sources=_dedupe_sources(matches),
        confidence=confidence,
        risk=claim_result.category,
        customer_wording_blocked=claim_result.is_blocked or not ALLOW_CUSTOMER_FACING_OUTPUT,
    )


def generate_answer(
    question: str,
    retrieval: dict[str, Any],
    intent: Intent = UNKNOWN_INTENT,
    skip_ambiguity_check: bool = False,
) -> GeneratedAnswer:
    """Non-streaming convenience wrapper around stream_answer() +
    finalize_answer(), for CLI/scripted callers with no UI to stream
    into. Does not generate customer-facing wording -- call
    generate_customer_wording() separately, on demand, if that's needed."""
    answer_text = "".join(stream_answer(question, retrieval, intent, skip_ambiguity_check))
    return finalize_answer(question, retrieval, answer_text)


def stream_customer_wording(question: str, answer_text: str) -> Iterator[str]:
    """Streaming counterpart to generate_customer_wording() -- call this
    only when a rep actually asks for it (check
    GeneratedAnswer.customer_wording_blocked first; this function doesn't
    re-check the claims guardrail itself). Once the caller has the full
    text (e.g. st.write_stream()'s return value), pass it to
    finalize_customer_wording()."""
    system_prompt = _load_customer_wording_prompt()
    user_message = f"Question: {question}\n\nInternal answer:\n{answer_text}"

    try:
        yield from _call_llm_stream(system_prompt, user_message)
    except ResponseGeneratorError:
        raise
    except Exception as e:
        raise ResponseGeneratorError(f"Customer-facing wording generation failed: {e}") from e


def finalize_customer_wording(raw_text: str) -> Optional[str]:
    """Cleans up a completed customer-wording stream's full text. Returns
    None if the model declined to produce one."""
    text = raw_text.strip()
    if not text or text.lower().startswith(("not requested", "n/a", "none", "omit")):
        return None
    return text


def generate_customer_wording(question: str, answer_text: str) -> Optional[str]:
    """Non-streaming convenience wrapper around stream_customer_wording() +
    finalize_customer_wording(), for CLI/scripted callers with no UI to
    stream into."""
    raw_text = "".join(stream_customer_wording(question, answer_text))
    return finalize_customer_wording(raw_text)


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

    result = generate_answer(demo_question, demo_retrieval, demo_intent)

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
    if result.customer_wording_blocked:
        print("Customer-facing wording: (blocked)")
    else:
        print(f"Customer-facing wording: {generate_customer_wording(demo_question, result.answer)}")
