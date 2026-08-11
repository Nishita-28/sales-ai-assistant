#cd sales-ai-assistant
#streamlit run app/streamlit_app.py
"""Internal AI Sales Assistant - Streamlit frontend."""

import html
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env explicitly since ADMIN_PASSWORD is read from it below.
load_dotenv()

# Streamlit sets the working directory to this file's folder, not the
# project root, so add the root to sys.path for the app.* imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import theme
from app.admin_page import render_admin_page
from app.claim_checker import warm_up as warm_up_claim_checker
from app.discovery_page import render_discovery_page
from app.feedback_store import record_feedback
from app.rag_pipeline import (
    finalize_customer_wording_question,
    finalize_streamed_answer,
    stream_answer_question,
    stream_customer_wording_question,
)
from app.requirements_page import render_requirements_page
from app.response_generator import ResponseGeneratorError
from app.response_generator import warm_up as warm_up_response_generator
from app.retriever import RetrieverError
from app.retriever import warm_up as warm_up_retriever
from app.sales_aid_page import render_sales_aid_page

APPROVED_DOCS_DIR = Path("data/approved_docs")

def get_admin_password() -> str:
    """Reads the admin password from secrets, then the environment. No
    hardcoded fallback -- an empty return triggers the "not set" error below."""
    try:
        return st.secrets["ADMIN_PASSWORD"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.environ.get("ADMIN_PASSWORD", "")


ADMIN_PASSWORD = get_admin_password()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
theme.apply_native_theme_option()  # no-op when UI_THEME=classic -- see app/theme.py
st.set_page_config(
    page_title="Internal AI Sales Assistant",
    layout="centered",
)
theme.inject_theme()  # no-op when UI_THEME=classic -- see app/theme.py

# ---------------------------------------------------------------------------
# Backend warm-up -- initializes API clients, the vector store connection,
# and cached files once per server process, so the first real question
# doesn't pay for setup that could happen at startup instead.
# ---------------------------------------------------------------------------
@st.cache_resource
def _warm_up_backend() -> None:
    try:
        warm_up_retriever()
        warm_up_response_generator()
        warm_up_claim_checker()
    except Exception:
        pass  # the first real question will retry and surface any real error


_warm_up_backend()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {question, answer, sources, confidence, risk, customer_wording_blocked, customer_wording}

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

SAMPLE_QUESTIONS = [
    "Can the FIXaHY H2 LD be used to monitor hydrogen buildup in a lead-acid battery room?",
    "Is the FIXaHY-4220MA PESO approved for hazardous areas?",
    "Can the FIXaHY H2 LD be integrated with our existing SCADA system over RS485 or 4-20mA?",
    "Can AURIGA be deployed in a hazardous area?",
]

RISK_COLORS = {
    "None": ("var(--bg-success, #EAF3DE)", "var(--text-success, #3B6D11)"),
    "Certification": ("#FAEEDA", "#854F0B"),
    "Accuracy": ("#FAEEDA", "#854F0B"),
    "Safety": ("#FCEBEB", "#791F1F"),
    "Pricing": ("#FAEEDA", "#854F0B"),
    "Legal": ("#FCEBEB", "#791F1F"),
    "Delivery": ("#FAEEDA", "#854F0B"),
    "Unknown": ("#F1EFE8", "#444441"),
}


# ---------------------------------------------------------------------------
# Copy-to-clipboard button (Streamlit has no native one)
# ---------------------------------------------------------------------------
_COPY_ICON_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:-2px;margin-right:6px;">'
    '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>'
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>'
    "</svg>"
)
_CHECK_ICON_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:-2px;margin-right:6px;">'
    '<path d="M20 6 9 17l-5-5"></path>'
    "</svg>"
)


def copy_button(text: str, label: str = "Copy answer", key: str = "", icon: bool = False):
    # Sanitize so the id can't break out of the id="..." / JS string it's used in.
    safe_id = "copy-btn-" + re.sub(r"[^a-zA-Z0-9_-]", "", key)
    # json.dumps gives a JS-safe quoted string; html.escape makes it safe
    # inside the double-quoted onclick="..." attribute.
    safe_text = html.escape(json.dumps(text))
    safe_label = html.escape(label)
    # icon=False (the default) keeps this exactly as it was pre-redesign --
    # only the enterprise theme's call sites opt into the icon variant.
    label_html = f"{_COPY_ICON_SVG}{safe_label}" if icon else safe_label
    copied_html = f"{_CHECK_ICON_SVG}Copied" if icon else "Copied"
    safe_label_html = html.escape(json.dumps(label_html))
    safe_copied_html = html.escape(json.dumps(copied_html))
    st.iframe(
        f"""
        <button id="{safe_id}" onclick="
            navigator.clipboard.writeText({safe_text});
            const btn = document.getElementById('{safe_id}');
            btn.innerHTML = {safe_copied_html};
            setTimeout(() => btn.innerHTML = {safe_label_html}, 1500);
        " style="
            padding:6px 12px; border-radius:6px; border:1px solid #ccc;
            background:#fff; cursor:pointer; font-size:13px; font-family:sans-serif;
            display:inline-flex; align-items:center;
        ">{label_html}</button>
        """,
        height=40,
    )


# ---------------------------------------------------------------------------
# Sidebar: knowledge base status + admin sign-in
# ---------------------------------------------------------------------------
def count_approved_docs() -> int:
    """Counts files in data/approved_docs."""
    if not APPROVED_DOCS_DIR.exists():
        return 0
    return sum(1 for p in APPROVED_DOCS_DIR.iterdir() if p.is_file())


with st.sidebar:
    if theme.is_enterprise_theme():
        # Logo is rendered separately (see theme.py) positioned above the
        # nav via CSS, since Streamlit's auto nav list is a DOM sibling
        # that can't be interleaved with content from `with st.sidebar:`.
        # Admin is a regular nav page here (see render_admin_gate below),
        # not a sidebar expander, so nothing admin-related renders here.
        st.markdown(theme.render_logo_html(), unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mnst-status">
              <div class="mnst-status-row"><span class="left"><span class="led"></span>Knowledge base</span><span class="val">{count_approved_docs()} docs</span></div>
              <div class="mnst-status-row"><span class="left"><span class="led"></span>Claim guardrail</span><span class="val">ON</span></div>
              <div class="mnst-status-row"><span class="left"><span class="led"></span>Mode</span><span class="val">Internal</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.subheader("Knowledge base")

        st.markdown("**Status**")
        st.success(f"Indexed docs: {count_approved_docs()}")
        st.info("Claim guardrail: ON")
        st.warning("Mode: Internal only")

        with st.expander("Admin sign-in"):
            if not ADMIN_PASSWORD:
                st.error("ADMIN_PASSWORD is not set. Add it to secrets before deploying.")
            elif st.session_state.admin_authenticated:
                st.caption("Signed in as admin -- see the Admin page.")
                if st.button("Sign out"):
                    st.session_state.admin_authenticated = False
                    st.rerun()
            else:
                with st.form("admin_login", clear_on_submit=True):
                    entered_password = st.text_input("Admin password", type="password")
                    submitted = st.form_submit_button("Unlock admin")
                if submitted:
                    if entered_password == ADMIN_PASSWORD:
                        st.session_state.admin_authenticated = True
                        st.rerun()
                    else:
                        st.error("Incorrect password.")


def render_admin_gate() -> None:
    """Admin as a regular nav page (enterprise theme only): shows a
    password form in the page content itself until unlocked, instead of a
    sidebar expander. Classic theme keeps the original sidebar-expander
    flow untouched, so Admin only appears in the nav list at all once
    st.session_state.admin_authenticated is already True (see pages list)."""
    if st.session_state.admin_authenticated:
        render_admin_page()
        return

    st.title("Admin")
    if not ADMIN_PASSWORD:
        st.error("ADMIN_PASSWORD is not set. Add it to secrets before deploying.")
        return

    st.caption("Sign in to manage documents, claims, feedback, and customer requirements.")
    with st.form("admin_login_page", clear_on_submit=True):
        entered_password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Unlock admin", type="primary")
    if submitted:
        if entered_password == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")


# ---------------------------------------------------------------------------
# Feedback: Correct/Wrong/Unsafe buttons on each answer
# ---------------------------------------------------------------------------
@st.dialog("What was wrong with this answer?")
def _report_wrong_dialog(question: str, answer: str) -> None:
    note = st.text_area("Notes (optional)", placeholder="What was incorrect?")
    if st.button("Submit", type="primary"):
        record_feedback(question, answer, "wrong", note)
        st.rerun()


# ---------------------------------------------------------------------------
# Assistant page
# ---------------------------------------------------------------------------
def render_assistant_page() -> None:
    if theme.is_enterprise_theme():
        _render_assistant_page_enterprise()
    else:
        _render_assistant_page_classic()


def _ask_and_record(question: str) -> None:
    """Runs one question through the pipeline and appends it to history.
    Shared by both the classic and enterprise layouts. Streams the answer
    live as it's generated (like ChatGPT) instead of showing a spinner
    until the full response is ready -- echoes the question first, in
    each theme's own style, so the live text doesn't appear with no
    visible prompt above it. Sources/badges/feedback buttons render
    normally once the completed turn is appended to history and the
    script reruns. Customer-facing wording is NOT generated here -- it's
    produced on demand (see _render_customer_wording()) only if a rep
    actually asks for it, since generating it eagerly on every question
    roughly doubled how long this step took, for a section most questions
    never need."""
    try:
        with st.spinner("Checking approved documents..."):
            retrieval, text_stream = stream_answer_question(question)
    except (RetrieverError, ResponseGeneratorError) as e:
        st.error(f"Something went wrong answering that question: {e}")
        return

    if theme.is_enterprise_theme():
        st.divider()
        st.markdown(f"**{question}**")
        answer_wrapper = st.container(border=True)
    else:
        with st.chat_message("user"):
            st.write(question)
        answer_wrapper = st.chat_message("assistant")

    try:
        with answer_wrapper:
            answer_text = st.write_stream(text_stream)
    except (RetrieverError, ResponseGeneratorError) as e:
        st.error(f"Something went wrong answering that question: {e}")
        return

    result = finalize_streamed_answer(question, retrieval, answer_text)
    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "risk": result["risk_flag"],
        "customer_wording_blocked": result["customer_wording_blocked"],
        "customer_wording": None,
    })
    st.rerun()


def _render_customer_wording(turn: dict, key_prefix: str) -> None:
    """Shared by both layouts: shows the blocked caption, a button to
    generate customer-facing wording on demand, or the generated wording
    once it exists. Mutates turn["customer_wording"] in place so it's
    cached in st.session_state.history and not regenerated on every
    rerun."""
    if turn["customer_wording_blocked"]:
        st.caption("Customer-facing wording blocked — escalate for review.")
        return

    if turn["customer_wording"]:
        with st.expander("Customer-facing wording", expanded=True):
            st.write(turn["customer_wording"])
            copy_button(
                turn["customer_wording"],
                "Copy customer wording",
                key=f"{key_prefix}-cust",
                icon=theme.is_enterprise_theme(),
            )
        return

    if st.button("Generate customer-facing wording", key=f"{key_prefix}-gen-cust"):
        try:
            with st.expander("Customer-facing wording", expanded=True):
                raw_text = st.write_stream(
                    stream_customer_wording_question(turn["question"], turn["answer"])
                )
        except ResponseGeneratorError as e:
            st.error(f"Something went wrong generating that: {e}")
            return
        turn["customer_wording"] = finalize_customer_wording_question(raw_text) or (
            "Could not generate customer-facing wording -- try again."
        )
        st.rerun()


def _render_assistant_page_classic() -> None:
    """The original layout, unchanged -- this is what UI_THEME=classic
    restores. Do not edit this without also updating the "revert" story:
    it exists specifically so reverting the enterprise redesign is a
    one-line env var change, not a reconstruction from memory."""
    st.title("Internal AI Sales Assistant")
    st.caption("Answers only from approved company documents. Always shows sources and confidence.")

    # Example question chips for first-time users
    if not st.session_state.history:
        st.markdown("**Try asking:**")
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, q in zip(cols, SAMPLE_QUESTIONS):
            if col.button(q, use_container_width=True):
                st.session_state.pending_question = q

    # Render past Q&A as a chat thread
    for idx, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            bg, fg = RISK_COLORS.get(turn["risk"], RISK_COLORS["Unknown"])
            badge_html = (
                f'<span style="font-size:12px;padding:3px 10px;border-radius:6px;'
                f'background:{bg};color:{fg};margin-right:6px;">Risk: {turn["risk"]}</span>'
                f'<span style="font-size:12px;padding:3px 10px;border-radius:6px;'
                f'background:#EAF3DE;color:#3B6D11;">Confidence: {turn["confidence"]}</span>'
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            st.write(turn["answer"])

            with st.expander("Sources"):
                for name, section, text in turn["sources"]:
                    st.markdown(f"- `{name}` — {section}")
                    if text:
                        st.markdown(f"> {text}")

            copy_button(turn["answer"], "Copy answer", key=f"ans-{idx}")

            _render_customer_wording(turn, key_prefix=f"cust-{idx}")

            fcol1, fcol2, fcol3 = st.columns(3)
            if fcol1.button("Correct", key=f"ok-{idx}"):
                record_feedback(turn["question"], turn["answer"], "correct")
                st.toast("Thanks for the feedback!")
            if fcol2.button("Wrong", key=f"bad-{idx}"):
                _report_wrong_dialog(turn["question"], turn["answer"])
            if fcol3.button("Unsafe", key=f"unsafe-{idx}"):
                record_feedback(turn["question"], turn["answer"], "unsafe")
                st.toast("Thanks for flagging this — reported for review.")

    # Question input
    question = st.chat_input("Ask a sales or application question")
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
    if question:
        question = question.strip()

    if question:
        _ask_and_record(question)


def _render_assistant_page_enterprise() -> None:
    """Search-bar-at-top, answer-card-below layout -- deliberately not a
    chat thread. Most recent answer is shown first, like a search/report
    tool rather than an accumulating chat log."""
    st.title("Assistant")
    st.caption("Answers only from approved company documents. Always shows sources and confidence.")

    if not st.session_state.history:
        st.markdown("**Try asking:**")
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, q in zip(cols, SAMPLE_QUESTIONS):
            if col.button(q, use_container_width=True):
                st.session_state.pending_question = q

    # A plain text_input + separate button doesn't submit on Enter -- Enter
    # just reruns the script without registering a click. st.form does,
    # since Enter inside a form submits it the same as its submit button.
    with st.form("assistant_ask_form", clear_on_submit=True):
        search_col, ask_col = st.columns([6, 1])
        question = search_col.text_input(
            "Ask a question",
            key="assistant_search_input",
            placeholder="Ask a sales or application question…",
            label_visibility="collapsed",
        )
        asked = ask_col.form_submit_button("Ask", type="primary", use_container_width=True)

    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        asked = True
    if question:
        question = question.strip()

    if asked and question:
        _ask_and_record(question)

    for idx, turn in reversed(list(enumerate(st.session_state.history))):
        st.divider()
        st.markdown(f"**{turn['question']}**")
        st.markdown(theme.render_badges_html(turn["confidence"], turn["risk"]), unsafe_allow_html=True)

        with st.container(border=True):
            st.write(turn["answer"])
            with st.expander("Sources"):
                for name, section, text in turn["sources"]:
                    st.markdown(f"- `{name}` — {section}")
                    if text:
                        st.markdown(f"> {text}")

        acol1, acol2, acol3, acol4 = st.columns(4)
        if acol1.button("Correct", key=f"ent-ok-{idx}", icon=":material/thumb_up:"):
            record_feedback(turn["question"], turn["answer"], "correct")
            st.toast("Thanks for the feedback!")
        if acol2.button("Wrong", key=f"ent-bad-{idx}", icon=":material/thumb_down:"):
            _report_wrong_dialog(turn["question"], turn["answer"])
        if acol3.button("Unsafe", key=f"ent-unsafe-{idx}", icon=":material/report:"):
            record_feedback(turn["question"], turn["answer"], "unsafe")
            st.toast("Thanks for flagging this — reported for review.")
        with acol4:
            copy_button(turn["answer"], "Copy", key=f"ent-ans-{idx}", icon=True)

        _render_customer_wording(turn, key_prefix=f"ent-cust-{idx}")


# ---------------------------------------------------------------------------
# Navigation.
# Enterprise theme: Admin is always a nav item; render_admin_gate shows a
#   password form in-page until unlocked (see above). Icons use Streamlit's
#   built-in Material Symbols (outline style) instead of emoji, to read as
#   a professional tool rather than a chat app -- classic theme keeps the
#   original emoji untouched.
# Classic theme: unchanged -- Admin only appears in the nav list at all
#   once the sidebar password gate (above) has already passed.
# ---------------------------------------------------------------------------
if theme.is_enterprise_theme():
    icon_assistant, icon_requirements, icon_discovery, icon_sales_aids, icon_admin = (
        ":material/chat:", ":material/description:", ":material/travel_explore:",
        ":material/article:", ":material/admin_panel_settings:",
    )
else:
    icon_assistant, icon_requirements, icon_discovery, icon_sales_aids, icon_admin = (
        "💬", "📋", "🧭", "📄", "🛠️",
    )

pages = [
    st.Page(render_assistant_page, title="Assistant", icon=icon_assistant, default=True),
    st.Page(render_requirements_page, title="Customer Requirements", icon=icon_requirements),
    st.Page(render_discovery_page, title="Discovery Questions", icon=icon_discovery),
    st.Page(render_sales_aid_page, title="Sales Aids", icon=icon_sales_aids),
]
if theme.is_enterprise_theme():
    pages.append(st.Page(render_admin_gate, title="Admin", icon=icon_admin))
elif st.session_state.admin_authenticated:
    pages.append(st.Page(render_admin_page, title="Admin", icon=icon_admin))

st.navigation(pages).run()
