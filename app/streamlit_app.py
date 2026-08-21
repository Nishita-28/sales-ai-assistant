#cd sales-ai-assistant
#streamlit run app/streamlit_app.py
"""Internal AI Sales Assistant - Streamlit frontend."""

import html
import json
import os
import re
import sys
import time
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
from app.background_jobs import get_active_jobs
from app.claim_checker import warm_up as warm_up_claim_checker
from app.db import DatabaseUnavailableError, is_postgres_enabled
from app.discovery_page import render_discovery_page
from app.document_storage import sync_local_docs_from_postgres
from app.feedback_store import record_feedback
from app.rag_pipeline import (
    finalize_customer_wording_question,
    finalize_streamed_answer,
    stream_answer_question,
    stream_customer_wording_question,
)
from app.registry_builder import rebuild_product_registry
from app.requirements_page import render_requirements_page
from app.response_generator import ResponseGeneratorError
from app.response_generator import warm_up as warm_up_response_generator
from app.retriever import RetrieverError
from app.retriever import build_index, load_and_chunk_approved_docs
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
theme.apply_native_theme_option()
st.set_page_config(
    page_title="Internal AI Sales Assistant",
    layout="centered",
)
theme.inject_theme()

# Cold-start sync from Neon -- Streamlit Cloud's filesystem is ephemeral,
# so a fresh container's local data/approved_docs/ is empty or stale.
# When DATABASE_URL is set, pulls every document from Postgres and rebuilds
# the local Chroma index + product registry, once per server process.
@st.cache_resource(show_spinner="Loading knowledge base -- this can take a couple of minutes on a cold start...")
def _sync_from_neon_on_cold_start() -> None:
    if not is_postgres_enabled():
        return
    try:
        doc_count = sync_local_docs_from_postgres(APPROVED_DOCS_DIR)
        if doc_count > 0:
            chunks = load_and_chunk_approved_docs(APPROVED_DOCS_DIR)
            build_index(chunks)
            rebuild_product_registry(APPROVED_DOCS_DIR)
    except Exception as e:
        # Not swallowed like _warm_up_backend below -- an admin needs to know
        # the assistant may be running with an empty/stale knowledge base.
        st.error(
            f"Failed to sync documents from the database on startup: {e}. "
            "The assistant may have no knowledge base until this is resolved."
        )


_sync_from_neon_on_cold_start()


# ---------------------------------------------------------------------------
# Backend warm-up -- initializes API clients, the vector store connection,
# and cached files once per server process, so the first real question
# doesn't pay for setup that could happen at startup instead.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting up...")
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


@st.fragment(run_every="3s")
def _render_background_job_status() -> None:
    """Streamlit can't push updates from a background thread into a browser
    session, so this fragment polls the shared in-process status dict on a
    short timer instead, independent of whatever page is active."""
    for job in get_active_jobs():
        if job.status == "running":
            st.info(f"Removing **{job.label}**...")
        elif job.status == "done":
            st.success(f"Removed **{job.label}**.")
        else:
            st.error(f"Failed to remove **{job.label}**: {job.error}")


with st.sidebar:
    # Logo is positioned above the nav via CSS (theme.py) since Streamlit's
    # auto nav list is a DOM sibling that can't be interleaved with content
    # from `with st.sidebar:`.
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
    _render_background_job_status()


def render_admin_gate() -> None:
    """Admin as a regular nav page: shows a password form in the page
    content itself until unlocked."""
    if st.session_state.admin_authenticated:
        render_admin_page()
        return

    st.title("Admin")
    if not ADMIN_PASSWORD:
        st.error("ADMIN_PASSWORD is not set. Add it to secrets before deploying.")
        return

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
def _ask_and_record(question: str) -> None:
    """Runs one question through the pipeline and appends it to history.
    Streams the answer live as it's generated, echoing the question first
    so the live text doesn't appear with no visible prompt above it.
    Customer-facing wording is generated on demand (see
    _render_customer_wording()), not eagerly here, since most questions
    never need it and it roughly doubles generation time."""
    try:
        with st.spinner("Checking approved documents..."):
            retrieval, text_stream = stream_answer_question(question)
    except (RetrieverError, ResponseGeneratorError) as e:
        st.error(f"Something went wrong answering that question: {e}")
        return

    st.divider()
    st.markdown(f"**{question}**")
    answer_wrapper = st.container(border=True)

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
    """Shows the blocked caption, a button to generate customer-facing
    wording on demand, or the generated wording once it exists. Mutates
    turn["customer_wording"] in place so it's cached in
    st.session_state.history and not regenerated on every rerun."""
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
                icon=True,
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


def render_assistant_page() -> None:
    """Search-bar-at-top, answer-card-below layout -- deliberately not a
    chat thread. Most recent answer is shown first, like a search/report
    tool rather than an accumulating chat log."""
    st.title("Assistant")

    # Rendered into an explicit st.empty() placeholder so it can be cleared
    # without an st.rerun(). Must call .empty() AFTER the `with` block
    # exits, not from inside it -- clearing a placeholder while still
    # writing into it corrupts Streamlit's render tree and crashes blank.
    try_asking_slot = st.empty()
    clicked_sample_question = None
    if not st.session_state.history and "pending_question" not in st.session_state:
        with try_asking_slot.container():
            st.markdown("**Try asking:**")
            cols = st.columns(len(SAMPLE_QUESTIONS))
            for col, q in zip(cols, SAMPLE_QUESTIONS):
                if col.button(q, width="stretch"):
                    clicked_sample_question = q
    if clicked_sample_question:
        st.session_state.pending_question = clicked_sample_question
        try_asking_slot.empty()

    # st.form so Enter submits the same as clicking the button.
    with st.form("assistant_ask_form", clear_on_submit=True):
        search_col, ask_col = st.columns([6, 1])
        question = search_col.text_input(
            "Ask a question",
            key="assistant_search_input",
            placeholder="Ask a sales or application question…",
            label_visibility="collapsed",
        )
        asked = ask_col.form_submit_button("Ask", type="primary", width="stretch")

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
                for name in dict.fromkeys(name for name, _, _ in turn["sources"]):
                    st.markdown(f"- `{name}`")

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
# Navigation. Admin is always a nav item; render_admin_gate shows a
# password form in-page until unlocked (see above).
# ---------------------------------------------------------------------------
icon_assistant, icon_requirements, icon_discovery, icon_sales_aids, icon_admin = (
    ":material/chat:", ":material/description:", ":material/travel_explore:",
    ":material/article:", ":material/admin_panel_settings:",
)

def _friendly_error(e: Exception) -> None:
    st.error(
        "Something went wrong loading this page -- please refresh. This is "
        "usually temporary (the database waking back up after being idle). "
        "If it keeps happening, let an admin know."
    )
    with st.expander("Technical detail"):
        st.code(f"{type(e).__name__}: {e}")


def _safe_page(render_fn):
    """Wraps a page function so an uncaught exception shows a plain refresh
    message instead of Streamlit's raw traceback. Needed because
    st.navigation(pages).run() does not let an exception raised inside the
    selected page function propagate out to a try/except around the .run()
    call itself -- that outer try/except (below) is only a second, harmless
    layer of defense, not sufficient on its own."""

    def wrapped() -> None:
        try:
            render_fn()
        except DatabaseUnavailableError as e:
            _friendly_error(e)
        except Exception as e:  # noqa: BLE001 -- last resort, any page bug included
            _friendly_error(e)

    # st.Page infers each page's URL pathname from the callable's __name__
    # when not otherwise unique -- every wrapped() closure shares that name
    # by default, which Streamlit refuses to start with (5 identical
    # pathnames). Restoring the original name keeps each pathname distinct.
    wrapped.__name__ = render_fn.__name__
    return wrapped


pages = [
    st.Page(_safe_page(render_assistant_page), title="Assistant", icon=icon_assistant, default=True),
    st.Page(_safe_page(render_requirements_page), title="Customer Requirements", icon=icon_requirements),
    st.Page(_safe_page(render_discovery_page), title="Discovery Questions", icon=icon_discovery),
    st.Page(_safe_page(render_sales_aid_page), title="Sales Aids", icon=icon_sales_aids),
    st.Page(_safe_page(render_admin_gate), title="Admin", icon=icon_admin),
]

# Kept as a second, harmless layer of defense -- see _safe_page's docstring
# for why this alone isn't enough to catch a page-level exception.
try:
    st.navigation(pages).run()
except DatabaseUnavailableError as e:
    _friendly_error(e)
