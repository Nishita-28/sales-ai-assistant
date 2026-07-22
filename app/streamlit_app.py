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

from app.admin_page import render_admin_page
from app.claim_checker import warm_up as warm_up_claim_checker
from app.feedback_store import record_feedback
from app.rag_pipeline import answer_question
from app.response_generator import ResponseGeneratorError
from app.response_generator import warm_up as warm_up_response_generator
from app.retriever import RetrieverError
from app.retriever import warm_up as warm_up_retriever

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
st.set_page_config(
    page_title="Internal AI Sales Assistant",
    layout="centered",
)

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
    st.session_state.history = []  # list of dicts: {question, answer, sources, confidence, risk, customer_wording}

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

SAMPLE_QUESTIONS = [
    "Can this be used to monitor hydrogen buildup in a lead-acid battery room?",
    "Is the FIXaHY sensor PESO approved for hazardous areas?",
    "Can I integrate this with our existing SCADA system over RS485 or 4-20mA?",
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
def copy_button(text: str, label: str = "Copy answer", key: str = ""):
    # Sanitize so the id can't break out of the id="..." / JS string it's used in.
    safe_id = "copy-btn-" + re.sub(r"[^a-zA-Z0-9_-]", "", key)
    # json.dumps gives a JS-safe quoted string; html.escape makes it safe
    # inside the double-quoted onclick="..." attribute.
    safe_text = html.escape(json.dumps(text))
    safe_label = html.escape(label)
    st.iframe(
        f"""
        <button id="{safe_id}" onclick="
            navigator.clipboard.writeText({safe_text});
            const btn = document.getElementById('{safe_id}');
            btn.innerText = 'Copied';
            setTimeout(() => btn.innerText = '{safe_label}', 1500);
        " style="
            padding:6px 12px; border-radius:6px; border:1px solid #ccc;
            background:#fff; cursor:pointer; font-size:13px; font-family:sans-serif;
        ">{safe_label}</button>
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
                for name, section in turn["sources"]:
                    st.markdown(f"- `{name}` — {section}")

            copy_button(turn["answer"], "Copy answer", key=f"ans-{idx}")

            if turn["customer_wording"]:
                with st.expander("Customer-facing wording"):
                    st.write(turn["customer_wording"])
                    copy_button(turn["customer_wording"], "Copy customer wording", key=f"cust-{idx}")
            else:
                st.caption("Customer-facing wording blocked — escalate for review.")

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
        with st.spinner("Checking approved documents..."):
            try:
                result = answer_question(question, want_customer_wording=True)
            except (RetrieverError, ResponseGeneratorError) as e:
                st.error(f"Something went wrong answering that question: {e}")
            else:
                st.session_state.history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "confidence": result["confidence"],
                    "risk": result["risk_flag"],
                    "customer_wording": result["customer_wording"],
                })
                st.rerun()


# ---------------------------------------------------------------------------
# Navigation -- Admin page only appears once the sidebar password gate passes.
# ---------------------------------------------------------------------------
pages = [st.Page(render_assistant_page, title="Assistant", icon="💬", default=True)]
if st.session_state.admin_authenticated:
    pages.append(st.Page(render_admin_page, title="Admin", icon="🛠️"))

st.navigation(pages).run()
