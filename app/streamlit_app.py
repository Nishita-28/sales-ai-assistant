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

# `streamlit run app/streamlit_app.py` sets sys.path[0] to this file's own
# directory (app/), not the project root, regardless of the cwd the command
# was run from -- so the `app.*` imports below would fail with "No module
# named 'app'" without this, even though every other module in this project
# resolves them fine (they're run via `python -m app.X`, which puts the cwd
# on sys.path instead).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag_pipeline import answer_question
from app.response_generator import ResponseGeneratorError
from app.retriever import RetrieverError

APPROVED_DOCS_DIR = Path("data/approved_docs")

def get_admin_password() -> str:
    """Reads ADMIN_PASSWORD from st.secrets, then env var, then falls back
    to "admin". TEMPORARY: that fallback is a local-testing placeholder --
    replace it with a real secret before this touches a shared environment."""
    try:
        return st.secrets["ADMIN_PASSWORD"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.environ.get("ADMIN_PASSWORD", "admin")


ADMIN_PASSWORD = get_admin_password()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Internal AI Sales Assistant",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {question, answer, sources, confidence, risk, customer_wording}

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

SAMPLE_QUESTIONS = [
    "What gases are we targeting?",
    "Can we detect hydrogen?",
    "Do we support RS485?",
    "Can we say this is ATEX certified?",
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
    # Sanitize the DOM id so it can't break out of the id="..." / JS string it's used in.
    safe_id = "copy-btn-" + re.sub(r"[^a-zA-Z0-9_-]", "", key)
    # json.dumps gives a JS-safe quoted string; html.escape then makes it
    # safe inside a double-quoted HTML attribute (onclick="...").
    safe_text = html.escape(json.dumps(text))
    safe_label = html.escape(label)
    # components.html is deprecated as of Streamlit 1.56; st.iframe replaces it.
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
# Sidebar: knowledge base status + admin upload (per spec 12.4 - optional)
# ---------------------------------------------------------------------------
def count_approved_docs() -> int:
    """Counts files in data/approved_docs. Swap for your indexer's real doc count if it tracks one already."""
    if not APPROVED_DOCS_DIR.exists():
        return 0
    return sum(1 for p in APPROVED_DOCS_DIR.iterdir() if p.is_file())


with st.sidebar:
    st.subheader("Knowledge base")

    st.markdown("**Status**")
    st.success(f"Indexed docs: {count_approved_docs()}")
    st.info("Claim guardrail: ON")
    st.warning("Mode: Internal only")

    with st.expander("Admin: upload / reindex"):
        if not ADMIN_PASSWORD:
            st.error("ADMIN_PASSWORD is not set. Add it to secrets before deploying.")
        elif st.session_state.admin_authenticated:
            st.caption("Signed in as admin.")
            uploaded = st.file_uploader("Add approved document", type=["docx", "pdf", "pptx", "csv", "md"])
            if st.button("Rebuild index"):
                st.toast("Index rebuild triggered (wire this to rag_pipeline --reindex)")
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
# Main area
# ---------------------------------------------------------------------------
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
for turn in st.session_state.history:
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

        copy_button(turn["answer"], "Copy answer", key=f"ans-{turn['question'][:10]}")

        if turn["customer_wording"]:
            with st.expander("Customer-facing wording"):
                st.write(turn["customer_wording"])
                copy_button(turn["customer_wording"], "Copy customer wording", key=f"cust-{turn['question'][:10]}")
        else:
            st.caption("Customer-facing wording blocked — escalate for review.")

        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.button("Correct", key=f"ok-{turn['question'][:10]}")
        fcol2.button("Wrong", key=f"bad-{turn['question'][:10]}")
        fcol3.button("Unsafe", key=f"unsafe-{turn['question'][:10]}")

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
