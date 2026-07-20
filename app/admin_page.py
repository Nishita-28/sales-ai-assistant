"""Admin page: document management, index rebuilds, and claims-policy
editing. Only reachable once app/streamlit_app.py has added it to
st.navigation() -- which it only does after the sidebar password gate
passes -- but render_admin_page() re-checks session state itself too, in
case this module is ever called from somewhere that skips that gate.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.claims_store import load_claims, save_claims
from app.restricted_policy import KNOWN_CATEGORIES, PolicyEntry, load_entries, save_entries
from app.retriever import (
    RetrieverError,
    SUPPORTED_DOC_EXTENSIONS,
    build_index,
    index_size,
    load_and_chunk_approved_docs,
)

APPROVED_DOCS_DIR = Path("data/approved_docs")
REMOVED_DOCS_DIR = Path("data/removed_docs")
APPROVED_CLAIMS_PATH = Path("data/approved_claims.md")
RESTRICTED_CLAIMS_PATH = Path("data/restricted_claims.yaml")


# ---------------------------------------------------------------------------
# Rebuild Index status bar (shown above every tab, not just Documents --
# rebuilding only makes sense in light of the current document list)
# ---------------------------------------------------------------------------
def _approved_doc_paths() -> list[Path]:
    """Only files load_and_chunk_approved_docs() will actually pick up --
    an unsupported file sitting in the folder (e.g. the .xlsx whose
    content is separately covered by a .csv copy) is real on disk but
    contributes zero chunks, so listing it here would misrepresent what's
    actually indexed."""
    if not APPROVED_DOCS_DIR.exists():
        return []
    return sorted(
        p
        for p in APPROVED_DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
    )


def _render_rebuild_status() -> None:
    if "docs_changed_since_rebuild" not in st.session_state:
        st.session_state.docs_changed_since_rebuild = False
    if "last_rebuilt_at" not in st.session_state:
        st.session_state.last_rebuilt_at = None

    doc_count = len(_approved_doc_paths())
    chunk_count = index_size()

    status = f"**{chunk_count}** chunks indexed from **{doc_count}** documents."
    status += (
        f" Last rebuilt {st.session_state.last_rebuilt_at.strftime('%H:%M:%S')}."
        if st.session_state.last_rebuilt_at
        else " Not rebuilt this session."
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        if st.session_state.docs_changed_since_rebuild:
            st.warning(status + " Documents have changed since the last rebuild.")
        else:
            st.info(status)
    with col2:
        if st.button("Rebuild Index Now", type="primary", use_container_width=True):
            with st.spinner("Rebuilding index -- this can take a minute..."):
                try:
                    chunks = load_and_chunk_approved_docs(APPROVED_DOCS_DIR)
                    count = build_index(chunks)
                except RetrieverError as e:
                    st.error(f"Rebuild failed: {e}")
                else:
                    st.session_state.last_rebuilt_at = datetime.now()
                    st.session_state.docs_changed_since_rebuild = False
                    st.success(f"Indexed {count} chunks from {doc_count} documents.")
                    st.rerun()


# ---------------------------------------------------------------------------
# Documents tab
# ---------------------------------------------------------------------------
@st.dialog("Remove document?")
def _confirm_remove_dialog(doc_path: Path) -> None:
    st.write(
        f"Remove **{doc_path.name}** from the knowledge base? "
        "It will be moved to `data/removed_docs/`, not permanently deleted."
    )
    col1, col2 = st.columns(2)
    if col1.button("Remove", type="primary", use_container_width=True):
        REMOVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(doc_path), str(REMOVED_DOCS_DIR / doc_path.name))
        st.session_state.docs_changed_since_rebuild = True
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


def _render_documents_tab() -> None:
    st.subheader("Add a document")
    uploaded_file = st.file_uploader(
        "Upload a new approved document",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_DOC_EXTENSIONS),
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        target_path = APPROVED_DOCS_DIR / uploaded_file.name
        if target_path.exists():
            st.warning(
                f"'{uploaded_file.name}' already exists in the knowledge base. "
                "Remove the existing file first to replace it."
            )
        elif st.button("Add to knowledge base", type="primary"):
            APPROVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(uploaded_file.getbuffer())
            st.session_state.docs_changed_since_rebuild = True
            st.success(f"Added {uploaded_file.name}. Rebuild the index above to include it.")
            st.rerun()

    st.subheader("Current documents")
    doc_paths = _approved_doc_paths()

    if not doc_paths:
        st.caption("No documents in the knowledge base yet.")
        return

    for doc_path in doc_paths:
        stat = doc_path.stat()
        size_kb = stat.st_size / 1024
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{doc_path.name}**")
            st.caption(f"{size_kb:.0f} KB · modified {modified}")
        with col2:
            if st.button("Remove", key=f"remove-{doc_path.name}", use_container_width=True):
                _confirm_remove_dialog(doc_path)


# ---------------------------------------------------------------------------
# Approved Claims tab -- plain reference documentation. Nothing in
# app/claim_checker.py reads this file; it's for human review only (spec
# 8.2/12.3), so a flat prose bullet list stays the right editor for it.
# ---------------------------------------------------------------------------
def _render_approved_claims_tab() -> None:
    header, bullets = load_claims(APPROVED_CLAIMS_PATH)

    edited = st.data_editor(
        pd.DataFrame({"Claim": bullets}),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor-approved-claims",
    )

    if st.button("Save Approved Claims", type="primary"):
        new_bullets = [str(v) for v in edited["Claim"].tolist()]
        save_claims(APPROVED_CLAIMS_PATH, header, new_bullets)
        st.success("Saved Approved Claims.")
        st.rerun()


# ---------------------------------------------------------------------------
# Restricted Claims tab -- this IS what app/claim_checker.py enforces
# (loaded from data/restricted_claims.yaml, see app/restricted_policy.py),
# so it needs the structured fields enforcement actually uses, not just
# free text: category, the keyword(s) that trigger it, and whether a match
# is always blocked regardless of retrieved evidence.
# ---------------------------------------------------------------------------
def _entries_to_rows(entries: list[PolicyEntry]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": [e.category for e in entries],
            "keywords": [", ".join(e.keywords) for e in entries],
            "always_unsupported": [e.always_unsupported for e in entries],
            "note": [e.note for e in entries],
        }
    )


def _rows_to_entries(df: pd.DataFrame) -> list[PolicyEntry]:
    entries = []
    for row in df.itertuples(index=False):
        keywords = [k.strip() for k in str(row.keywords).split(",") if k.strip()]
        entries.append(
            PolicyEntry(
                category=row.category,
                keywords=keywords,
                always_unsupported=bool(row.always_unsupported),
                note=row.note or "",
            )
        )
    return entries


def _render_restricted_claims_tab() -> None:
    st.caption(
        "This list is what the assistant actually enforces -- edits here take "
        "effect immediately, no code changes or restart needed."
    )

    entries = load_entries(RESTRICTED_CLAIMS_PATH)

    edited = st.data_editor(
        _entries_to_rows(entries),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor-restricted-claims",
        column_config={
            "category": st.column_config.SelectboxColumn(options=KNOWN_CATEGORIES, required=True),
            "keywords": st.column_config.TextColumn(help="Comma-separated, e.g. \"atex, atex certified\""),
            "always_unsupported": st.column_config.CheckboxColumn(
                help="Block a match regardless of retrieved evidence (for flat denials, e.g. ATEX)."
            ),
            "note": st.column_config.TextColumn(help="Shown here only -- not used for matching."),
        },
    )

    if st.button("Save Restricted Claims", type="primary"):
        save_entries(RESTRICTED_CLAIMS_PATH, _rows_to_entries(edited))
        st.success("Saved Restricted Claims. Enforcement updated immediately.")
        st.rerun()


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------
def render_admin_page() -> None:
    if not st.session_state.get("admin_authenticated"):
        st.error("Sign in as admin from the sidebar to access this page.")
        st.stop()

    st.title("Admin")
    st.caption("Document management, index rebuilds, and claim policy editing.")

    _render_rebuild_status()

    documents_tab, approved_tab, restricted_tab = st.tabs(
        ["Documents", "Approved Claims", "Restricted Claims"]
    )

    with documents_tab:
        _render_documents_tab()

    with approved_tab:
        _render_approved_claims_tab()

    with restricted_tab:
        _render_restricted_claims_tab()
