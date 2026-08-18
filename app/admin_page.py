"""Admin page: document management, index rebuilds, and claims-policy
editing. render_admin_page() re-checks the sign-in state itself, in case
it's ever reached without going through the sidebar gate first.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app import theme
from app.background_jobs import start_job
from app.claims_store import load_claims, save_claims
from app.deals_store import MEDDPICC_FIELDS, delete_deal, get_recommendation, list_deals, missing_fields
from app.document_types import (
    ALL_TYPES,
    get_document_type,
    remove_document_type,
    set_document_type,
)
from app.feedback_store import (
    clear_all_feedback,
    clear_feedback_before,
    count_feedback,
    daily_feedback_counts,
    delete_feedback,
    list_all_feedback,
    most_reported_question,
    recent_reports,
    resolve_feedback,
)
from app import product_field_overrides
from app.concentration import ConcentrationRange
from app.product_index import NomenclatureSegment, load_product_index
from app.registry_builder import rebuild_product_registry
from app.requirements_fields import FIELD_TYPES, load_fields, save_fields, unique_key
from app.requirements_store import delete_requirement, list_requirements, list_requirements_for_deal
from app.sales_aid_store import get_result, list_sales_aids_for_deal
from app.restricted_policy import KNOWN_CATEGORIES, PolicyEntry, load_entries, save_entries
from app.retriever import (
    RetrieverError,
    SUPPORTED_DOC_EXTENSIONS,
    add_document_to_index,
    build_index,
    index_size,
    load_and_chunk_approved_docs,
    remove_document_from_index,
)

APPROVED_DOCS_DIR = Path("data/approved_docs")
REMOVED_DOCS_DIR = Path("data/removed_docs")
APPROVED_CLAIMS_PATH = Path("data/approved_claims.md")
RESTRICTED_CLAIMS_PATH = Path("data/restricted_claims.yaml")


# ---------------------------------------------------------------------------
# Rebuild Index status bar (shown above every tab)
# ---------------------------------------------------------------------------
def _approved_doc_paths() -> list[Path]:
    """Only files that are actually indexable -- an unsupported file in the
    folder is real on disk but contributes zero chunks. Also excludes
    Word/Excel/PowerPoint's own "~$..." lock files -- proven necessary
    by direct feedback: opening one of the real approved .docx files in
    Word (to review or edit it) drops a same-named "~$..." lock file
    right into this same folder, which was being listed as if it were
    a real 14th document (0 KB, unclassified) -- confusing on its own,
    and removing it could leave the "Remove document?" dialog open
    afterward, since the lock file was never a real, chunkable document
    in the first place and reindexing "removal" of something that was
    never really indexed could error."""
    if not APPROVED_DOCS_DIR.exists():
        return []
    return sorted(
        p
        for p in APPROVED_DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_DOC_EXTENSIONS and not p.name.startswith("~$")
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
        if st.button(
            "Rebuild Index Now",
            type="primary",
            use_container_width=True,
            help=(
                "Rebuild is automatic after adding, removing, or editing a document. "
                "Use this after a failed rebuild or when changing an existing document's Type."
            ),
        ):
            with st.spinner("Rebuilding index -- this can take a minute..."):
                try:
                    chunks = load_and_chunk_approved_docs(APPROVED_DOCS_DIR)
                    count = build_index(chunks)
                    rebuild_product_registry(APPROVED_DOCS_DIR)
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
        # The file move + type removal are near-instant, so they happen
        # right here -- the document disappears from "Current documents"
        # immediately. The slow part (removing it from the vector index,
        # then a full registry rebuild -- ~a minute, see
        # registry_builder.rebuild_product_registry's own docstring) runs
        # on a background thread instead (see app.background_jobs), so
        # this dialog can close immediately rather than sitting on a
        # blocking spinner. Status shows in the sidebar on every page
        # until it finishes (see streamlit_app.py's sidebar fragment).
        REMOVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        doc_name = doc_path.name
        shutil.move(str(doc_path), str(REMOVED_DOCS_DIR / doc_name))
        remove_document_type(doc_name)

        def _finish_removal(doc_name: str = doc_name) -> None:
            remove_document_from_index(doc_name)
            rebuild_product_registry(APPROVED_DOCS_DIR)

        start_job(f"remove-{doc_name}", doc_name, _finish_removal)
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


_TYPE_HELP = {
    "Product Catalogue": "One specific product's exact specs and ordering options.",
    "Use Case Guide": "Which industries and applications MNST's products serve.",
    "Technical Guide": "General background on how a sensing technology works.",
    "Historical Sales Record": "A log of past deals, used as supporting evidence.",
    "Competitive & Internal Strategy": "Competitor comparisons and internal sales positioning.",
    "Golden Frameworks & Sales Tactics": "Qualification frameworks and sales tactics for internal coaching.",
    "Other": "Doesn't fit the categories above yet.",
}

# Which page(s) each type feeds, read off the real routing logic (app/
# document_types.py's MAIN_ASSISTANT_EXCLUDED_TYPES, is_product_catalogue,
# and discovery_generator.py's methodology/recommendable scoping) rather
# than restated from memory, so this can't quietly drift out of sync with
# what the code does -- kept plain (just the page list) per explicit
# feedback that the routing rationale belongs in code comments, not this
# admin-facing copy. Shown once in a reference expander rather than
# repeated under every document row -- a caption per row was tried for the
# field editor earlier and multiplying per-row elements is exactly what
# previously caused a Streamlit tab-rendering bug on this same page (see
# _render_product_field_row's comment).
_TYPE_USED_BY = {
    "Product Catalogue": "Assistant, Discovery Questions, Sales Aids, Customer Requirements",
    "Use Case Guide": "Assistant, Discovery Questions, Sales Aids",
    "Technical Guide": "Assistant, Discovery Questions, Sales Aids",
    "Historical Sales Record": "Assistant, Discovery Questions, Sales Aids",
    "Competitive & Internal Strategy": "Discovery Questions, Sales Aids",
    "Golden Frameworks & Sales Tactics": "Discovery Questions, Sales Aids",
    "Other": "Assistant, Discovery Questions, Sales Aids",
}


def _render_upload_result() -> None:
    """Persists the outcome of the last upload/re-index across the
    st.rerun() that follows it -- a plain st.success()/st.warning() right
    before st.rerun() would only flash briefly, easy to miss for
    something correctness-critical like a structural-review warning.
    Stays visible until the admin dismisses it or triggers another
    upload/save, which overwrites it."""
    result = st.session_state.get("last_upload_result")
    if not result:
        return
    name, count, doc_type, warnings = result
    count_part = f" ({count} chunks)" if count is not None else ""
    type_part = f" as {doc_type}" if doc_type else ""
    if warnings:
        st.warning(
            f"Indexed **{name}**{count_part}{type_part}, but with issues found during extraction:\n\n"
            + "\n".join(f"- {w}" for w in warnings)
        )
    else:
        st.success(f"Indexed {name}{count_part}{type_part}. No extraction issues found.")
    if st.button("Dismiss", key="dismiss-upload-result"):
        del st.session_state["last_upload_result"]
        st.rerun()


def _render_documents_tab() -> None:
    _render_rebuild_status()
    _render_upload_result()

    st.subheader("Add a document")
    st.caption("Upload a document to add it to the knowledge base.")
    with st.expander("ⓘ Supported formats & extraction notes"):
        st.caption(
            f"Supported: {', '.join(sorted(ext.lstrip('.') for ext in SUPPORTED_DOC_EXTENSIONS))}. "
            "A document with very complex tables -- multiple header levels, or more than one logical "
            "section combined into a single physical table -- may not extract cleanly even though it "
            "uploads successfully; you'll get a warning below if that happens, but simpler, flatter "
            "tables are more reliable."
        )
    # Keyed with a version counter, bumped on a successful add -- otherwise
    # the uploader keeps showing the just-added file "staged" after the
    # rerun (same stale-widget-state issue as the data_editor key fix
    # above), which reads as if the upload never actually went through.
    if "doc_uploader_version" not in st.session_state:
        st.session_state.doc_uploader_version = 0

    uploaded_file = st.file_uploader(
        "Upload a new approved document",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_DOC_EXTENSIONS),
        label_visibility="collapsed",
        key=f"doc-uploader-{st.session_state.doc_uploader_version}",
    )
    if uploaded_file is not None:
        target_path = APPROVED_DOCS_DIR / uploaded_file.name
        if target_path.exists():
            st.warning(
                f"'{uploaded_file.name}' already exists in the knowledge base. "
                "Remove the existing file first to replace it."
            )
        else:
            doc_type = st.selectbox("Document type", ALL_TYPES, key="new-doc-type")
            st.caption(_TYPE_HELP[doc_type])
            if st.button("Add to knowledge base", type="primary"):
                APPROVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(uploaded_file.getbuffer())
                set_document_type(uploaded_file.name, doc_type)
                with st.spinner("Indexing new document..."):
                    try:
                        count, warnings = add_document_to_index(target_path)
                        rebuild_product_registry(APPROVED_DOCS_DIR)
                    except RetrieverError as e:
                        st.session_state.docs_changed_since_rebuild = True
                        st.error(f"Added the file, but indexing failed: {e}. Use Rebuild Index to retry.")
                    else:
                        st.session_state.last_upload_result = (uploaded_file.name, count, doc_type, warnings)
                        st.session_state.doc_uploader_version += 1
                        st.rerun()

    st.subheader("Current documents")
    doc_paths = _approved_doc_paths()

    if not doc_paths:
        st.caption("No documents in the knowledge base yet.")

    for doc_path in doc_paths:
        stat = doc_path.stat()
        size_kb = stat.st_size / 1024
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        current_type = get_document_type(doc_path.name)
        # [3, 3, 1], not the original [4, 2, 1] -- proven necessary by
        # direct feedback: the Type dropdown was too narrow to read its
        # own longest options ("Golden Frameworks & Sales Tactics",
        # "Competitive & Internal Strategy") even once a choice was
        # already selected, let alone while open.
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            st.markdown(f"**{doc_path.name}**")
            st.caption(f"{size_kb:.0f} KB · modified {modified}")
        with col2:
            type_options = ALL_TYPES if current_type is not None else ["(unclassified)"] + ALL_TYPES
            default_index = type_options.index(current_type) if current_type is not None else 0
            chosen = st.selectbox(
                "Type", type_options, index=default_index, key=f"type-{doc_path.name}", label_visibility="collapsed"
            )
            if chosen != "(unclassified)" and chosen != current_type:
                set_document_type(doc_path.name, chosen)
                st.rerun()
        with col3:
            if st.button("Remove", key=f"remove-{doc_path.name}", use_container_width=True):
                _confirm_remove_dialog(doc_path)

        # A plain-text edit box for a .md document -- e.g. tactics/
        # framework reference content that's genuinely just prose to
        # tweak, not a .docx catalogue with tables/nomenclature that
        # needs a real document editor. Saving re-indexes just this one
        # document, so an edit takes effect immediately.
        if doc_path.suffix.lower() == ".md":
            with st.expander(f"Edit {doc_path.name}"):
                edited_text = st.text_area(
                    "Content",
                    value=doc_path.read_text(encoding="utf-8"),
                    height=300,
                    key=f"md-edit-{doc_path.name}",
                    label_visibility="collapsed",
                )
                if st.button("Save changes", key=f"md-save-{doc_path.name}", type="primary"):
                    doc_path.write_text(edited_text, encoding="utf-8")
                    with st.spinner("Re-indexing..."):
                        try:
                            remove_document_from_index(doc_path.name)
                            _, warnings = add_document_to_index(doc_path)
                            rebuild_product_registry(APPROVED_DOCS_DIR)
                        except RetrieverError as e:
                            st.session_state.docs_changed_since_rebuild = True
                            st.error(f"Saved the file, but re-indexing failed: {e}. Use Rebuild Index to retry.")
                        else:
                            st.session_state.last_upload_result = (doc_path.name, None, None, warnings)
                            st.rerun()

    if any(get_document_type(p.name) is None for p in doc_paths):
        st.warning(
            "Some documents above have no assigned type yet (likely added before this classification existed, "
            "or via a script that bypassed this form). They default to open/reference behavior until classified -- "
            "pick a type for each from the dropdown."
        )

    with st.expander("What does each document type feed?"):
        for doc_type in ALL_TYPES:
            st.markdown(f"**{doc_type}** -- {_TYPE_HELP[doc_type]}")
            st.caption(f"Used by: {_TYPE_USED_BY[doc_type]}")


# ---------------------------------------------------------------------------
# Approved Claims tab -- indexed alongside the real approved documents (see
# retriever.load_and_chunk_approved_docs), so this content is retrievable
# and can inform an answer. It's still not the same as Restricted Claims:
# it's admin-typed text, not independently verified against the approved
# documents, so it can't on its own satisfy a restricted-category claim
# (pricing, certifications, safety, delivery) -- see
# claim_checker.guardrail_source_text.
# ---------------------------------------------------------------------------
def _render_approved_claims_tab() -> None:
    header, bullets = load_claims(APPROVED_CLAIMS_PATH)

    # A st.data_editor with a fixed key keeps its own {edited_rows,
    # added_rows, deleted_rows} diff in session_state, keyed by row
    # POSITION -- and that diff survives across reruns (including
    # navigating to a different page and back, since session_state
    # outlives any one script run). After Save writes the shorter file
    # and reruns, the stale diff still refers to the old row positions
    # and gets silently re-applied on top of the fresh (already-correct)
    # data on the next render -- which is exactly what made a deleted
    # row appear to "come back". Bumping the key on every successful
    # save forces Streamlit to treat it as a brand-new widget with no
    # carried-over diff, so it starts clean from the just-saved data.
    if "approved_claims_editor_version" not in st.session_state:
        st.session_state.approved_claims_editor_version = 0

    edited = st.data_editor(
        pd.DataFrame({"Claim": bullets}),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"editor-approved-claims-{st.session_state.approved_claims_editor_version}",
    )

    if st.button("Save Approved Claims", type="primary"):
        new_bullets = [str(v) for v in edited["Claim"].tolist()]
        save_claims(APPROVED_CLAIMS_PATH, header, new_bullets)
        with st.spinner("Re-indexing..."):
            try:
                remove_document_from_index(APPROVED_CLAIMS_PATH.name)
                add_document_to_index(APPROVED_CLAIMS_PATH)
            except RetrieverError as e:
                st.session_state.docs_changed_since_rebuild = True
                st.error(f"Saved, but indexing failed: {e}. Use Rebuild Index on the Documents tab to retry.")
            else:
                st.session_state.approved_claims_editor_version += 1
                st.success("Saved Approved Claims.")
                st.rerun()
    # No separate "Remove a claim" selectbox+button here -- the data_editor
    # above already supports row deletion natively (num_rows="dynamic").

    with st.expander("What does this page do?"):
        st.markdown(
            "A curated list of facts already verified against the approved product documents -- "
            "a quick way to add a standalone fact without editing a full document. Saving here "
            "indexes this list, so it can inform the assistant's answers, same as any other "
            "approved document."
        )
        st.caption(
            "It's still not the same as Restricted Claims: a claim in a restricted category "
            "(pricing, certifications, safety, delivery) needs to be backed by a real approved "
            "document to count as supported, not just stated here."
        )


# ---------------------------------------------------------------------------
# Restricted Claims tab -- this is what the assistant actually enforces, so
# it needs structured fields, not just free text.
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

    # Bump the widget key on every save -- see _render_approved_claims_tab's
    # comment for why a fixed key on a dynamic-rows data_editor lets a
    # deleted row silently reappear after a save + rerun (or navigating
    # away and back).
    if "restricted_claims_editor_version" not in st.session_state:
        st.session_state.restricted_claims_editor_version = 0

    edited = st.data_editor(
        _entries_to_rows(entries),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"editor-restricted-claims-{st.session_state.restricted_claims_editor_version}",
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
        st.session_state.restricted_claims_editor_version += 1
        st.success("Saved Restricted Claims. Enforcement updated immediately.")
        st.rerun()

    # No separate "Remove an entry" selectbox+button here -- proven
    # redundant by direct feedback: the data_editor above already lets
    # a row be deleted directly (num_rows="dynamic"), so a second,
    # separate removal control duplicated that with no real benefit.

    with st.expander("What does this page do?"):
        st.markdown(
            "The guardrail policy: each row is a risk **category** (Certification, Pricing, "
            "Safety, ...) plus the **keywords** that trigger it. Whenever a question or answer "
            "matches one of those keywords, the assistant checks whether the retrieved source "
            "documents actually back it up."
        )
        st.markdown(
            "- Matched but **backed by a real document** -- allowed, shown with the category as a risk badge.\n"
            "- Matched but **not backed** -- blocked from customer-facing wording.\n"
            "- **always_unsupported** checked -- always blocked for that keyword, regardless of "
            "any supporting text (for a flat denial, e.g. a certification MNST doesn't hold)."
        )
        st.caption("note is shown here only, for human context -- it isn't used for matching.")


# ---------------------------------------------------------------------------
# Feedback tab -- what the Correct/Wrong/Unsafe buttons in the Assistant
# page actually record.
# ---------------------------------------------------------------------------
@st.dialog("Delete this report?")
def _confirm_delete_report_dialog(report_id: int) -> None:
    st.write("This permanently removes the report from the database. This cannot be undone.")
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary", use_container_width=True):
        delete_feedback(report_id)
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


# Below this many active days of feedback, daily points are too sparse to
# read as a trend (e.g. a single event on a given day swings that day's
# accuracy to a meaningless 0% or 100%) -- fall back to weekly buckets
# instead. Above it, daily is fine-grained enough to be useful.
MIN_ACTIVE_DAYS_FOR_DAILY_CHART = 5

# Need at least this many points on the x-axis for a line to show a trend
# at all -- below it, a chart would just be one dot, so show an empty
# state instead.
MIN_CHART_POINTS = 2


def _render_accuracy_section() -> None:
    """AI Performance: an accuracy trend built from the same Correct/Wrong/
    Unsafe events the Assistant page's feedback buttons record.

    IMPORTANT CAVEAT (this is why the metric is labeled "Feedback Accuracy",
    not "AI Accuracy"): this only reflects the answers a rep bothered to
    rate, not a random sample of every answer given. If feedback is sparse,
    or reps only click a button when something's wrong (a common real-world
    bias -- correct answers rarely get acknowledged), this number will skew
    pessimistic and should not be read as the model's true accuracy.
    """
    rows = daily_feedback_counts()
    if not rows:
        st.info(
            "No feedback recorded yet. Once reps start marking answers Correct, "
            "Wrong, or Unsafe on the Assistant page, accuracy trends will appear here."
        )
        st.divider()
        return

    daily = pd.DataFrame([dict(r) for r in rows])
    daily["day"] = pd.to_datetime(daily["day"])

    correct_total = int(daily["correct"].sum())
    wrong_total = int(daily["wrong"].sum())
    unsafe_total = int(daily["unsafe"].sum())
    total = correct_total + wrong_total + unsafe_total
    feedback_accuracy = (correct_total / total * 100) if total else 0.0

    st.subheader("AI Performance")
    st.caption(
        '"Feedback Accuracy" is the share of *rated* answers marked Correct -- '
        "meaningful only if reps give feedback consistently, not just on failures."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Feedback Accuracy", f"{feedback_accuracy:.0f}%")
    m2.metric("Total Events", total, help="Total Feedback Events")
    m3.metric("Correct", correct_total)
    m4.metric("Wrong", wrong_total)
    m5.metric("Unsafe", unsafe_total)

    active_days = len(daily)
    if active_days >= MIN_ACTIVE_DAYS_FOR_DAILY_CHART:
        chart_source = daily.rename(columns={"day": "period"})
        granularity = "daily"
    else:
        weekly = daily.set_index("day")[["correct", "wrong", "unsafe"]].resample("W-MON").sum()
        weekly = weekly[(weekly["correct"] + weekly["wrong"] + weekly["unsafe"]) > 0]
        chart_source = weekly.reset_index().rename(columns={"day": "period"})
        granularity = "weekly"

    with st.container(border=theme.is_enterprise_theme()):
        st.markdown("**AI Accuracy Over Time**")
        if len(chart_source) < MIN_CHART_POINTS:
            st.caption(
                "Not enough history yet to show a trend -- check back once feedback "
                "has accumulated across more days."
            )
        else:
            chart_source = chart_source.copy()
            chart_source["Accuracy %"] = (
                chart_source["correct"]
                / (chart_source["correct"] + chart_source["wrong"] + chart_source["unsafe"])
                * 100
            )
            st.caption(f"Showing {granularity} feedback accuracy.")
            st.line_chart(
                chart_source.set_index("period")["Accuracy %"],
                color="#184fa3",
                height=260,
            )

    st.divider()


@st.dialog("Clear feedback data?")
def _confirm_clear_feedback_dialog(cutoff_date: Optional[str]) -> None:
    if cutoff_date:
        count = count_feedback(before=cutoff_date)
        st.write(
            f"This permanently deletes {count} feedback event(s) recorded before "
            f"{cutoff_date}. This cannot be undone."
        )
    else:
        count = count_feedback()
        st.write(
            f"This permanently deletes all {count} feedback event(s) and resets the "
            "AI Performance chart above. This cannot be undone."
        )
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary", use_container_width=True, disabled=count == 0):
        if cutoff_date:
            clear_feedback_before(cutoff_date)
        else:
            clear_all_feedback()
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


def _render_feedback_data_management() -> None:
    """Lets an admin edit the data behind the AI Performance chart directly
    -- delete one event, or bulk-clear old/test data -- rather than only
    ever being able to act on active wrong/unsafe reports."""
    with st.expander("Manage feedback data"):
        st.caption(
            "Delete individual events below, or bulk-clear data that's skewing the "
            "chart above (e.g. old test entries)."
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        cutoff = col1.date_input("Clear events recorded before", value=None)
        if col2.button("Clear before date", use_container_width=True, disabled=cutoff is None):
            _confirm_clear_feedback_dialog(cutoff.isoformat())
        if col3.button("Clear all", use_container_width=True):
            _confirm_clear_feedback_dialog(None)

        st.divider()

        rows = list_all_feedback()
        if not rows:
            st.caption("No feedback events recorded.")
            return

        st.caption(f"Most recent {len(rows)} event(s):")
        for row in rows:
            c1, c2, c3 = st.columns([2, 5, 1])
            c1.caption(f"{row['created_at']} · {row['verdict'].capitalize()}")
            c2.caption(row["question"])
            if c3.button(
                "", icon=":material/delete:", key=f"del-fb-{row['id']}", help="Delete this event"
            ):
                delete_feedback(row["id"])
                st.rerun()


def _render_feedback_tab() -> None:
    _render_accuracy_section()
    _render_feedback_data_management()

    reported = most_reported_question()
    if reported:
        st.caption(f'Most reported question ({reported[1]}x): "{reported[0]}"')

    st.subheader("Active Wrong / Unsafe reports")
    reports = recent_reports()
    if not reports:
        st.caption("No active reports.")

    for report in reports:
        with st.container(border=True):
            st.markdown(f"**{report['verdict'].capitalize()}** · {report['created_at']}")
            st.write(f"Q: {report['question']}")
            st.write(f"A: {report['answer']}")
            if report["note"]:
                st.caption(f"Note: {report['note']}")
            col1, col2 = st.columns(2)
            if col1.button(
                "Resolve",
                key=f"resolve-{report['id']}",
                use_container_width=True,
                help="Resolve marks the issue as fixed but keeps it counted in accuracy.",
            ):
                resolve_feedback(report["id"])
                st.rerun()
            if col2.button(
                "Delete",
                key=f"delete-{report['id']}",
                use_container_width=True,
                help="Delete removes the report and excludes it from the accuracy calculation.",
            ):
                _confirm_delete_report_dialog(report["id"])


# ---------------------------------------------------------------------------
# Deals tab -- what Pre-Call Discovery's deal picker actually saves (see
# app/deals_store.py). The only place to view or clean up a deal outside
# of raw database access -- there's no rep-facing "delete a deal" anywhere.
# ---------------------------------------------------------------------------
@st.dialog("Delete this deal?")
def _confirm_delete_deal_dialog(
    deal_id: int, customer_name: str, company: str, linked_requirements: int, linked_sales_aids: int
) -> None:
    warning = (
        f"Permanently delete the deal for **{customer_name}** ({company}), including all captured "
        "MEDDPICC qualification fields? This cannot be undone."
    )
    # deals.db, requirements.db, and sales_aids.db are separate SQLite
    # files -- there's no cross-database foreign key to enforce this, so
    # a rep must be told explicitly rather than the delete silently
    # orphaning rows that still point at a deal_id which no longer exists.
    if linked_requirements:
        plural, verb = ("requirement", "references") if linked_requirements == 1 else ("requirements", "reference")
        warning += (
            f"\n\n**{linked_requirements} captured {plural}** in Customer Requirements still {verb} "
            "this deal and will be orphaned (not deleted, but no longer linked to anything)."
        )
    if linked_sales_aids:
        plural, verb = ("sales aid", "references") if linked_sales_aids == 1 else ("sales aids", "reference")
        warning += (
            f"\n\n**{linked_sales_aids} generated {plural}** still {verb} this deal and will "
            "be orphaned the same way."
        )
    st.write(warning)
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary", use_container_width=True):
        delete_deal(deal_id)
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


def _render_deals_tab() -> None:
    deals = list_deals()
    if not deals:
        st.caption("No deals started yet -- created from the Deal picker on Pre-Call Discovery.")
        return

    for deal in deals:
        with st.container(border=True):
            filled = len(MEDDPICC_FIELDS) - len(missing_fields(deal))
            st.markdown(
                f"**{deal['company']}** -- {deal['customer_name']} · "
                f"{filled}/{len(MEDDPICC_FIELDS)} MEDDPICC fields captured · "
                f"updated {deal['updated_at']}"
            )
            st.caption(deal["use_case"] or "No use case recorded yet.")
            still_missing = [label for _, label, _ in missing_fields(deal)]
            if still_missing:
                st.caption(f"Still missing: {', '.join(still_missing)}")

            recommendation = get_recommendation(deal)
            linked_requirements = list_requirements_for_deal(deal["id"])
            linked_sales_aids = list_sales_aids_for_deal(deal["id"])

            # At-a-glance health row -- proven necessary by testing: a
            # deal with zero requirements and zero sales aids previously
            # showed nothing at all here (both expanders below only
            # render once their count is non-zero), indistinguishable
            # from a deal an admin just hadn't scrolled to yet.
            st.markdown(
                theme.render_deal_status_badges(
                    recommendation["outcome"] if recommendation else None,
                    len(linked_requirements),
                    len(linked_sales_aids),
                ),
                unsafe_allow_html=True,
            )
            if recommendation:
                # .get, not [] -- a recommendation saved before the
                # "product" field existed won't have this key at all.
                product = recommendation.get("product", "")
                outcome_line = f"Last recommendation: {recommendation['outcome']}"
                if product:
                    outcome_line += f" -- {product}"
                st.caption(outcome_line)

            if linked_requirements:
                with st.expander(f"{len(linked_requirements)} linked customer requirement(s)"):
                    for r in linked_requirements:
                        st.caption(f"{r['created_at']} -- {r['product_family'] or 'No product selected'}")

            if linked_sales_aids:
                with st.expander(f"{len(linked_sales_aids)} linked sales aid(s)"):
                    for a in linked_sales_aids:
                        st.caption(f"{a['created_at']} -- {get_result(a).get('title') or 'Untitled'}")

            if st.button("Delete", key=f"delete-deal-{deal['id']}", use_container_width=True):
                _confirm_delete_deal_dialog(
                    deal["id"], deal["customer_name"], deal["company"],
                    len(linked_requirements), len(linked_sales_aids),
                )


# ---------------------------------------------------------------------------
# Customer Requirements tab -- what the Customer Requirements page's form
# actually saves. Submissions are permanent until an admin deletes one here.
# ---------------------------------------------------------------------------
@st.dialog("Delete this requirement?")
def _confirm_delete_requirement_dialog(requirement_id: int, customer_name: str, company: str) -> None:
    st.write(f"Permanently delete the requirement captured for **{customer_name}** ({company})? This cannot be undone.")
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary", use_container_width=True):
        delete_requirement(requirement_id)
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


def _render_requirements_records_tab() -> None:
    rows = list_requirements()
    if not rows:
        st.caption("No customer requirements captured yet.")
        return

    for row in rows:
        with st.container(border=True):
            st.markdown(f"**{row['customer_name']}** -- {row['company']} · {row['created_at']}")
            area = row["installation_area"]
            if row["hazard_zone"]:
                area += f" ({row['hazard_zone']})"
            st.caption(f"{row['application'] or 'No application noted'} | {row['install_type']} | {area}")
            if st.button("Delete", key=f"delete-req-{row['id']}", use_container_width=True):
                _confirm_delete_requirement_dialog(row["id"], row["customer_name"], row["company"])


def _parse_options_text(text: str) -> dict[str, str]:
    """"Name: description" (or just "Name" alone) per line -> {name:
    description}, "Unknown" when no description was given -- same
    sentinel _format_option/_selectable_segments already treat as "show
    the bare name, nothing to add" elsewhere in this codebase."""
    options: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, desc = line.partition(":")
        options[name.strip()] = desc.strip() if sep else "Unknown"
    return options


def _options_to_text(options: dict[str, str]) -> str:
    return "\n".join(name if desc == "Unknown" else f"{name}: {desc}" for name, desc in options.items())


def _admin_selectable_segments(nomenclature) -> list[tuple[str, str, dict[str, str]]]:
    """Local copy of requirements_page._selectable_segments -- every
    nomenclature segment with real decoded values, (segment_code,
    label, {value_code: display text}) tuples. Deliberately NOT
    imported from app.requirements_page here -- that module is a
    Streamlit *page* (registered as an st.Page target in
    streamlit_app.py), and importing from a page module while a
    DIFFERENT page (Admin) is actively rendering was the prime suspect
    for a real, reproducible bug: selecting a product here could leave
    the Admin page showing a mix of stale and new tab content. Whether
    or not that theory is the exact mechanism, duplicating this small,
    pure function avoids the cross-page import entirely, which is a
    safe, low-cost way to rule it out."""
    if not isinstance(nomenclature, dict):
        return []
    result = []
    for seg_code, segment in nomenclature.items():
        if isinstance(segment, NomenclatureSegment) and segment.values:
            display = {
                val_code: (str(v) if isinstance(v, ConcentrationRange) else str(v))
                for val_code, v in segment.values.items()
            }
            result.append((seg_code, segment.label, display))
    return result


def _effective_product_fields(product) -> list[tuple[str, dict[str, str], bool, str]]:
    """(label, options, is_ordering_code, field_type) for every selectable
    field this product currently shows on the real Customer Requirements
    form -- both real ordering-code segments (2+ documented values,
    e.g. Output Signal) and catalogue/admin "additional" ones (e.g.
    Range on a product where Range isn't part of the order code),
    minus anything hidden or edited-away here. is_ordering_code is True
    only for a real, un-edited ordering-code segment -- shown so the
    admin knows editing it removes that position from the suggested
    product code (see product_field_overrides.
    is_nomenclature_label_hidden_or_edited); field_type is always
    "select" for one of those, since it's still a real ordering code."""
    hidden = product_field_overrides.get_hidden(product.product_name)
    fields_override = product_field_overrides.get_fields(product.product_name)
    result: list[tuple[str, dict[str, str], bool, str]] = []
    for _seg_code, label, display in _admin_selectable_segments(product.nomenclature):
        if len(display) < 2 or label in hidden or label in fields_override:
            continue
        result.append((label, display, True, "select"))
    for label, field in product_field_overrides.effective_additional_params(product).items():
        result.append((label, field.get("options", {}), False, field.get("type", "select")))
    return result


@st.dialog("Delete this field?")
def _confirm_delete_product_field_dialog(product_name: str, label: str) -> None:
    st.write(f"Delete **{label}** from {product_name}'s Customer Requirements fields? This cannot be undone.")
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary", use_container_width=True):
        product_field_overrides.remove_field(product_name, label)
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


@st.dialog("Delete this field?")
def _confirm_delete_generic_field_dialog(key: str, label: str) -> None:
    st.write(f"Delete **{label}** from the general Customer Requirements fields? This cannot be undone.")
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary", use_container_width=True):
        fields = load_fields()
        save_fields([f for f in fields if f["key"] != key])
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


def _render_product_field_row(
    product_name: str, label: str, options: dict[str, str], is_ordering_code: bool, field_type: str = "select"
) -> None:
    # Deliberately NO st.container(border=True) wrapper, and label +
    # options collapsed into one markdown call instead of separate
    # markdown/caption/caption calls -- proven necessary by direct
    # feedback: the original, more deeply-nested per-row layout (a
    # bordered container, 3 columns, 2-3 text elements each) pushed this
    # page's total element count high enough that switching AWAY from
    # this tab to a totally different one (e.g. Approved Claims) could
    # leave that OTHER tab's real content sitting alongside orphaned,
    # not-cleaned-up DOM from whatever tab was active before -- a
    # Streamlit/React reconciliation issue tied to the SIZE of what
    # changes in one rerun, not anything specific to this tab's logic.
    # Fewer elements per row keeps the whole page's rerun delta smaller.
    edit_key = f"editing-product-{product_name}-{label}"
    editing = st.session_state.get(edit_key, False)
    if not editing:
        col1, col2, col3 = st.columns([3, 1, 1])
        code_note = " *(ordering code)*" if is_ordering_code else ""
        opts_note = f" -- {', '.join(options.keys())}" if options else ""
        col1.markdown(f"**{label}**{code_note} -- {field_type}{opts_note}")
        if col2.button("Edit", key=f"edit-btn-product-{product_name}-{label}", use_container_width=True):
            st.session_state[edit_key] = True
            st.rerun()
        if col3.button("Delete", key=f"del-btn-product-{product_name}-{label}", use_container_width=True):
            _confirm_delete_product_field_dialog(product_name, label)
    else:
        with st.form(f"edit-form-product-{product_name}-{label}"):
            new_type = st.selectbox(
                "Type", FIELD_TYPES, index=FIELD_TYPES.index(field_type) if field_type in FIELD_TYPES else 0
            )
            new_options_text = st.text_area(
                "Options",
                value=_options_to_text(options),
                height=120,
            )
            st.caption("One per line. Add Option name: description if needed.")
            save_col, cancel_col = st.columns(2)
            save = save_col.form_submit_button("Save", type="primary", use_container_width=True)
            cancel = cancel_col.form_submit_button("Cancel", use_container_width=True)
        if save:
            new_options = _parse_options_text(new_options_text)
            if new_type in ("select", "multiselect", "radio") and not new_options:
                st.error("Enter at least one option.")
            else:
                product_field_overrides.set_field(product_name, label, new_options, new_type)
                st.session_state[edit_key] = False
                st.success("Saved.")
                st.rerun()
        if cancel:
            st.session_state[edit_key] = False
            st.rerun()


def _render_generic_field_row(field: dict) -> None:
    # Same flattened, no-bordered-container layout as
    # _render_product_field_row -- see its comment for why.
    key = field["key"]
    edit_key = f"editing-generic-{key}"
    editing = st.session_state.get(edit_key, False)
    if not editing:
        col1, col2, col3 = st.columns([3, 1, 1])
        hidden_tag = " *(hidden)*" if not field.get("visible", True) else ""
        type_line = field.get("type", "text")
        if field.get("options"):
            type_line += " -- " + ", ".join(field["options"])
        col1.markdown(f"**{field['label']}**{hidden_tag} -- {type_line}")
        if col2.button("Edit", key=f"edit-btn-generic-{key}", use_container_width=True):
            st.session_state[edit_key] = True
            st.rerun()
        # A core field's widget is hardcoded in requirements_page.py
        # (not generated from this list), so deleting the row
        # wouldn't remove it from the form, only reset its label
        # back to default and silently re-show it -- hiding
        # (visible=False, row kept) is the only real removal for one
        # of these. A custom field has no such widget to fall back
        # to, so it's deleted outright.
        delete_label = "Hide" if field.get("core") else "Delete"
        if col3.button(delete_label, key=f"del-btn-generic-{key}", use_container_width=True):
            if field.get("core"):
                # Reversible (just sets visible=False, un-hide via Edit's
                # "Visible" checkbox) -- no confirmation needed.
                fields = load_fields()
                save_fields([{**f, "visible": False} if f["key"] == key else f for f in fields])
                st.rerun()
            else:
                _confirm_delete_generic_field_dialog(key, field["label"])
    else:
        with st.form(f"edit-form-generic-{key}"):
            new_label = st.text_input("Label", value=field["label"])
            new_type = st.selectbox(
                "Type", FIELD_TYPES, index=FIELD_TYPES.index(field.get("type", "text"))
            )
            new_options = st.text_input(
                "Options (comma-separated -- only used for select / multiselect / radio)",
                value=", ".join(field.get("options") or []),
            )
            new_required = st.checkbox("Required", value=field.get("required", False))
            new_visible = st.checkbox("Visible", value=field.get("visible", True))
            save_col, cancel_col = st.columns(2)
            save = save_col.form_submit_button("Save", type="primary", use_container_width=True)
            cancel = cancel_col.form_submit_button("Cancel", use_container_width=True)
        if save:
            label = new_label.strip()
            if not label:
                st.error("Label can't be empty.")
            else:
                options_list = [o.strip() for o in new_options.split(",") if o.strip()]
                fields = load_fields()
                save_fields([
                    {
                        **f, "label": label, "type": new_type, "options": options_list,
                        "required": new_required, "visible": new_visible,
                    } if f["key"] == key else f
                    for f in fields
                ])
                st.session_state[edit_key] = False
                st.success("Saved.")
                st.rerun()
        if cancel:
            st.session_state[edit_key] = False
            st.rerun()


def _render_requirements_form_fields_tab() -> None:
    st.caption(
        "Edit the fields used in Customer Requirements. Product-specific changes are saved as "
        "overrides and survive index rebuilds."
    )
    with st.expander("How this works"):
        st.caption(
            "Covers both each product's own selectable specs (Output Signal, Range, Background, "
            "etc.) and the generic questions below (Industry, Certifications, etc.). Changes take "
            "effect immediately, no restart needed. A product-specific edit lives in its own file "
            "that survives a Rebuild Index -- data/product_registry.json itself gets fully "
            "regenerated from the catalogues on every rebuild, so editing it directly would just "
            "get overwritten."
        )

    try:
        products, _ = load_product_index()
    except (FileNotFoundError, OSError):
        products = []

    if products:
        product_names = [p.product_name for p in products]
        selected_name = st.selectbox(
            "Product (admin)", product_names, key="ADMIN_PAGE_PRODUCT_FIELD_SELECTOR_UNIQUE_9f3a"
        )
        selected = next(p for p in products if p.product_name == selected_name)

        st.markdown(f"**{selected_name}'s selectable fields**")
        product_fields = _effective_product_fields(selected)
        if not product_fields:
            st.caption("No selectable fields for this product yet.")
        for label, options, is_ordering_code, field_type in product_fields:
            _render_product_field_row(selected_name, label, options, is_ordering_code, field_type)

        st.markdown(f"**Add a field to {selected_name}**")
        with st.form(f"add-product-field-{selected_name}", clear_on_submit=True):
            col1, col2 = st.columns(2)
            new_label = col1.text_input("Label", key=f"new-product-field-label-{selected_name}")
            new_type = col2.selectbox("Type", FIELD_TYPES, key=f"new-product-field-type-{selected_name}")
            new_options_text = st.text_area(
                "Options",
                height=100,
                key=f"new-product-field-options-{selected_name}",
            )
            st.caption("One per line. Add Option name: description if needed.")
            add_submitted = st.form_submit_button("Add Field", type="primary")
        if add_submitted:
            label = new_label.strip()
            existing_labels = {l for l, _, _, _ in product_fields}
            if not label:
                st.error("Enter a label.")
            elif label in existing_labels:
                st.error(f"\"{label}\" already exists for this product -- edit it above instead.")
            else:
                options = _parse_options_text(new_options_text)
                if new_type in ("select", "multiselect", "radio") and not options:
                    st.error("Enter at least one option.")
                else:
                    product_field_overrides.set_field(selected_name, label, options, new_type)
                    st.success(f"Added \"{label}\" to {selected_name}.")
                    st.rerun()
    else:
        st.caption("No products in the registry yet -- add a Product Catalogue document first.")

    st.divider()
    st.markdown("**General fields** (not tied to a specific product)")
    st.caption(
        "Customer Name/Company aren't listed here -- they come from the Deal picker at the top of "
        "the page, not this list."
    )
    fields = load_fields()
    for field in fields:
        _render_generic_field_row(field)

    st.markdown("**Add a general field**")
    with st.form("add-requirement-field-form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        new_label = col1.text_input("Label")
        new_type = col2.selectbox("Type", FIELD_TYPES)
        new_options = st.text_input(
            "Options (comma-separated -- only used for select / multiselect / radio)"
        )
        new_required = st.checkbox("Required")
        add_submitted = st.form_submit_button("Add Field", type="primary")
    if add_submitted:
        label = new_label.strip()
        if not label:
            st.error("Enter a label for the new field.")
        else:
            options = [o.strip() for o in new_options.split(",") if o.strip()]
            key = unique_key(label, "", {f["key"] for f in fields})
            save_fields([
                *fields,
                {
                    "key": key,
                    "label": label,
                    "type": new_type,
                    "options": options,
                    "required": new_required,
                    "visible": True,
                    "core": False,
                },
            ])
            st.success(f"Added \"{label}\".")
            st.rerun()


def _render_requirements_admin_tab() -> None:
    # No nested st.tabs() here -- proven necessary by direct feedback:
    # a SECOND, inner st.tabs() nested inside this outer 6-tab group's
    # own "Customer Requirements" tab was fragile in a way a single
    # level of tabs isn't -- interacting with a widget deep inside the
    # inner tabs (e.g. the Product picker below) could leave the page
    # showing a completely different OUTER tab's content (e.g.
    # "Documents") while "Customer Requirements" still showed as the
    # visually active tab label. An st.fragment was tried as a fix and
    # made it worse (real content bleeding between tabs, not just a
    # timing flash), so it was reverted. Flattening to one level of
    # tabs removes the specific nested-tabs interaction entirely.
    # Submitted Requirements now lives in a collapsed expander instead
    # of its own inner tab.
    with st.expander(f"Submitted Requirements ({len(list_requirements())})"):
        _render_requirements_records_tab()
    st.divider()
    _render_requirements_form_fields_tab()


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------
def render_admin_page() -> None:
    if not st.session_state.get("admin_authenticated"):
        st.error("Sign in as admin from the sidebar to access this page.")
        st.stop()

    st.title("Admin")
    st.caption("Document management, index rebuilds, and claim policy editing.")

    documents_tab, approved_tab, restricted_tab, feedback_tab, deals_tab, requirements_tab = st.tabs(
        ["Documents", "Approved Claims", "Restricted Claims", "Feedback", "Deals", "Customer Requirements"],
        key="admin-main-tabs",
    )

    with documents_tab:
        _render_documents_tab()

    with approved_tab:
        _render_approved_claims_tab()

    with restricted_tab:
        _render_restricted_claims_tab()

    with feedback_tab:
        _render_feedback_tab()

    with deals_tab:
        _render_deals_tab()

    with requirements_tab:
        _render_requirements_admin_tab()
