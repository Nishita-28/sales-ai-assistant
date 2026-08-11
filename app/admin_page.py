"""Admin page: document management, index rebuilds, and claims-policy
editing. render_admin_page() re-checks the sign-in state itself, in case
it's ever reached without going through the sidebar gate first.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app import theme
from app.claims_store import load_claims, save_claims
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
from app.registry_builder import rebuild_product_registry
from app.requirements_fields import FIELD_TYPES, load_fields, merge_core_fields, save_fields, unique_key
from app.requirements_store import delete_requirement, list_requirements
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
    folder is real on disk but contributes zero chunks."""
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
        REMOVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(doc_path), str(REMOVED_DOCS_DIR / doc_path.name))
        remove_document_type(doc_path.name)
        try:
            remove_document_from_index(doc_path.name)
            rebuild_product_registry(APPROVED_DOCS_DIR)
        except RetrieverError as e:
            st.session_state.docs_changed_since_rebuild = True
            st.error(f"Moved the file, but removing it from the index failed: {e}. Use Rebuild Index to retry.")
        else:
            st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


_TYPE_HELP = {
    "Product Catalogue": "One specific purchasable product's exact specs and ordering codes. Populates the Product Registry.",
    "Use Case Guide": "Which industries/applications MNST's products serve -- factual, spans multiple products.",
    "Technical Guide": "Vendor-neutral background on how a sensing technology works -- not product- or competitor-specific.",
    "Historical Sales Record": "A log of real past deals -- evidence, never treated as a recommendable product.",
    "Internal Sales Strategy": "MNST's own subjective/dated sales judgment (competitive positioning, objections). Excluded from the main Assistant.",
    "Sales Methodology Reference": "Qualification frameworks, negotiation tactics, outreach cadences -- internal coaching material. Excluded from the main Assistant.",
    "Other": "Doesn't fit the categories above yet -- fully open for now, revisit and reclassify when you can.",
}


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
        else:
            doc_type = st.selectbox("Document type", ALL_TYPES, key="new-doc-type")
            st.caption(_TYPE_HELP[doc_type])
            if st.button("Add to knowledge base", type="primary"):
                APPROVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(uploaded_file.getbuffer())
                set_document_type(uploaded_file.name, doc_type)
                with st.spinner("Indexing new document..."):
                    try:
                        count = add_document_to_index(target_path)
                        rebuild_product_registry(APPROVED_DOCS_DIR)
                    except RetrieverError as e:
                        st.session_state.docs_changed_since_rebuild = True
                        st.error(f"Added the file, but indexing failed: {e}. Use Rebuild Index to retry.")
                    else:
                        st.success(f"Added and indexed {uploaded_file.name} ({count} chunks) as {doc_type}.")
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
        current_type = get_document_type(doc_path.name)
        col1, col2, col3 = st.columns([4, 2, 1])
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

    if any(get_document_type(p.name) is None for p in doc_paths):
        st.warning(
            "Some documents above have no assigned type yet (likely added before this classification existed, "
            "or via a script that bypassed this form). They default to open/reference behavior until classified -- "
            "pick a type for each from the dropdown."
        )


# ---------------------------------------------------------------------------
# Approved Claims tab -- plain reference documentation for human review;
# nothing here is actually enforced by the assistant.
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

    if bullets:
        st.divider()
        col1, col2 = st.columns([4, 1])
        to_delete = col1.selectbox(
            "Remove a claim", bullets, key="delete-approved-claim-select", label_visibility="collapsed"
        )
        if col2.button("Delete", key="delete-approved-claim-btn", use_container_width=True):
            save_claims(APPROVED_CLAIMS_PATH, header, [b for b in bullets if b != to_delete])
            st.success(f"Removed: {to_delete}")
            st.rerun()


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

    if entries:
        st.divider()

        def _entry_label(i: int) -> str:
            e = entries[i]
            keywords = ", ".join(e.keywords[:3]) + ("..." if len(e.keywords) > 3 else "")
            return f"{e.category} -- {keywords}"

        col1, col2 = st.columns([4, 1])
        idx = col1.selectbox(
            "Remove an entry",
            range(len(entries)),
            format_func=_entry_label,
            key="delete-restricted-select",
            label_visibility="collapsed",
        )
        if col2.button("Delete", key="delete-restricted-btn", use_container_width=True):
            remaining = [e for i, e in enumerate(entries) if i != idx]
            save_entries(RESTRICTED_CLAIMS_PATH, remaining)
            st.success(f"Removed: {_entry_label(idx)}")
            st.rerun()


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
        return

    for report in reports:
        with st.container(border=True):
            st.markdown(f"**{report['verdict'].capitalize()}** · {report['created_at']}")
            st.write(f"Q: {report['question']}")
            st.write(f"A: {report['answer']}")
            if report["note"]:
                st.caption(f"Note: {report['note']}")
            col1, col2 = st.columns(2)
            if col1.button("Resolve", key=f"resolve-{report['id']}", use_container_width=True):
                resolve_feedback(report["id"])
                st.rerun()
            if col2.button("Delete", key=f"delete-{report['id']}", use_container_width=True):
                _confirm_delete_report_dialog(report["id"])


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


def _render_requirements_form_fields_tab() -> None:
    st.caption(
        "Edit the Customer Requirements page's generic questions -- label, options, whether it's "
        "required, and whether it's shown at all. Changes take effect immediately, no restart needed. "
        "Built-in fields (Customer Name, Company, Certifications, etc.) can be renamed, relabeled, or "
        "hidden, but not deleted outright -- they're tied to real stored data or logic elsewhere "
        "(Customer Name and Company specifically are always required and always shown, regardless of "
        "the Visible checkbox). Add a new row for a brand-new field; leave its Key blank, it's "
        "generated from the Label. Per-product ordering options (Output Signal, Range, Background, "
        "etc.) aren't edited here -- those come straight from the approved product catalogues."
    )

    fields = load_fields()
    df = pd.DataFrame(
        [
            {
                "key": f["key"],
                "label": f["label"],
                "type": f.get("type", "text"),
                "options": ", ".join(f.get("options") or []),
                "required": bool(f.get("required", False)),
                "visible": bool(f.get("visible", True)),
            }
            for f in fields
        ]
    )

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor-requirements-fields",
        column_config={
            "key": st.column_config.TextColumn(
                help="Auto-generated from the label for a new field -- leave blank.", disabled=True
            ),
            "type": st.column_config.SelectboxColumn(options=FIELD_TYPES, required=True),
            "options": st.column_config.TextColumn(
                help="Comma-separated -- only used for select / multiselect / radio fields."
            ),
            "required": st.column_config.CheckboxColumn(),
            "visible": st.column_config.CheckboxColumn(
                help="Unchecked = not shown on the form. A built-in field whose row is deleted here "
                "is kept and hidden instead of removed."
            ),
        },
    )

    if st.button("Save Form Fields", type="primary"):
        existing_by_key = {f["key"]: f for f in fields}
        edited_fields = []
        seen_keys: set[str] = set()
        for row in edited.itertuples(index=False):
            label = str(row.label).strip()
            if not label:
                continue
            key = unique_key(label, str(row.key or ""), seen_keys)
            seen_keys.add(key)
            options = [o.strip() for o in str(row.options).split(",") if o.strip()] if row.options else []
            edited_fields.append(
                {
                    "key": key,
                    "label": label,
                    "type": row.type,
                    "options": options,
                    "required": bool(row.required),
                    "visible": bool(row.visible),
                    "core": existing_by_key.get(key, {}).get("core", False),
                }
            )
        save_fields(merge_core_fields(edited_fields))
        st.success("Saved. The Customer Requirements page reflects this immediately.")
        st.rerun()

    custom_fields = [f for f in fields if not f.get("core")]
    if custom_fields:
        st.divider()
        col1, col2 = st.columns([4, 1])
        to_delete = col1.selectbox(
            "Remove a custom field",
            [f["key"] for f in custom_fields],
            format_func=lambda k: next(f["label"] for f in custom_fields if f["key"] == k),
            key="delete-req-field-select",
            label_visibility="collapsed",
        )
        if col2.button("Delete", key="delete-req-field-btn", use_container_width=True):
            save_fields([f for f in fields if f["key"] != to_delete])
            st.success("Removed.")
            st.rerun()


def _render_requirements_admin_tab() -> None:
    records_tab, fields_tab = st.tabs(["Submitted Requirements", "Form Fields"])
    with records_tab:
        _render_requirements_records_tab()
    with fields_tab:
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

    _render_rebuild_status()

    documents_tab, approved_tab, restricted_tab, feedback_tab, requirements_tab = st.tabs(
        ["Documents", "Approved Claims", "Restricted Claims", "Feedback", "Customer Requirements"]
    )

    with documents_tab:
        _render_documents_tab()

    with approved_tab:
        _render_approved_claims_tab()

    with restricted_tab:
        _render_restricted_claims_tab()

    with feedback_tab:
        _render_feedback_tab()

    with requirements_tab:
        _render_requirements_admin_tab()
