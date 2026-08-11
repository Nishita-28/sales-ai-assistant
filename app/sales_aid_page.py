"""Sales Aid page: generates a short, customer-facing comparison document
for a specific use case, grounded in the approved knowledge base -- meant
to be handed to a customer's technical champion to circulate internally.
"""
from __future__ import annotations

import io
import re

import streamlit as st
from docx import Document

from app import theme
from app.response_generator import ResponseGeneratorError
from app.retriever import RetrieverError
from app.sales_aid_generator import finalize_sales_aid, stream_sales_aid

EXAMPLE_USE_CASE = (
    "Multiple potential leak spots in close vicinity, with 100% H2 contained in pipelines."
)


def render_sales_aid_page() -> None:
    st.title("Sales Aid Generator")
    st.caption(
        "Describe the customer's use case to get a short comparison document you can hand to "
        "the customer's technical champion -- grounded in the approved knowledge base only."
    )

    # st.form so Enter in "Compare against" submits like clicking the button
    # does -- a plain text_input + separate button doesn't submit on Enter.
    with st.form("sales_aid_form"):
        use_case = st.text_area("Use case / scenario", placeholder=EXAMPLE_USE_CASE, height=100)
        compare_against = st.text_input(
            "Compare against (optional)",
            placeholder="e.g. Metal Oxide Semiconductor Sensors -- leave blank to let the AI pick",
        )
        generate = st.form_submit_button("Generate sales aid", type="primary")

    if generate:
        if not use_case.strip():
            st.error("Describe the use case first.")
        else:
            try:
                with st.spinner("Checking approved documents..."):
                    matches, text_stream = stream_sales_aid(use_case, compare_against)
                with st.container(border=theme.is_enterprise_theme()):
                    raw_reply = st.write_stream(text_stream)
            except (RetrieverError, ResponseGeneratorError) as e:
                st.error(f"Something went wrong generating the sales aid: {e}")
            else:
                st.session_state.sales_aid_result = finalize_sales_aid(use_case, matches, raw_reply)
                st.rerun()

    result = st.session_state.get("sales_aid_result")
    if not result:
        return

    st.divider()

    if result.ready_for_customer:
        st.success("Ready to share with the customer.")
    else:
        st.warning(
            "Internal draft -- needs review before sharing with the customer "
            f"(risk category: {result.risk})."
        )

    if result.customer_priorities:
        st.markdown("**Customer priorities identified**")
        for p in result.customer_priorities:
            st.markdown(f"- {p}")

    with st.container(border=theme.is_enterprise_theme()):
        if result.title:
            st.subheader(result.title)
        if result.use_case_framing:
            st.write(result.use_case_framing)

        if result.comparison:
            st.markdown("**Comparison**")
            st.markdown("\n".join(result.comparison))

    st.text_area("Customer-ready summary", value=result.customer_summary, height=150)

    if result.sources:
        with st.expander("Drawn from"):
            for name, _ in result.sources:
                st.markdown(f"- `{name}`")

    st.download_button(
        "Download sales aid (.docx)",
        data=_sales_aid_as_docx(result),
        file_name=f"{(result.title or 'sales_aid').strip().replace(' ', '_')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


_TABLE_SEPARATOR_RE = re.compile(r":?-+:?")


def _parse_markdown_table(lines: list[str]) -> list[list[str]]:
    """result.comparison is the raw markdown table the LLM produced
    (header, separator, data rows, pipe-delimited) -- this turns it back
    into plain rows of cell text, dropping the "|---|---|" separator row,
    so it can be rebuilt as a real docx table instead of exported as
    literal pipe characters (which is what made the old .txt download's
    table unreadable in Notepad/Word)."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(_TABLE_SEPARATOR_RE.fullmatch(c) for c in cells):
            continue
        rows.append(cells)
    return rows


def _sales_aid_as_docx(result) -> bytes:
    """Word export of the full sales aid -- a real docx table for the
    comparison (correct alignment in Word/Google Docs/Outlook, unlike a
    plain-text export of markdown pipe syntax), matching the page's own
    stated purpose: something to hand to a customer's technical champion
    to circulate internally."""
    doc = Document()
    doc.add_heading(result.title or "Sales Aid", level=1)

    if result.customer_priorities:
        doc.add_heading("Customer priorities identified", level=2)
        for p in result.customer_priorities:
            doc.add_paragraph(p, style="List Bullet")

    if result.use_case_framing:
        doc.add_paragraph(result.use_case_framing)

    rows = _parse_markdown_table(result.comparison)
    if rows:
        doc.add_heading("Comparison", level=2)
        cols = max(len(r) for r in rows)
        table = doc.add_table(rows=len(rows), cols=cols)
        table.style = "Light Grid Accent 1"
        for r, row_cells in enumerate(rows):
            for c, text in enumerate(row_cells[:cols]):
                cell = table.cell(r, c)
                cell.text = text
                if r == 0:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.bold = True

    if result.customer_summary:
        doc.add_heading("Customer-ready summary", level=2)
        doc.add_paragraph(result.customer_summary)

    if result.sources:
        doc.add_heading("Drawn from", level=2)
        for name, _ in result.sources:
            doc.add_paragraph(name, style="List Bullet")

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
