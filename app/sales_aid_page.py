"""Sales Aid page: generates a short, customer-facing comparison document
for a specific use case, grounded in the approved knowledge base -- meant
to be handed to a customer's technical champion to circulate internally.
"""
from __future__ import annotations

import streamlit as st

from app import theme
from app.response_generator import ResponseGeneratorError
from app.retriever import RetrieverError
from app.sales_aid_generator import generate_sales_aid

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
            with st.spinner("Checking approved documents..."):
                try:
                    result = generate_sales_aid(use_case, compare_against)
                except (RetrieverError, ResponseGeneratorError) as e:
                    st.error(f"Something went wrong generating the sales aid: {e}")
                else:
                    st.session_state.sales_aid_result = result

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
