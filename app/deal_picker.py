"""Shared "which deal" picker UI -- used by any page that captures
something which should be linked to a specific customer deal (Pre-Call
Discovery, Customer Requirements, and eventually Sales Aid). Extracted
here once a second page needed the identical picker, rather than
duplicated per page.

Kept out of app/deals_store.py deliberately -- every other *_store.py
module in this codebase is pure data-layer (no `streamlit` import), and
this is UI.
"""
from __future__ import annotations

import streamlit as st

from app.deals_store import create_deal, list_deals

NEW_DEAL_LABEL = "+ New deal"


def deal_label(deal) -> str:
    use_case_snippet = deal["use_case"][:40] + ("..." if len(deal["use_case"]) > 40 else "")
    return f"{deal['company']} -- {deal['customer_name']} ({use_case_snippet or 'no use case yet'})"


def render_deal_picker(key_prefix: str) -> tuple[int | None, str, str, str]:
    """Select an existing deal, or start a new one. Returns (deal_id,
    customer_name, company, use_case) -- deal_id is None while "+ New deal"
    is selected and hasn't been started yet. Deal creation is fully owned
    here, inside one st.form.

    Uses the shared, page-agnostic st.session_state.active_deal_id as the
    default selection, so a deal chosen on one page is already selected on
    another; key_prefix only namespaces this picker's own widget keys.
    Selection is by deal id, not display label, since two deals for the
    same customer/company can produce an identical label. The widget key
    is suffixed with the active deal id since a fixed key isn't reliable
    enough for BaseWeb Select to resync its displayed value on switch."""
    deals = list_deals()
    deal_by_id = {d["id"]: d for d in deals}
    options: list[int | None] = [None] + list(deal_by_id.keys())

    def _format(option_id: int | None) -> str:
        return NEW_DEAL_LABEL if option_id is None else deal_label(deal_by_id[option_id])

    active_id = st.session_state.get("active_deal_id")
    default_index = options.index(active_id) if active_id in options else 0
    widget_key = f"{key_prefix}_deal_select_{active_id if active_id is not None else 'new'}"

    chosen_id = st.selectbox(
        "Deal", options, index=default_index, format_func=_format, key=widget_key
    )

    if chosen_id is None:
        with st.form(f"{key_prefix}_new_deal_form"):
            col1, col2 = st.columns(2)
            customer_name = col1.text_input("Customer Name", key=f"{key_prefix}_new_customer_name")
            company = col2.text_input("Company", key=f"{key_prefix}_new_company")
            started = st.form_submit_button("Start New Deal")

        if started:
            if customer_name.strip() and company.strip():
                new_id = create_deal(customer_name.strip(), company.strip())
                set_active_deal(new_id)
                st.rerun()
            else:
                st.error("Enter both Customer Name and Company.")

        return None, customer_name, company, ""

    deal = deal_by_id[chosen_id]
    st.caption(f"**{deal['company']}** -- {deal['customer_name']}")
    return deal["id"], deal["customer_name"], deal["company"], deal["use_case"]


def set_active_deal(deal_id: int | None) -> None:
    st.session_state.active_deal_id = deal_id
