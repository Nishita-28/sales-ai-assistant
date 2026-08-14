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
    customer_name, company, use_case) -- deal_id is None only while
    "+ New deal" is selected and hasn't been started yet (customer_name/
    company will also read as "" in that state, even if the rep has
    typed into the boxes -- see below for why).

    Starting a new deal is a single atomic step: Customer Name + Company
    + a "Start New Deal" button, all inside one st.form. Proven
    necessary by direct feedback: with these as plain, separate widgets
    next to a page's own "Generate"/"Save" button, clicking that button
    could race ahead of a just-typed field not yet synced to the
    server -- the browser still showed the typed text, but the script
    that ran to handle the click saw an empty value and rejected it. A
    real st.form is Streamlit's own mechanism for making several widgets
    commit together atomically with their submit button, which removes
    that race entirely rather than trying to out-guess its timing. The
    consequence: a caller can no longer create a deal itself from
    whatever render_deal_picker() returns while deal_id is None (those
    values are always "" pre-submission by design) -- deal creation is
    fully owned here now.

    Uses the shared, page-agnostic st.session_state.active_deal_id (not
    prefixed) to pick the picker's default selection -- so a deal chosen
    on one page (e.g. Discovery) is already selected when a rep
    navigates to another (e.g. Customer Requirements), instead of
    re-picking the same deal per page. key_prefix only namespaces this
    picker's own WIDGET keys, since two pages' pickers are never both on
    screen in the same script run but should still never risk a key
    collision.

    Selection is by deal id (via format_func), never by the display
    label string -- proven necessary by testing: two deals for the same
    customer/company with an identical use case produce an identical
    label, and matching by label text would silently collide, making
    one of them unreachable from the picker.

    The selectbox's own widget key is suffixed with the active deal id
    (not a fixed string) so that switching the active deal -- via
    "Start New Deal" or a manual pick -- always mounts a genuinely new
    widget instance instead of updating an existing one in place.
    Proven necessary by testing in a real browser (not just AppTest,
    which doesn't exercise the frontend and never caught this): right
    after creating a deal, the selectbox kept visually showing "+ New
    deal" even though the backend had already moved on to the new
    deal -- and the NEXT interaction with any other widget on the page
    (e.g. blurring the use-case box) sent that stale, still-"+ New
    deal" frontend value back to the server, silently reverting the
    whole page (active deal, and anything typed into deal-scoped boxes
    like the use case) back to a blank "+ New deal" state. A fixed key
    across that transition apparently isn't enough for the underlying
    BaseWeb Select component to reliably resync its displayed value;
    changing the key forces a clean remount instead."""
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
