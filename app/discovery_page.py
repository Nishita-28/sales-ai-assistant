"""Pre-call discovery workflow: a rep describes a prospective customer's use
case, gets back documented "Right to Win" points -- each with its own
supporting evidence and exploratory questions -- records the customer's
actual answers during the call, and then -- once there's enough evidence --
gets a knowledge-base-grounded product recommendation (or an honest "not
enough evidence yet" with what to ask next). One continuous workflow, not a
separate feature.
"""
from __future__ import annotations

import html

import streamlit as st

from app import theme
from app.deal_picker import render_deal_picker, set_active_deal
from app.deals_store import (
    get_deal,
    get_discovery,
    get_qualification,
    get_recommendation,
    missing_fields,
    save_discovery,
    save_qualification,
    save_recommendation,
    update_deal_fields,
)
from app.discovery_generator import (
    MIN_ANSWERS_FOR_RECOMMENDATION,
    DiscoveryResult,
    QualificationQuestion,
    RecommendationResult,
    RightToWinPoint,
    finalize_discovery_questions,
    finalize_qualification_questions,
    finalize_recommendation,
    insufficient_recommendation,
    stream_discovery_points,
    stream_discovery_questions,
    stream_qualification_questions,
    stream_recommendation,
)
from app.generation_helpers import dedupe_sources
from app.response_generator import ResponseGeneratorError
from app.retriever import RetrieverError

EXAMPLE_USE_CASE = (
    "Customer runs a lead-acid battery room in a warehouse and wants to monitor "
    "for hydrogen buildup. Not sure yet if the area is classified hazardous."
)

OUTCOME_RENDER = {
    "Trade-offs": st.info,
    "Insufficient": st.warning,
}


def _render_recommend_outcome(product: str) -> None:
    """A "Recommend" outcome gets the app's own brand blue, not
    Streamlit's default green st.success -- literal hex values (not the
    theme.py CSS vars) so this still renders correctly under
    UI_THEME=classic, where inject_theme() never runs and those vars
    are undefined. The recommended product name (parsed separately from
    the prose by discovery_generator's "Recommended Product:" section)
    is shown right in the box, not left buried in the first sentence of
    the paragraph below it."""
    product_html = (
        f'<div style="margin-top:0.25rem;font-size:1.15rem;font-weight:700;">{html.escape(product)}</div>'
        if product
        else ""
    )
    st.markdown(
        '<div style="background:#edf3fb;border:1px solid #184fa3;color:#123c80;'
        'padding:0.75rem 1rem;border-radius:0.5rem;">'
        f'Outcome: Recommend{product_html}</div>',
        unsafe_allow_html=True,
    )


def _save_discovery_answer(deal_id: int, question: str, widget_key: str) -> None:
    """on_change callback for a Right-to-Win answer box -- saves the updated
    answers dict (plus the current Right-to-Win points, so the saved blob
    stays self-contained) straight to deals.db the moment the rep commits
    an answer, so the DB copy is always current as of the last committed
    answer, not just the last full "Generate" click."""
    answers = st.session_state.setdefault("discovery_answers", {})
    answers[question] = st.session_state.get(widget_key, "")
    st.session_state.discovery_answers = answers
    result = st.session_state.get("discovery_result")
    if result is not None:
        save_discovery(deal_id, [p.to_dict() for p in result.right_to_win], answers, result.sources)


def _render_right_to_win_card(
    i: int, point: RightToWinPoint, answers: dict[str, str], deal_id: int | None = None
) -> None:
    """One Right-to-Win card -- title, a one-line "why it matters" (kept
    short so a rep prepping for a call isn't scrolling past dense
    paragraphs to reach the questions), the questions to ask, and the
    detailed evidence tucked into a collapsed expander. Shared by the live
    (mid-stream) render (deal_id left as None -- transient cards during
    generation don't save on every keystroke) and the persisted
    post-generation render (deal_id passed through, so answers save
    immediately), using the same `answers` dict and widget key scheme so a
    typed-ahead answer survives the switch between them."""
    with st.container(border=theme.is_enterprise_theme()):
        st.markdown(f"**Right to Win #{i} -- {point.title}**")

        # Falls back to a truncated evidence snippet only if the model
        # somehow didn't produce a distinct Why It Matters line -- the
        # line above should never be blank.
        why = point.why_it_matters or (
            point.evidence[:160] + "..." if len(point.evidence) > 160 else point.evidence
        )
        if why:
            st.write(why)

        if point.questions:
            st.markdown("**Questions to ask**")
            for j, q in enumerate(point.questions, start=1):
                widget_key = f"dq-ans-{i}-{j}"
                save_kwargs = (
                    {"on_change": _save_discovery_answer, "args": (deal_id, q, widget_key)}
                    if deal_id is not None
                    else {}
                )
                answers[q] = st.text_input(
                    q,
                    value=answers.get(q, ""),
                    key=widget_key,
                    placeholder="Customer's answer (leave blank if not asked yet)",
                    **save_kwargs,
                )

        if point.evidence:
            with st.expander("Why we think this"):
                st.caption(point.evidence)
    if not theme.is_enterprise_theme():
        st.markdown("")


def _save_qualification_answer(deal_id: int, field_key: str, widget_key: str) -> None:
    """on_change callback for a qualification answer box -- saves straight to
    deals.db the moment the rep commits an answer, so next time this deal is
    opened, this field is no longer missing and won't be asked again (see
    deals_store.missing_fields)."""
    value = st.session_state.get(widget_key, "").strip()
    if value:
        update_deal_fields(deal_id, **{field_key: value})


def _render_qualification_card(deal_id: int, q: QualificationQuestion) -> None:
    widget_key = f"qual-ans-{deal_id}-{q.field_key}"
    with st.container(border=theme.is_enterprise_theme()):
        st.markdown(f"**{q.field_label}**")
        st.text_input(
            q.question,
            key=widget_key,
            placeholder="Customer's answer (leave blank if not asked yet)",
            on_change=_save_qualification_answer,
            args=(deal_id, q.field_key, widget_key),
        )
        st.caption(q.rationale)
    if not theme.is_enterprise_theme():
        st.markdown("")


def _load_saved_recommendation(deal_id: int | None) -> RecommendationResult | None:
    """The deal's last saved recommendation, reconstructed from its
    stored dict -- or None if it never got one (or deal_id is None, for
    "+ New deal")."""
    if deal_id is None:
        return None
    deal = get_deal(deal_id)
    raw = get_recommendation(deal) if deal is not None else None
    if raw is None:
        return None
    return RecommendationResult(
        confirmed_priorities=raw["confirmed_priorities"],
        outcome=raw["outcome"],
        # .get, not [] -- a recommendation saved before the "product"/
        # "candidate_products" fields existed won't have these keys at all.
        product=raw.get("product", ""),
        candidate_products=raw.get("candidate_products", []),
        recommendation=raw["recommendation"],
        missing_information=raw["missing_information"],
        sources=[tuple(s) for s in raw["sources"]],
        risk=raw["risk"],
    )


def _load_saved_discovery(deal_id: int | None) -> tuple[DiscoveryResult | None, dict[str, str]]:
    """The deal's last saved Right-to-Win generation and the rep's answers
    against it, reconstructed from the stored blob -- or (None, {}) if this
    deal has no saved generation yet."""
    if deal_id is None:
        return None, {}
    deal = get_deal(deal_id)
    raw = get_discovery(deal) if deal is not None else None
    if raw is None:
        return None, {}
    points = [RightToWinPoint(**p) for p in raw["right_to_win"]]
    result = DiscoveryResult(right_to_win=points, sources=[tuple(s) for s in raw.get("sources", [])])
    return result, raw["answers"]


def _load_saved_qualification(deal_id: int | None) -> list[QualificationQuestion]:
    """The deal's last saved qualification-question generation, filtered
    down to fields still actually missing per deals_store.missing_fields
    (the live source of truth) -- a field answered since these questions
    were last generated should disappear from this list even though the
    saved blob itself hasn't been regenerated."""
    if deal_id is None:
        return []
    deal = get_deal(deal_id)
    if deal is None:
        return []
    raw = get_qualification(deal)
    if not raw:
        return []
    still_missing_keys = {key for key, _, _ in missing_fields(deal)}
    return [QualificationQuestion(**q) for q in raw if q["field_key"] in still_missing_keys]


def render_discovery_page() -> None:
    st.title("Pre-Call Discovery")
    st.caption(
        "Describe the customer's use case as you understand it so far. Get back documented "
        "Right-to-Win points -- each with its own evidence and exploratory questions -- not a "
        "product recommendation yet. Record the customer's answers during the call, then "
        "request a recommendation once you have enough of them."
    )

    deal_id, customer_name, company, deal_use_case = render_deal_picker("discovery")

    # On a deal switch, repopulate (not just clear) from this deal's own
    # last saved generation -- Right-to-Win, qualification, and
    # recommendation all persist to deals.db (see deals_store.save_discovery/
    # save_qualification/save_recommendation), so reopening a deal shows
    # what was last there instead of an empty page.
    if st.session_state.get("active_deal_id") != deal_id:
        set_active_deal(deal_id)
        saved_result, saved_answers = _load_saved_discovery(deal_id)
        st.session_state.discovery_result = saved_result
        st.session_state.discovery_answers = saved_answers
        st.session_state.discovery_recommendation = _load_saved_recommendation(deal_id)
        st.session_state.discovery_qualification = _load_saved_qualification(deal_id)

    # Everything below needs an active deal to attach to.
    if deal_id is None:
        st.info("Pick an existing deal, or start a new one above (Customer Name + Company), to continue.")
        return

    # Auto-saves the draft to the deal on every commit (losing focus, or
    # Ctrl+Enter) so switching tabs or clicking elsewhere never loses what's
    # been typed. Does not auto-trigger generation -- Streamlit's on_change
    # fires the same whether the box lost focus from Ctrl+Enter or simply
    # alt-tabbing away, so generation only ever happens from the explicit
    # button below.
    def _save_use_case_draft(deal_id: int | None, widget_key: str) -> None:
        if deal_id is not None:
            value = st.session_state.get(widget_key, "").strip()
            if value:
                update_deal_fields(deal_id, use_case=value)

    # Keyed per-deal (not a fixed key) so switching deals swaps in that
    # deal's own saved use case instead of Streamlit reusing whatever
    # text was last typed under a fixed widget key.
    use_case_key = f"discovery_use_case_box_{deal_id if deal_id is not None else 'new'}"
    use_case = st.text_area(
        "Customer use case",
        value=deal_use_case,
        placeholder=EXAMPLE_USE_CASE,
        height=120,
        key=use_case_key,
        on_change=_save_use_case_draft,
        args=(deal_id, use_case_key),
    )
    generate = st.button("Generate discovery questions", type="primary")

    if generate:
        if not use_case.strip():
            st.error("Describe the use case first.")
        else:
            try:
                with st.spinner("Checking approved documents..."):
                    matches, text_stream = stream_discovery_questions(use_case)
                    sources = dedupe_sources(matches)

                # Renders each Right-to-Win card as soon as it's complete,
                # instead of making a rep wait for the whole multi-point
                # reply -- only the point still being generated shows as raw
                # streaming text below the already-rendered cards. Also
                # saves to deals.db after each point completes (not just once
                # the whole reply is done), so navigating away mid-generation
                # never loses points that already finished streaming.
                live_answers: dict[str, str] = {}
                completed_points: list[RightToWinPoint] = []
                points_slot = st.container()
                live_slot = st.empty()
                raw_reply = ""
                for new_points, tail_text, raw_reply in stream_discovery_points(text_stream):
                    for point in new_points:
                        completed_points.append(point)
                        with points_slot:
                            _render_right_to_win_card(len(completed_points), point, live_answers)
                        save_discovery(
                            deal_id, [p.to_dict() for p in completed_points], live_answers, sources
                        )
                    with live_slot.container(border=theme.is_enterprise_theme()):
                        st.markdown(tail_text)
                live_slot.empty()
            except (RetrieverError, ResponseGeneratorError) as e:
                st.error(f"Something went wrong generating questions: {e}")
            else:
                # deal_id is guaranteed set here; the page returns early above when it's None.
                update_deal_fields(deal_id, use_case=use_case)
                set_active_deal(deal_id)
                result = finalize_discovery_questions(matches, raw_reply)
                st.session_state.discovery_result = result
                st.session_state.discovery_answers = {}
                st.session_state.pop("discovery_recommendation", None)
                save_discovery(deal_id, [p.to_dict() for p in result.right_to_win], {}, result.sources)

                # Which of this deal's 8 MEDDPICC fields are still blank,
                # and what to ask about them next -- deterministic field
                # selection, LLM-phrased question/rationale grounded in
                # the Golden Frameworks & Sales Tactics document. Never
                # blocks the page on failure -- Right-to-Win is the core
                # deliverable of this click; qualification is additive.
                try:
                    with st.spinner("Checking qualification status..."):
                        deal = get_deal(deal_id)
                        missing, qual_matches, qual_stream = stream_qualification_questions(deal)
                        qual_reply = "".join(qual_stream) if qual_stream is not None else ""
                    qualification = finalize_qualification_questions(missing, qual_reply) if missing else []
                    st.session_state.discovery_qualification = qualification
                    save_qualification(deal_id, [q.to_dict() for q in qualification])
                except (RetrieverError, ResponseGeneratorError):
                    st.session_state.discovery_qualification = []

                st.rerun()

    result = st.session_state.get("discovery_result")
    saved_recommendation = st.session_state.get("discovery_recommendation")

    # Only bail out with nothing at all to show -- a saved recommendation
    # must still render even without fresh Right-to-Win results in this
    # session.
    if not result and not saved_recommendation:
        return

    answers: dict[str, str] = st.session_state.setdefault("discovery_answers", {})

    if result:
        st.divider()

        if not result.right_to_win:
            st.info("No documented Right to Win identified from the available evidence.")
        else:
            st.markdown("**Potential Right to Win, ranked highest to lowest relevance**")
            for i, point in enumerate(result.right_to_win, start=1):
                _render_right_to_win_card(i, point, answers, deal_id=deal_id)

        if result.sources:
            with st.expander("Drawn from"):
                for name, _ in result.sources:
                    st.markdown(f"- `{name}`")

        # -------------------------------------------------------------
        # Qualification -- deterministic MEDDPICC gap-filling for the
        # active deal (see discovery_generator.stream_qualification_
        # questions). Answering one of these saves straight to
        # deals.db -- next time this deal is opened, that field won't
        # be asked about again.
        # -------------------------------------------------------------
        qualification: list[QualificationQuestion] = st.session_state.get("discovery_qualification") or []
        if qualification and deal_id is not None:
            st.divider()
            st.markdown("**Qualification -- what's still missing for this deal**")
            for q in qualification:
                _render_qualification_card(deal_id, q)
        elif deal_id is not None and st.session_state.get("discovery_qualification") == []:
            st.divider()
            st.caption("All 8 MEDDPICC qualification fields are already captured for this deal.")

        # ---------------------------------------------------------------
        # Stage 2: once enough answers are in, match them against the KB.
        # ---------------------------------------------------------------
        st.divider()
        st.markdown("**Product recommendation**")

        answered_count = sum(1 for a in answers.values() if a.strip())
        if answered_count < MIN_ANSWERS_FOR_RECOMMENDATION:
            st.caption(
                f"Record at least {MIN_ANSWERS_FOR_RECOMMENDATION} answers to enable a recommendation "
                f"({answered_count}/{MIN_ANSWERS_FOR_RECOMMENDATION} so far)."
            )

        recommend_clicked = st.button(
            "Generate recommendation from captured answers",
            disabled=answered_count < MIN_ANSWERS_FOR_RECOMMENDATION,
        )

        if recommend_clicked:
            all_questions = [q for point in result.right_to_win for q in point.questions]
            qa_pairs = [(q, answers.get(q, "")) for q in all_questions]
            try:
                with st.spinner("Matching answers against approved documents..."):
                    # Reads the use case straight from the deal record
                    # rather than a separate session-state copy -- one
                    # source of truth now that every recommendation has
                    # a real deal_id backing it.
                    use_case_for_recommendation = get_deal(deal_id)["use_case"] if deal_id is not None else ""
                    matches, qa_block, text_stream = stream_recommendation(
                        use_case_for_recommendation, qa_pairs
                    )
                if text_stream is None:
                    # Shouldn't happen -- the button above is disabled
                    # below the answer floor -- but stay honest if it
                    # somehow does.
                    recommendation = insufficient_recommendation(answered_count)
                else:
                    with st.container(border=theme.is_enterprise_theme()):
                        raw_reply = st.write_stream(text_stream)
                    recommendation = finalize_recommendation(matches, qa_block, raw_reply)
            except (RetrieverError, ResponseGeneratorError) as e:
                st.error(f"Something went wrong generating a recommendation: {e}")
            else:
                st.session_state.discovery_recommendation = recommendation
                if deal_id is not None:
                    save_recommendation(deal_id, recommendation.to_dict())
                st.rerun()
    else:
        st.divider()
        st.caption(
            "Showing this deal's last saved recommendation. Describe the use case and generate "
            "Right-to-Win questions above to refresh it."
        )

    recommendation = st.session_state.get("discovery_recommendation")
    if recommendation:
        if recommendation.confirmed_priorities:
            st.markdown("**Confirmed priorities (from the customer's actual answers)**")
            for p in recommendation.confirmed_priorities:
                st.markdown(f"- {p}")

        if recommendation.outcome == "Recommend":
            _render_recommend_outcome(recommendation.product)
        else:
            render_outcome = OUTCOME_RENDER.get(recommendation.outcome, st.info)
            render_outcome(f"Outcome: {recommendation.outcome}")

        if recommendation.outcome == "Trade-offs" and recommendation.candidate_products:
            st.markdown(
                "**Candidates:** " + " vs. ".join(f"`{p}`" for p in recommendation.candidate_products)
            )

        st.write(recommendation.recommendation)

        if recommendation.outcome != "Recommend" and recommendation.missing_information:
            if recommendation.missing_information.strip().lower() != "none":
                st.markdown("**What's still needed**")
                st.write(recommendation.missing_information)

        if recommendation.risk and recommendation.risk != "None":
            st.caption(
                f"Risk flag: {recommendation.risk} -- verify this recommendation before sharing "
                "it with the customer."
            )

        if recommendation.sources:
            with st.expander("Recommendation drawn from"):
                for name, _ in recommendation.sources:
                    st.markdown(f"- `{name}`")
