"""Pre-call discovery workflow: a rep describes a prospective customer's use
case, gets back documented "Right to Win" points -- each with its own
supporting evidence and exploratory questions -- records the customer's
actual answers during the call, and then -- once there's enough evidence --
gets a knowledge-base-grounded product recommendation (or an honest "not
enough evidence yet" with what to ask next). One continuous workflow, not a
separate feature.
"""
from __future__ import annotations

import streamlit as st

from app import theme
from app.discovery_generator import (
    MIN_ANSWERS_FOR_RECOMMENDATION,
    RightToWinPoint,
    finalize_discovery_questions,
    finalize_recommendation,
    insufficient_recommendation,
    stream_discovery_points,
    stream_discovery_questions,
    stream_recommendation,
)
from app.response_generator import ResponseGeneratorError
from app.retriever import RetrieverError

EXAMPLE_USE_CASE = (
    "Customer runs a lead-acid battery room in a warehouse and wants to monitor "
    "for hydrogen buildup. Not sure yet if the area is classified hazardous."
)

OUTCOME_RENDER = {
    "Recommend": st.success,
    "Trade-offs": st.info,
    "Insufficient": st.warning,
}


def _render_right_to_win_card(i: int, point: RightToWinPoint, answers: dict[str, str]) -> None:
    """One Right-to-Win card -- title, evidence, and one answer-capture
    box per question. Shared by the live (mid-stream) render and the
    persisted post-generation render so the two look identical and a
    typed-ahead answer survives the switch between them (same `answers`
    dict, same widget key scheme)."""
    with st.container(border=theme.is_enterprise_theme()):
        st.markdown(f"**{i}. {point.title}**")
        if point.evidence:
            st.caption(point.evidence)
        for j, q in enumerate(point.questions, start=1):
            answers[q] = st.text_input(
                q,
                value=answers.get(q, ""),
                key=f"dq-ans-{i}-{j}",
                placeholder="Customer's answer (leave blank if not asked yet)",
            )
    if not theme.is_enterprise_theme():
        st.markdown("")


def render_discovery_page() -> None:
    st.title("Pre-Call Discovery")
    st.caption(
        "Describe the customer's use case as you understand it so far. Get back documented "
        "Right-to-Win points -- each with its own evidence and exploratory questions -- not a "
        "product recommendation yet. Record the customer's answers during the call, then "
        "request a recommendation once you have enough of them."
    )

    # Ctrl+Enter in a text_area only commits the typed text into the
    # widget (Streamlit reserves plain Enter for line breaks in a
    # multi-line box) -- it doesn't click a separate button on its own.
    # on_change fires on that same commit (Ctrl+Enter, or clicking away),
    # so wiring it to set this flag makes Ctrl+Enter actually trigger
    # generation instead of silently doing nothing visible.
    def _mark_auto_generate() -> None:
        st.session_state.discovery_auto_generate = True

    use_case = st.text_area(
        "Customer use case",
        placeholder=EXAMPLE_USE_CASE,
        height=120,
        key="discovery_use_case_box",
        on_change=_mark_auto_generate,
    )
    generate = st.button("Generate discovery questions", type="primary")
    generate = generate or st.session_state.pop("discovery_auto_generate", False)

    if generate:
        if not use_case.strip():
            st.error("Describe the use case first.")
        else:
            try:
                with st.spinner("Checking approved documents..."):
                    matches, text_stream = stream_discovery_questions(use_case)

                # Renders each Right-to-Win card (title, evidence, answer
                # boxes) the moment it's actually complete, instead of
                # making a rep wait for the whole multi-point reply --
                # only the point still being generated shows as raw
                # streaming text, in its own box below whatever's already
                # been rendered as real cards.
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
                    with live_slot.container(border=theme.is_enterprise_theme()):
                        st.markdown(tail_text)
                live_slot.empty()
            except (RetrieverError, ResponseGeneratorError) as e:
                st.error(f"Something went wrong generating questions: {e}")
            else:
                st.session_state.discovery_result = finalize_discovery_questions(matches, raw_reply)
                st.session_state.discovery_use_case = use_case
                st.session_state.discovery_answers = {}
                st.session_state.pop("discovery_recommendation", None)
                st.rerun()

    result = st.session_state.get("discovery_result")
    if not result:
        return

    st.divider()

    answers: dict[str, str] = st.session_state.setdefault("discovery_answers", {})

    if not result.right_to_win:
        st.info("No documented Right to Win identified from the available evidence.")
    else:
        st.markdown("**Potential Right to Win, ranked highest to lowest relevance**")
        for i, point in enumerate(result.right_to_win, start=1):
            _render_right_to_win_card(i, point, answers)

    if result.sources:
        with st.expander("Drawn from"):
            for name, _ in result.sources:
                st.markdown(f"- `{name}`")

    # -----------------------------------------------------------------
    # Stage 2: once enough answers are in, match them against the KB.
    # -----------------------------------------------------------------
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
                matches, qa_block, text_stream = stream_recommendation(
                    st.session_state.get("discovery_use_case", ""), qa_pairs
                )
            if text_stream is None:
                # Shouldn't happen -- the button above is disabled below the
                # answer floor -- but stay honest if it somehow does.
                recommendation = insufficient_recommendation(answered_count)
            else:
                with st.container(border=theme.is_enterprise_theme()):
                    raw_reply = st.write_stream(text_stream)
                recommendation = finalize_recommendation(matches, qa_block, raw_reply)
        except (RetrieverError, ResponseGeneratorError) as e:
            st.error(f"Something went wrong generating a recommendation: {e}")
        else:
            st.session_state.discovery_recommendation = recommendation
            st.rerun()

    recommendation = st.session_state.get("discovery_recommendation")
    if recommendation:
        if recommendation.confirmed_priorities:
            st.markdown("**Confirmed priorities (from the customer's actual answers)**")
            for p in recommendation.confirmed_priorities:
                st.markdown(f"- {p}")

        render_outcome = OUTCOME_RENDER.get(recommendation.outcome, st.info)
        render_outcome(f"Outcome: {recommendation.outcome}")
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
