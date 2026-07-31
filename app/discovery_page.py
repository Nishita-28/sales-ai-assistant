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
    generate_discovery_questions,
    generate_recommendation,
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


def render_discovery_page() -> None:
    st.title("Pre-Call Discovery")
    st.caption(
        "Describe the customer's use case as you understand it so far. Get back documented "
        "Right-to-Win points -- each with its own evidence and exploratory questions -- not a "
        "product recommendation yet. Record the customer's answers during the call, then "
        "request a recommendation once you have enough of them."
    )

    use_case = st.text_area(
        "Customer use case",
        placeholder=EXAMPLE_USE_CASE,
        height=120,
    )
    generate = st.button("Generate discovery questions", type="primary")

    if generate:
        if not use_case.strip():
            st.error("Describe the use case first.")
        else:
            with st.spinner("Checking approved documents..."):
                try:
                    result = generate_discovery_questions(use_case)
                except (RetrieverError, ResponseGeneratorError) as e:
                    st.error(f"Something went wrong generating questions: {e}")
                else:
                    st.session_state.discovery_result = result
                    st.session_state.discovery_use_case = use_case
                    st.session_state.discovery_answers = {}
                    st.session_state.pop("discovery_recommendation", None)

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
            with st.container(border=theme.is_enterprise_theme()):
                st.markdown(f"**{i}. {point.title}**")
                if point.evidence:
                    st.caption(point.evidence)
                for j, q in enumerate(point.questions, start=1):
                    st.checkbox(q, key=f"dq-asked-{i}-{j}")
                    answers[q] = st.text_input(
                        "Answer",
                        value=answers.get(q, ""),
                        key=f"dq-ans-{i}-{j}",
                        label_visibility="collapsed",
                        placeholder="Customer's answer (leave blank if not asked yet)",
                    )
            if not theme.is_enterprise_theme():
                st.markdown("")

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
        with st.spinner("Matching answers against approved documents..."):
            try:
                recommendation = generate_recommendation(
                    st.session_state.get("discovery_use_case", ""), qa_pairs
                )
            except (RetrieverError, ResponseGeneratorError) as e:
                st.error(f"Something went wrong generating a recommendation: {e}")
            else:
                st.session_state.discovery_recommendation = recommendation

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
