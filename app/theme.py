"""Enterprise UI theme for the Streamlit app.

Toggleable via the UI_THEME env var so the redesign can be reverted
instantly without touching any page's logic:
  UI_THEME=enterprise (default) -- the navy-sidebar/card redesign.
  UI_THEME=classic              -- the original look, exactly as before.

Every page calls inject_theme() once; it's a no-op under "classic". Badge
rendering is centralized here too, so all pages render confidence/risk
pills consistently.
"""
from __future__ import annotations

import base64
import functools
import os
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).parent / "assets"

UI_THEME = os.environ.get("UI_THEME", "enterprise").strip().lower()


def is_enterprise_theme() -> bool:
    return UI_THEME == "enterprise"


def apply_native_theme_option() -> None:
    """Sets Streamlit's own theme.primaryColor at runtime (no-op under
    classic). Needed on top of inject_theme()'s CSS: canvas-rendered
    widgets like st.data_editor's selected-cell outline (glide-data-grid)
    read Streamlit's native theme, not the DOM, so CSS can't recolor them --
    only this reaches them. Must run before st.set_page_config()."""
    if not is_enterprise_theme():
        return
    try:
        st._config.set_option("theme.primaryColor", "#184fa3")
    except Exception:
        pass


@functools.lru_cache(maxsize=None)
def _b64(filename: str) -> str:
    return base64.b64encode((ASSETS_DIR / filename).read_bytes()).decode("ascii")


def logo_data_uri() -> str:
    return f"data:image/png;base64,{_b64('logo.png')}"


# Badge color tokens, shared across pages -- keeps Assistant/Discovery/Sales
# Aids visually consistent instead of each page inventing its own palette.
_BADGE_COLORS = {
    "good": ("#e8f6ee", "#157a4f"),
    # Brand blue, not green -- kept distinct from "good" (used elsewhere
    # for Confidence badges) rather than recoloring that shared kind, so
    # only the deal-status badges below are affected.
    "good-blue": ("#edf3fb", "#123c80"),
    "warn": ("#fbf1de", "#96650f"),
    "crit": ("#fbebea", "#ab281f"),
    "neutral": ("#eef0f3", "#5b5f66"),
}

_RISK_KIND = {
    "None": "good",
    "Certification": "warn",
    "Accuracy": "warn",
    "Pricing": "warn",
    "Delivery": "warn",
    "Safety": "crit",
    "Legal": "crit",
    "Unknown": "neutral",
}


def _badge_html(label: str, kind: str) -> str:
    bg, fg = _BADGE_COLORS[kind]
    return (
        f'<span class="mnst-badge" style="background:{bg};color:{fg};">'
        f'<span class="mnst-badge-led" style="background:{fg};"></span>{label}</span>'
    )


def render_badges_html(confidence: str, risk: str) -> str:
    """Returns the HTML for a confidence + risk badge pair -- pass to
    st.markdown(..., unsafe_allow_html=True)."""
    conf_badge = _badge_html(f"Confidence: {confidence}", "good" if confidence == "High" else "warn")
    risk_kind = _RISK_KIND.get(risk, "neutral")
    risk_badge = _badge_html(f"Risk: {risk}", risk_kind)
    return f'<div class="mnst-badge-row">{conf_badge}{risk_badge}</div>'


_RECOMMENDATION_OUTCOME_KIND = {
    "Recommend": "good-blue",
    "Trade-offs": "warn",
    "Insufficient": "warn",
}


def render_deal_status_badges(
    recommendation_outcome: str | None, requirements_count: int, sales_aids_count: int
) -> str:
    """Returns the HTML for a deal's at-a-glance health row -- pass to
    st.markdown(..., unsafe_allow_html=True). Lets an admin scanning the
    Deals tab tell "fully worked" from "nothing linked yet" without
    expanding every deal's requirement/sales-aid lists one by one."""
    if recommendation_outcome is None:
        rec_badge = _badge_html("No recommendation yet", "neutral")
    else:
        kind = _RECOMMENDATION_OUTCOME_KIND.get(recommendation_outcome, "neutral")
        rec_badge = _badge_html(f"Recommendation outcome: {recommendation_outcome}", kind)
    req_badge = _badge_html(
        f"Requirements: {requirements_count}", "good-blue" if requirements_count else "neutral"
    )
    aid_badge = _badge_html(
        f"Sales Aids: {sales_aids_count}", "good-blue" if sales_aids_count else "neutral"
    )
    return f'<div class="mnst-badge-row">{rec_badge}{req_badge}{aid_badge}</div>'


def render_logo_html() -> str:
    return (
        '<div class="mnst-brand">'
        f'<img src="{logo_data_uri()}" alt="Multi Nano Sense" />'
        '<div class="mnst-brand-word">MNST<span>Sales Assistant</span></div>'
        "</div>"
    )


_CSS = """
@font-face {{ font-family: "Ubuntu"; font-weight: 400; font-style: normal; src: url(data:font/ttf;base64,{u400}) format("truetype"); }}
@font-face {{ font-family: "Ubuntu"; font-weight: 500; font-style: normal; src: url(data:font/ttf;base64,{u500}) format("truetype"); }}
@font-face {{ font-family: "Ubuntu"; font-weight: 700; font-style: normal; src: url(data:font/ttf;base64,{u700}) format("truetype"); }}

:root {{
  --mnst-page: #f5f8fc;
  --mnst-surface: #ffffff;
  --mnst-sidebar: #071c3b;
  --mnst-ink-900: #15171a;
  --mnst-ink-700: #40444c;
  --mnst-ink-500: #6d7178;
  --mnst-border: #e8e8e8;
  --mnst-border-strong: #d7d9dd;
  --mnst-blue: #184fa3;
  --mnst-blue-600: #123c80;
  --mnst-blue-50: #edf3fb;
  --mnst-rail-ink-900: #ffffff;
  --mnst-rail-ink-600: #b7c6e2;
  --mnst-rail-ink-400: #7288ac;
  --mnst-rail-border: #14294f;
  --mnst-rail-active-bg: #12305e;
}}

html, body, .stApp {{ font-family: "Ubuntu", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif !important; }}
.stApp {{ background: var(--mnst-page); }}
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ font-weight: 700 !important; letter-spacing: -0.01em; color: var(--mnst-ink-900); }}
p, .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--mnst-ink-500); }}

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"] {{ background: var(--mnst-sidebar); border-right: 1px solid var(--mnst-rail-border); }}
[data-testid="stSidebar"] * {{ color: var(--mnst-rail-ink-600); }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: var(--mnst-rail-ink-900); }}
[data-testid="stSidebar"] hr {{ border-color: var(--mnst-rail-border); }}

/* Layout goal: logo at the very top, then nav, then the knowledge-base
   status panel at the bottom. Real constraint (confirmed via computed-
   style inspection): stSidebarContent has THREE flex children --
   stSidebarHeader (collapse toggle), stSidebarNav (auto page list), and
   stSidebarUserContent (everything from `with st.sidebar:`, logo AND
   status panel glued into ONE block since they're rendered in the same
   Python block) -- always in that DOM order, independent of script order.
   flexbox `order` can only reorder these three whole blocks, it can't
   interleave nav BETWEEN the logo and the status panel that live inside
   the same block. So the logo is pulled out of flow with
   position:absolute and drawn over reserved top padding instead; nav
   then order:0, and the (logo-minus, status-panel-only-visually) user
   content block order:1 so it lands after nav, at the bottom. */
[data-testid="stSidebarContent"] {{
  display: flex !important; flex-direction: column !important; justify-content: flex-start !important;
  position: relative !important; padding-top: 84px !important;
}}
[data-testid="stSidebarHeader"] {{ position: absolute !important; top: 6px !important; right: 6px !important; z-index: 10 !important; }}
[data-testid="stSidebarNav"] {{ order: 0 !important; flex-grow: 0 !important; margin-top: 0 !important; }}
[data-testid="stSidebarUserContent"] {{
  order: 1 !important; flex-grow: 0 !important; flex-shrink: 0 !important;
  height: auto !important; min-height: 0 !important;
  /* auto margin (not a fixed px value) pushes this block all the way to
     the bottom of the flex column, consuming the leftover space instead
     of leaving it empty below the panel -- the actual fix for "put it at
     the bottom", not just "put it after nav". */
  margin-top: auto !important; margin-bottom: 20px !important;
  padding-top: 16px !important; border-top: 1px solid var(--mnst-rail-border) !important;
}}

/* position:fixed (anchored to the viewport corner) rather than absolute --
   testing showed the "nearest positioned ancestor" for absolute here
   resolves to a nested Streamlit wrapper inside stSidebarUserContent
   (itself pushed down near the bottom by the order:1 rule above), not
   the outer stSidebarContent, so the logo rendered overlapping the
   status panel instead of at the top. Fixed positioning anchors to the
   viewport directly and sidesteps that ambiguity. */
.mnst-brand {{
  position: fixed !important; top: 18px; left: 18px; width: 224px; z-index: 20;
  display: flex; align-items: center; gap: 12px; padding: 0; margin: 0; border-bottom: none;
}}
.mnst-brand img {{ width: 36px; height: 36px; display: block; background: #fff; border-radius: 7px; padding: 3px; }}
.mnst-brand-word {{ font-size: 16.5px; font-weight: 700; line-height: 1.3; color: var(--mnst-rail-ink-900) !important; }}
.mnst-brand-word span {{ display: block; font-size: 11.5px; font-weight: 500; color: var(--mnst-rail-ink-400) !important; letter-spacing: 0.04em; text-transform: uppercase; margin-top: 2px; }}

/* Streamlit's auto-generated page nav links -- data-testid="stSidebarNavLink"
   is the <a> itself (not a child of some "stSidebarNav" wrapper), and it
   carries aria-current="page" directly when active, so the active-state
   selector must match on that element, not a nonexistent ancestor. */
/* Streamlit nests the visible label 2-3 levels deep (span > div > p), not
   directly in a span, so text color has to be forced on every descendant
   with a universal selector -- targeting just "span" or "p" alone missed
   the actual <p data-testid="stMarkdownContainer"> text node in testing. */
[data-testid="stSidebarNavLink"], [data-testid="stSidebarNavLink"] * {{
  color: var(--mnst-rail-ink-600) !important; font-weight: 500 !important; font-size: 15px !important;
}}
[data-testid="stSidebarNavLink"] {{
  border-radius: 8px !important; border-left: 2px solid transparent !important;
  padding-top: 12px !important; padding-bottom: 12px !important;
  user-select: none !important;
}}
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {{ font-size: 20px !important; }}
[data-testid="stSidebarNavLink"]:hover, [data-testid="stSidebarNavLink"]:hover * {{
  background: #0d2547 !important; color: #fff !important;
}}
/* Background goes on the link AND every descendant (icon, text span),
   not just the link itself -- otherwise the :hover rule above (which
   also paints every descendant) can win on the icon/text spans while
   this rule wins on the outer link, since clicking a nav item makes it
   the current page while the mouse is still resting on it (hover never
   actually ends) -- two different backgrounds layered on top of each
   other read as a separate highlighted box around just the text. */
[data-testid="stSidebarNavLink"][aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] * {{
  color: #fff !important; background: var(--mnst-rail-active-bg) !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  border-left: 2px solid #ffffff !important;
}}

/* Knowledge-base status block, pinned to the bottom of the sidebar via
   the margin-top:auto on its parent (stSidebarUserContent) above. */
.mnst-status {{ padding: 16px 14px; border: 1px solid var(--mnst-rail-border); border-radius: 12px; background: #0a2040; display: flex; flex-direction: column; gap: 12px; margin-top: 8px; }}
.mnst-status-row {{ display: flex; align-items: center; justify-content: space-between; font-size: 13px; color: var(--mnst-rail-ink-400); }}
.mnst-status-row .left {{ display: flex; align-items: center; gap: 8px; }}
.mnst-status-row .led {{ width: 7px; height: 7px; border-radius: 50%; background: #ffffff; flex-shrink: 0; display: inline-block; }}
.mnst-status-row .val {{ color: var(--mnst-rail-ink-900); font-weight: 600; }}

/* ---------------- Buttons ---------------- */
/* [data-testid^="stBaseButton"] is the real element in current Streamlit
   (stBaseButton-secondary / stBaseButton-primary); .stButton > button
   kept as a fallback for older versions. */
[data-testid^="stBaseButton"], [data-testid^="stBaseButton"] *, .stButton > button, .stFormSubmitButton > button {{
  color: var(--mnst-ink-700) !important;
}}
[data-testid^="stBaseButton"], .stButton > button, .stFormSubmitButton > button {{
  border-radius: 8px !important; font-weight: 600 !important; border: 1px solid var(--mnst-border-strong) !important;
  background: var(--mnst-surface) !important;
}}
[data-testid^="stBaseButton"]:hover, [data-testid^="stBaseButton"]:hover *,
.stButton > button:hover, .stFormSubmitButton > button:hover {{
  border-color: var(--mnst-blue) !important; color: var(--mnst-blue) !important;
}}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] *,
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  color: #fff !important;
}}
[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background: var(--mnst-blue) !important; border-color: var(--mnst-blue) !important;
}}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primary"]:hover *,
.stButton > button[kind="primary"]:hover {{
  background: var(--mnst-blue-600) !important; border-color: var(--mnst-blue-600) !important; color: #fff !important;
}}

/* ---------------- Inputs ---------------- */
/* Streamlit's newer (react-aria) text inputs put the VISIBLE border/
   background on stTextInputRootElement, a wrapper div -- not on the raw
   <input> itself, which renders borderless inside it. Styling only the
   <input> left Streamlit's own default red-accent focus ring on the
   wrapper showing through underneath our blue one (confirmed via DOM
   inspection -- this was a real, visible bug, not a guess). So the
   wrapper is the primary target now; the inner input is made borderless
   so it can't show a second, conflicting outline of its own. */
[data-testid="stTextInputRootElement"], [data-testid="stTextArea"] > div {{
  border-radius: 10px !important; border: 1px solid var(--mnst-border-strong) !important; background: var(--mnst-surface) !important;
}}
[data-testid="stTextInputRootElement"]:has(input:focus), [data-testid="stTextArea"] > div:has(textarea:focus) {{
  border-color: var(--mnst-blue) !important; box-shadow: 0 0 0 3px var(--mnst-blue-50) !important;
}}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stChatInput"] textarea {{
  border: none !important; outline: none !important; box-shadow: none !important; background: transparent !important;
}}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {{ color: var(--mnst-ink-700) !important; font-weight: 500 !important; }}

/* ---------------- Tabs (e.g. Admin page) ---------------- */
/* The active-tab underline is a separate element, .react-aria-SelectionIndicator,
   only rendered on the selected [data-testid="stTab"] -- confirmed via DOM
   inspection (Admin > Approved Claims tab underline was red by default). */
[data-testid="stTab"] .react-aria-SelectionIndicator {{ background-color: var(--mnst-blue) !important; }}
[data-testid="stTab"][aria-selected="true"] {{ color: var(--mnst-blue) !important; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: var(--mnst-blue) !important; font-weight: 600 !important; }}

/* ---------------- Form controls (multiselect / radio / checkbox) ---------------- */
/* Streamlit's stock accent for selected controls is red (#FF4B4B, its
   default primaryColor) since no custom theme was ever configured -- that
   showed up as orange multiselect chips and radio dots, confirmed via a
   real selection on the Customer Requirements form. */
[data-baseweb="tag"] {{ background-color: var(--mnst-blue) !important; }}
/* Real structure (confirmed via full DOM dump, not guessed): a selected
   stRadioOption is label > div > div > div(ring) > div(fill), with the
   text label (stMarkdownContainer, also a <div>) as a SIBLING at the same
   3-levels-deep position as the ring -- "div div div" matched both the
   ring AND the text container and painted a blue box behind the label
   text too. The ring needs :not() to exclude stMarkdownContainer at that
   same depth; the fill dot one level deeper doesn't have that ambiguity,
   since the text branch terminates at <p>, not another <div>. */
[data-testid="stRadioOption"][data-selected="true"] div div div:not([data-testid="stMarkdownContainer"]) {{
  background: var(--mnst-blue) !important; border-color: var(--mnst-blue) !important;
}}
[data-testid="stRadioOption"][data-selected="true"] div div div div {{ background: var(--mnst-blue) !important; border-color: var(--mnst-blue) !important; }}
[data-testid="stCheckbox"] svg {{ fill: var(--mnst-blue) !important; }}
[data-testid="stCheckbox"] input:checked ~ div {{ background: var(--mnst-blue) !important; border-color: var(--mnst-blue) !important; }}

/* ---------------- Cards (bordered containers) ---------------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 12px !important; border: 1px solid var(--mnst-border) !important; background: var(--mnst-surface) !important;
  box-shadow: 0 1px 2px rgba(15,17,20,0.04), 0 2px 8px rgba(15,17,20,0.04) !important;
}}

/* ---------------- Expander (Sources) ---------------- */
[data-testid="stExpander"] {{ border: 1px solid var(--mnst-border) !important; border-radius: 10px !important; background: var(--mnst-surface) !important; }}
[data-testid="stExpander"] summary {{ font-weight: 600 !important; color: var(--mnst-ink-500) !important; font-size: 13px !important; }}

/* ---------------- Badges (confidence / risk pills) ---------------- */
.mnst-badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 2px; }}
.mnst-badge {{
  font-family: "Ubuntu", sans-serif; font-size: 11.5px; font-weight: 700; letter-spacing: 0.02em;
  padding: 4px 11px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;
}}
.mnst-badge-led {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
"""


def inject_theme() -> None:
    """Injects the enterprise theme CSS. No-op under UI_THEME=classic, so
    reverting to the original look is a one-line env var change -- no
    files to restore, no page logic touched."""
    if not is_enterprise_theme():
        return
    css = _CSS.format(u400=_b64("ubuntu-400.ttf"), u500=_b64("ubuntu-500.ttf"), u700=_b64("ubuntu-700.ttf"))
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
