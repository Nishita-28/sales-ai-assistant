"""Enterprise UI theme for the Streamlit app -- the navy-sidebar/card
redesign, applied unconditionally. Badge rendering is centralized here too,
so all pages render confidence/risk pills consistently.
"""
from __future__ import annotations

import base64
import functools
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).parent / "assets"


def apply_native_theme_option() -> None:
    """Sets Streamlit's own native theme options at runtime. Needed on top
    of inject_theme()'s CSS: canvas-rendered widgets -- st.data_editor /
    st.dataframe (glide-data-grid) -- read colors from Streamlit's native
    theme config, not the DOM, so CSS alone can't reach them. Must run
    before st.set_page_config()."""
    try:
        st._config.set_option("theme.primaryColor", "#184fa3")
        st._config.set_option("theme.backgroundColor", "#0a1730")
        st._config.set_option("theme.secondaryBackgroundColor", "#0d1d3d")
        st._config.set_option("theme.textColor", "#f4f7fc")
    except Exception:
        pass


@functools.lru_cache(maxsize=None)
def _b64(filename: str) -> str:
    return base64.b64encode((ASSETS_DIR / filename).read_bytes()).decode("ascii")


def logo_data_uri() -> str:
    return f"data:image/png;base64,{_b64('logo.png')}"


def background_url() -> str:
    """A plain URL, not a base64 data URI -- embedding the image as base64
    would resend it in full on every rerun, badly slowing cold starts.
    Served from app/static/ instead (requires server.enableStaticServing,
    set in .streamlit/config.toml), which the browser fetches once and
    caches. Relative path so it still resolves under a URL subpath."""
    return "app/static/sidebar-bg.png"


# Badge color tokens, shared across pages -- keeps Assistant/Discovery/Sales
# Aids visually consistent instead of each page inventing its own palette.
_BADGE_COLORS = {
    "good": ("#e8f6ee", "#157a4f"),
    # Distinct from "good" (used for Confidence badges) so only deal-status badges are affected.
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
/* font-display: swap avoids text staying invisible while this base64-embedded
   font decodes on each rerun -- it paints in a fallback font immediately,
   then swaps to Ubuntu once ready. */
@font-face {{ font-family: "Ubuntu"; font-weight: 400; font-style: normal; font-display: swap; src: url(data:font/ttf;base64,{u400}) format("truetype"); }}
@font-face {{ font-family: "Ubuntu"; font-weight: 500; font-style: normal; font-display: swap; src: url(data:font/ttf;base64,{u500}) format("truetype"); }}
@font-face {{ font-family: "Ubuntu"; font-weight: 700; font-style: normal; font-display: swap; src: url(data:font/ttf;base64,{u700}) format("truetype"); }}

:root {{
  /* Dark theme throughout -- the app sits on a fixed background image (see
     .stApp below), so every token here targets light text on translucent
     panels over a dark photo. --mnst-page is the fallback color while the
     image loads. */
  --mnst-page: #050e1f;
  --mnst-surface: rgba(9, 20, 42, 0.74);
  --mnst-surface-solid: #0a1730;
  --mnst-sidebar: rgba(4, 10, 22, 0.86);
  --mnst-ink-900: #f4f7fc;
  --mnst-ink-700: #c9d4e8;
  --mnst-ink-500: #93a3c4;
  --mnst-border: rgba(255, 255, 255, 0.5);
  --mnst-border-strong: rgba(255, 255, 255, 0.9);
  --mnst-blue: #3f7fdb;
  --mnst-blue-600: #5c95ea;
  --mnst-blue-50: rgba(63, 127, 219, 0.28);
  --mnst-rail-ink-900: #ffffff;
  --mnst-rail-ink-600: #b7c6e2;
  --mnst-rail-ink-400: #7288ac;
  --mnst-rail-border: rgba(255, 255, 255, 0.1);
  /* Solid, not translucent -- an alpha-blended rectangle over a busy
     background can show a faint anti-aliased rim at its edge, reading as
     an unexplained box. An opaque color has no edge to blend. */
  --mnst-rail-active-bg: #1f4e8a;
}}

html, body, .stApp {{ font-family: "Ubuntu", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif !important; }}
/* Longhand properties, not the `background` shorthand, which Chromium
   silently drops once an embedded base64 image pushes the declaration past
   roughly 2MB. */
/* A flat dark scrim (same color as --mnst-page) is layered over the image
   itself so its bright detail reads as background atmosphere rather than
   competing with the UI content for attention. */
.stApp {{
  background-color: var(--mnst-page) !important;
  background-image: linear-gradient(rgba(5, 14, 31, 0.72), rgba(5, 14, 31, 0.72)), url("{bg}") !important;
  background-position: center center !important;
  background-size: cover !important;
  background-repeat: no-repeat !important;
  background-attachment: fixed !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
/* !important needed: Streamlit's base stylesheet sets its own dark text
   color that otherwise wins the cascade. */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ font-weight: 700 !important; letter-spacing: -0.01em; color: var(--mnst-ink-900) !important; }}
/* Paragraphs (the AI's actual answer) get the brightest ink tone and a
   heavier weight since they're the primary content; true captions keep
   the dim tone. */
p, .stMarkdown p {{ color: var(--mnst-ink-900) !important; font-weight: 500 !important; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--mnst-ink-500) !important; }}

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"] {{ background: var(--mnst-sidebar); border-right: 1px solid var(--mnst-rail-border); }}
[data-testid="stSidebar"] * {{ color: var(--mnst-rail-ink-600); }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: var(--mnst-rail-ink-900); }}
[data-testid="stSidebar"] hr {{ border-color: var(--mnst-rail-border); }}

/* Layout goal: logo at the top, then nav, then the status panel at the
   bottom. stSidebarContent has three flex children in fixed DOM order --
   stSidebarHeader, stSidebarNav, and stSidebarUserContent (logo and status
   panel combined, since both come from the same `with st.sidebar:` block).
   flexbox `order` can reorder these three blocks but can't interleave nav
   between the logo and status panel inside the same block, so the logo is
   pulled out of flow with position:absolute over reserved top padding, nav
   gets order:0, and the remaining status-panel content gets order:1. */
[data-testid="stSidebarContent"] {{
  display: flex !important; flex-direction: column !important; justify-content: flex-start !important;
  position: relative !important; padding-top: 84px !important;
}}
[data-testid="stSidebarHeader"] {{ position: absolute !important; top: 6px !important; right: 6px !important; z-index: 10 !important; }}
[data-testid="stSidebarNav"] {{ order: 0 !important; flex-grow: 0 !important; margin-top: 0 !important; }}
[data-testid="stSidebarUserContent"] {{
  order: 1 !important; flex-grow: 0 !important; flex-shrink: 0 !important;
  height: auto !important; min-height: 0 !important;
  /* auto margin (not a fixed value) pushes this block to the bottom of the flex column. */
  margin-top: auto !important; margin-bottom: 20px !important;
  padding-top: 16px !important; border-top: 1px solid var(--mnst-rail-border) !important;
}}

/* position:fixed anchors to the viewport directly, avoiding ambiguity about
   which ancestor is "nearest positioned" inside the reordered flex layout
   above. */
.mnst-brand {{
  position: fixed !important; top: 18px; left: 18px; width: 224px; z-index: 20;
  display: flex; align-items: center; gap: 12px; padding: 0; margin: 0; border-bottom: none;
}}
.mnst-brand img {{ width: 36px; height: 36px; display: block; background: #fff; border-radius: 7px; padding: 3px; }}
.mnst-brand-word {{ font-size: 16.5px; font-weight: 700; line-height: 1.3; color: var(--mnst-rail-ink-900) !important; }}
.mnst-brand-word span {{ display: block; font-size: 11.5px; font-weight: 500; color: var(--mnst-rail-ink-400) !important; letter-spacing: 0.04em; text-transform: uppercase; margin-top: 2px; }}

/* stSidebarNavLink is the <a> itself and carries aria-current="page"
   directly, so the active-state selector targets it, not an ancestor. */
/* The visible label is nested 2-3 levels deep, so text color needs a
   universal descendant selector rather than targeting a specific tag. */
[data-testid="stSidebarNavLink"], [data-testid="stSidebarNavLink"] * {{
  color: var(--mnst-rail-ink-600) !important; font-weight: 500 !important; font-size: 15px !important;
}}
[data-testid="stSidebarNavLink"] {{
  border-radius: 8px !important; border-left: 2px solid transparent !important;
  padding-top: 12px !important; padding-bottom: 12px !important;
  user-select: none !important;
}}
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {{ font-size: 20px !important; }}
/* Streamlit's own base stylesheet paints the current-page link background as
   rgba(151, 166, 195, 0.25) via the `background` shorthand -- overriding
   with `background-color` alone doesn't win the cascade against it. */
/* Background is painted on the <a> only, not descendants -- painting the
   same translucent color on both a parent and a nested child stacks two
   semi-transparent layers, compositing to a visibly denser rectangle around
   the text. Descendants get an explicit transparent background instead. */
[data-testid="stSidebarNavLink"]:hover {{
  background: rgba(63, 127, 219, 0.14) !important; color: #fff !important;
}}
[data-testid="stSidebarNavLink"]:hover * {{
  background: transparent !important; color: #fff !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: var(--mnst-rail-active-bg) !important; color: #fff !important; font-weight: 700 !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] * {{
  background: transparent !important; color: #fff !important; font-weight: 700 !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  border-left: 2px solid #ffffff !important;
}}
/* Suppresses the browser's default focus ring, redundant with the
   border-left active-state marker and far more visible on this dark theme. */
[data-testid="stSidebarNavLink"]:focus, [data-testid="stSidebarNavLink"]:focus-visible {{
  outline: none !important; box-shadow: none !important;
}}
/* Defensive: opts this element out of forced-colors/high-contrast mode
   overriding author styles with its own indicator. */
[data-testid="stSidebarNavLink"] {{
  forced-color-adjust: none !important;
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
  border-radius: 8px !important; font-weight: 600 !important; border: 2px solid var(--mnst-border-strong) !important;
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
/* Streamlit's text inputs put the visible border/background on
   stTextInputRootElement, a wrapper div -- not the raw <input>, which
   renders borderless inside it. The wrapper is the target so the input
   can't show a second, conflicting outline of its own. */
/* Solid, not translucent -- sitting directly over the busy background
   image, the usual surface translucency washed out placeholder and typed
   text. */
[data-testid="stTextInputRootElement"], [data-testid="stTextArea"] > div {{
  border-radius: 10px !important; border: 2px solid var(--mnst-border-strong) !important; background: var(--mnst-surface-solid) !important;
}}
[data-testid="stTextInputRootElement"]:has(input:focus), [data-testid="stTextArea"] > div:has(textarea:focus) {{
  border-color: var(--mnst-blue) !important; box-shadow: 0 0 0 3px var(--mnst-blue-50) !important;
}}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stChatInput"] textarea {{
  border: none !important; outline: none !important; box-shadow: none !important; background: transparent !important;
  color: var(--mnst-ink-900) !important;
}}
[data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder, [data-testid="stChatInput"] textarea::placeholder {{
  color: var(--mnst-ink-500) !important; opacity: 1 !important;
}}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {{ color: var(--mnst-ink-700) !important; font-weight: 500 !important; }}

/* stNumberInputContainer uses Streamlit's own hardcoded light-theme
   background, not a CSS variable, so it needs explicit overriding. Same
   solid-panel treatment as the text input above for a consistent style. */
[data-testid="stNumberInputContainer"] {{
  background: var(--mnst-surface-solid) !important; border: 2px solid var(--mnst-border-strong) !important; border-radius: 10px !important;
}}
[data-testid="stNumberInputContainer"]:has(input:focus) {{
  border-color: var(--mnst-blue) !important; box-shadow: 0 0 0 3px var(--mnst-blue-50) !important;
}}
[data-testid="stNumberInputField"] {{ color: var(--mnst-ink-900) !important; }}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{ color: var(--mnst-ink-700) !important; }}
[data-testid="stNumberInputStepDown"]:hover, [data-testid="stNumberInputStepUp"]:hover {{ color: #fff !important; }}

/* ---------------- File uploader ---------------- */
[data-testid="stFileUploaderDropzone"] {{
  background: var(--mnst-surface) !important; border: 1px dashed var(--mnst-border-strong) !important; border-radius: 10px !important;
}}
[data-testid="stFileUploaderDropzone"] * {{ color: var(--mnst-ink-500) !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] span {{ color: var(--mnst-ink-700) !important; }}

/* ---------------- Select / Multiselect ---------------- */
/* Same wrapper-div issue as text inputs above, one level deeper: stSelectbox's
   <input> sits inside an unnamed, Streamlit-generated wrapper div that
   carries the real background; stMultiSelect's equivalent wrapper sits one
   level above its tags container. :has() reaches both without depending on
   unstable class names. */
[data-testid="stSelectbox"] div:has(> input) {{
  background: var(--mnst-surface) !important; border: 2px solid var(--mnst-border-strong) !important; border-radius: 10px !important;
}}
[data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]) {{
  background: var(--mnst-surface) !important; border: 2px solid var(--mnst-border-strong) !important; border-radius: 10px !important;
}}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input {{
  color: var(--mnst-ink-900) !important;
}}
/* The option list renders as a portal near the end of <body>, not nested
   in the widget -- [role="listbox"] itself is transparent, its parent
   carries the real background. */
div:has(> [role="listbox"]) {{
  background: var(--mnst-surface-solid) !important; border: 2px solid var(--mnst-border-strong) !important;
}}
[role="option"] {{ color: var(--mnst-ink-900) !important; }}
[role="option"]:hover, [role="option"][aria-selected="true"] {{ background: var(--mnst-blue-50) !important; }}

/* ---------------- Tabs (e.g. Admin page) ---------------- */
/* The active-tab underline is a separate element, .react-aria-SelectionIndicator,
   only rendered on the selected [data-testid="stTab"]. */
[data-testid="stTab"] .react-aria-SelectionIndicator {{ background-color: var(--mnst-blue) !important; }}
[data-testid="stTab"][aria-selected="true"] {{ color: var(--mnst-blue) !important; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: var(--mnst-blue) !important; font-weight: 600 !important; }}

/* ---------------- Form controls (multiselect / radio / checkbox) ---------------- */
/* Streamlit's stock accent for selected controls is red (#FF4B4B, its
   default primaryColor) since no native theme was configured. */
[data-baseweb="tag"] {{ background-color: var(--mnst-blue) !important; }}
/* A selected stRadioOption is label > div > div > div(ring) > div(fill), with
   the text label as a sibling div at the same depth as the ring -- "div div
   div" alone would also match the text container and paint a box behind it,
   so :not() excludes stMarkdownContainer. The fill dot one level deeper
   doesn't have this ambiguity since the text branch ends at <p>. */
[data-testid="stRadioOption"][data-selected="true"] div div div:not([data-testid="stMarkdownContainer"]) {{
  background: var(--mnst-blue) !important; border-color: var(--mnst-blue) !important;
}}
[data-testid="stRadioOption"][data-selected="true"] div div div div {{ background: var(--mnst-blue) !important; border-color: var(--mnst-blue) !important; }}
[data-testid="stCheckbox"] svg {{ fill: var(--mnst-blue) !important; }}
[data-testid="stCheckbox"] input:checked ~ div {{ background: var(--mnst-blue) !important; border-color: var(--mnst-blue) !important; }}

/* ---------------- Cards (bordered containers) ---------------- */
/* Translucent + blurred, not solid -- a glassy panel over the background
   image rather than a flat card that would hide it. Soft light glow shadow,
   since a dark shadow is invisible against a dark background. */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 12px !important; border: 2px solid var(--mnst-border) !important; background: var(--mnst-surface) !important;
  backdrop-filter: blur(14px) !important; -webkit-backdrop-filter: blur(14px) !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.2), 0 8px 24px rgba(0,0,0,0.28) !important;
}}

/* ---------------- Expander (Sources) ---------------- */
[data-testid="stExpander"] {{
  border: 2px solid var(--mnst-border) !important; border-radius: 10px !important; background: var(--mnst-surface) !important;
  backdrop-filter: blur(14px) !important; -webkit-backdrop-filter: blur(14px) !important;
}}
[data-testid="stExpander"] summary {{ font-weight: 600 !important; color: var(--mnst-ink-500) !important; font-size: 13px !important; }}

/* ---------------- Spinner (cold-start / warm-up messages) ---------------- */
/* Unstyled, this renders as Streamlit's default white pill with muted gray
   text -- illegible against this app's dark background image. Matches the
   card/expander treatment above instead of standing out as an unstyled
   leftover. */
[data-testid="stSpinner"] {{
  background: var(--mnst-surface-solid) !important; border: 2px solid var(--mnst-border) !important;
  border-radius: 10px !important; padding: 10px 14px !important;
}}
[data-testid="stSpinner"] p {{ color: var(--mnst-ink-900) !important; font-weight: 500 !important; }}

/* ---------------- Badges (confidence / risk pills) ---------------- */
.mnst-badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 2px; }}
.mnst-badge {{
  font-family: "Ubuntu", sans-serif; font-size: 11.5px; font-weight: 700; letter-spacing: 0.02em;
  padding: 4px 11px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;
}}
.mnst-badge-led {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
"""


def inject_theme() -> None:
    """Injects the enterprise theme CSS."""
    css = _CSS.format(
        u400=_b64("ubuntu-400.ttf"), u500=_b64("ubuntu-500.ttf"), u700=_b64("ubuntu-700.ttf"),
        bg=background_url(),
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
