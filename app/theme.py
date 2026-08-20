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
    """Sets Streamlit's own native theme options at runtime (no-op under
    classic). Needed on top of inject_theme()'s CSS: canvas-rendered
    widgets -- st.data_editor / st.dataframe (glide-data-grid) draw their
    cells, header row, and selection outline onto a <canvas>, reading
    colors from Streamlit's native theme config, not the DOM -- CSS can't
    reach inside a canvas at all. Confirmed live: with only primaryColor
    set, every data_editor/dataframe on the site rendered as a plain
    white grid with black text, untouched by any of the dark-theme CSS
    and glaringly inconsistent with everything around it. backgroundColor/
    secondaryBackgroundColor/textColor are what the grid actually reads
    for its own cell colors. Must run before st.set_page_config()."""
    if not is_enterprise_theme():
        return
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
    """A real, plain URL -- not a base64 data URI. Confirmed by testing:
    embedding this same ~1.7MB image as base64 inside the injected CSS
    (re-sent in full on every single Streamlit rerun, not cached) made
    cold starts take minutes instead of seconds. Served instead from
    app/static/ (requires server.enableStaticServing, set in
    .streamlit/config.toml), which the browser fetches once and caches
    like any ordinary asset. Relative, not a leading-slash absolute path,
    so it still resolves correctly if the app is ever deployed under a
    URL subpath."""
    return "app/static/sidebar-bg.png"


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
/* font-display: swap -- without it, the default (auto, which behaves
   like "block" in Chromium) holds all text INVISIBLE while the browser
   decodes this base64-embedded font, on every single page. Since this
   whole stylesheet re-injects on every Streamlit rerun (every nav
   click), that's a fresh invisible-text flash each time: the layout
   swaps instantly but the text itself doesn't paint until the font
   finishes decoding a moment later -- exactly the "page shifts, text
   shows up after" lag reported live. swap paints text immediately in
   a fallback font, then swaps to Ubuntu once it's ready, instead of
   hiding it in the meantime. */
@font-face {{ font-family: "Ubuntu"; font-weight: 400; font-style: normal; font-display: swap; src: url(data:font/ttf;base64,{u400}) format("truetype"); }}
@font-face {{ font-family: "Ubuntu"; font-weight: 500; font-style: normal; font-display: swap; src: url(data:font/ttf;base64,{u500}) format("truetype"); }}
@font-face {{ font-family: "Ubuntu"; font-weight: 700; font-style: normal; font-display: swap; src: url(data:font/ttf;base64,{u700}) format("truetype"); }}

:root {{
  /* Dark theme, everywhere -- the whole app sits on the fixed background
     image (see .stApp below), so every token here is calibrated for
     light text/translucent panels over a dark photo, not the old
     light-page/dark-sidebar split. --mnst-page is the fallback color
     while the image loads (and shows at the image's own dark edges),
     not a visible surface of its own anymore. */
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
  /* Solid, not translucent -- a rounded, alpha-blended rectangle sitting
     over the busy background image can show a faint anti-aliased rim at
     its own edge, which reads as an unexplained "box" even though no
     separate border/outline element is actually there (confirmed by
     testing: outline, border, box-shadow, and filter all compute as
     fully inert on the link, every descendant, every ancestor, and both
     pseudo-elements). An opaque color has no edge to blend, removing the
     ambiguity at the source instead of chasing a property that isn't
     the real cause. */
  --mnst-rail-active-bg: #1f4e8a;
}}

html, body, .stApp {{ font-family: "Ubuntu", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif !important; }}
/* Longhand properties, not the `background` shorthand -- confirmed by
   testing that the shorthand silently drops the whole declaration (computed
   backgroundImage stayed "none") once the embedded base64 image pushed the
   single property value past roughly 2MB, even though the CSS text itself
   was well-formed. Longhand background-image alone doesn't hit the same
   wall. */
/* A flat dark scrim (same color as --mnst-page) is layered over the
   image itself, not just placed behind translucent cards -- the image's
   bright wave/skyline detail was competing with the actual UI content
   for attention rather than reading as background atmosphere. Muting the
   image directly, once, fixes that everywhere instead of needing every
   card to individually out-contrast it. */
.stApp {{
  background-color: var(--mnst-page) !important;
  background-image: linear-gradient(rgba(5, 14, 31, 0.72), rgba(5, 14, 31, 0.72)), url("{bg}") !important;
  background-position: center center !important;
  background-size: cover !important;
  background-repeat: no-repeat !important;
  background-attachment: fixed !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
/* color needs !important here -- confirmed by testing: Streamlit's own
   base stylesheet sets a specific dark text color (rgb(49,51,63)) that
   otherwise wins the cascade over this rule. Invisible under the old
   light theme (both colors were dark-ish, indistinguishable), but a real
   near-black-on-dark-background bug once the theme flipped to light text. */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ font-weight: 700 !important; letter-spacing: -0.01em; color: var(--mnst-ink-900) !important; }}
/* Split from a single shared rule: plain paragraphs are the actual
   content this tool exists to deliver -- an AI-generated answer,
   sitting inside a card, is the whole point of the page, not secondary
   detail -- so it gets the brighter ink tone. True captions (source
   labels, timestamps, "Sources" expander text) keep the dim tone,
   since those really are secondary. */
p, .stMarkdown p {{ color: var(--mnst-ink-700) !important; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--mnst-ink-500) !important; }}

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
/* The real root cause of the persistent "box", found only by forcibly
   overriding background/background-image/text-shadow via an injected
   test rule and watching it disappear, then confirmed by reading
   getComputedStyle('background') (not just backgroundColor) directly on
   the <a> itself: Streamlit's OWN base stylesheet paints the current-page
   sidebar link with rgba(151, 166, 195, 0.25) -- a light grey-blue,
   clearly tuned for Streamlit's default light theme. Every earlier
   attempt at this (background-color only, outline, border, box-shadow,
   filter, backdrop-filter, removing the fill entirely) targeted the
   wrong property or masked it inconsistently; this light translucent
   fill was always the thing rendering underneath, and against a dark
   background it reads as a pale, unexplained rectangle. `background`
   (the shorthand Streamlit's own rule uses), not `background-color`,
   is required to actually win the cascade here. */
/* Background is painted on the <a> ONLY, never on descendants: painting
   the same translucent color on both a parent and a smaller child box
   (icon span, text span, <p>) stacks two semi-transparent layers where
   they overlap, compositing to a visibly denser rectangle tightly
   around the label text -- confirmed via getComputedStyle on every
   descendant of a hovered link, each independently carrying its own
   copy of the identical background. Descendants get color only, plus
   an explicit transparent background so nothing else can repaint them. */
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
/* The browser's own default focus ring (a 3px solid white box drawn
   around the whole link, confirmed via computed style) is unrelated to
   the border-left above -- it fires because clicking a nav item leaves
   it focused, and it's far more visible against this dark theme than it
   ever was against the light one. The left border already marks "current
   page" on its own; the extra focus box is redundant, not intentional. */
[data-testid="stSidebarNavLink"]:focus, [data-testid="stSidebarNavLink"]:focus-visible {{
  outline: none !important; box-shadow: none !important;
}}
/* A visible box still surrounds the active nav item even with the above
   in place and even on a completely fresh page load with nothing focused
   (document.activeElement is <body>) -- ruled out by testing: outline,
   border, box-shadow, and filter all compute as genuinely inert (style:
   none / 0px) on the link itself, every descendant, every ancestor, and
   both ::before/::after pseudo-elements. Nothing in this stylesheet is
   drawing it. The remaining explanation is the browser's own forced-
   colors/high-contrast accessibility mode, which deliberately overrides
   author styles (including outline:none) to draw its own indicator --
   forced-color-adjust is the one property meant to opt an element out of
   that override, so it's the next thing to try. */
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
/* Streamlit's newer (react-aria) text inputs put the VISIBLE border/
   background on stTextInputRootElement, a wrapper div -- not on the raw
   <input> itself, which renders borderless inside it. Styling only the
   <input> left Streamlit's own default red-accent focus ring on the
   wrapper showing through underneath our blue one (confirmed via DOM
   inspection -- this was a real, visible bug, not a guess). So the
   wrapper is the primary target now; the inner input is made borderless
   so it can't show a second, conflicting outline of its own. */
/* Solid (not translucent) background here on purpose: this sits directly
   over the busy, bright wave/skyline background image, and the surface
   token's usual translucency let that image show through enough to wash
   out both the placeholder and typed text -- confirmed by the input being
   nearly unreadable in a screenshot taken over a bright stretch of the
   image. A fully opaque panel reads clearly regardless of what's behind it. */
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

/* stNumberInputContainer carries Streamlit's own hardcoded light-theme
   background (rgb(240,242,246)) directly, not a CSS variable -- unlike
   the text-input wrapper above, so it was never touched by any of the
   dark-theme rules and stayed a bright, un-themed box next to everything
   else on the page (confirmed via computed style: background and both
   step-button icon colors were all still Streamlit's light-theme
   defaults). Same solid-panel treatment as the text input above, so the
   two read as one consistent input style. */
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
/* Same wrapper-div issue as text inputs above, but one level deeper and
   with no data-testid on the actual wrapper -- confirmed via DOM
   inspection: stSelectbox's <input> sits inside an unnamed, Streamlit-
   generated div (an unstable st-emotion-cache-* class, useless to target
   directly) that carries the real visible background; stMultiSelect's
   equivalent wrapper sits one level above its own stMultiSelectTagsContainer.
   :has() reaches both without depending on those unstable class names. */
[data-testid="stSelectbox"] div:has(> input) {{
  background: var(--mnst-surface) !important; border: 2px solid var(--mnst-border-strong) !important; border-radius: 10px !important;
}}
[data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]) {{
  background: var(--mnst-surface) !important; border: 2px solid var(--mnst-border-strong) !important; border-radius: 10px !important;
}}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input {{
  color: var(--mnst-ink-900) !important;
}}
/* The option list is a portal, rendered near the end of <body> rather than
   nested inside the widget -- confirmed via DOM inspection: [role="listbox"]
   itself is transparent, its immediate parent carries the real (previously
   solid white) background. */
div:has(> [role="listbox"]) {{
  background: var(--mnst-surface-solid) !important; border: 2px solid var(--mnst-border-strong) !important;
}}
[role="option"] {{ color: var(--mnst-ink-900) !important; }}
[role="option"]:hover, [role="option"][aria-selected="true"] {{ background: var(--mnst-blue-50) !important; }}

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
/* Translucent + blurred (not solid) -- a glassy panel over the
   background image, matching the reference mockups, instead of a flat
   card that would otherwise hide the image completely everywhere
   content appears. The old shadow (a near-black rgba) was calibrated
   for a white card on a light page and is invisible against a dark
   background -- replaced with a soft light glow instead. */
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
    css = _CSS.format(
        u400=_b64("ubuntu-400.ttf"), u500=_b64("ubuntu-500.ttf"), u700=_b64("ubuntu-700.ttf"),
        bg=background_url(),
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
