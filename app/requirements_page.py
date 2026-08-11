"""Customer requirement capture page: lets a sales rep record a customer's
technical needs during or after a call, for a clean handoff to production.

Grounded in the real Product Index (see app/product_index.py) rather than
a generic, hand-guessed list of options -- checked directly against the
approved catalogues: "Ethernet" was previously offered as a communication
protocol though no approved product documents it, and ATEX/IECEx/CE were
offered as certification options though none of them are documented or
certified for any current product (see data/restricted_claims.yaml).
"""
from __future__ import annotations

import io
import json
import re

import pandas as pd
import streamlit as st

from app.concentration import ConcentrationRange, parse_concentration_range
from app.product_index import NomenclatureSegment, ProductIndexEntry, load_product_index
from app.requirements_fields import field_map
from app.requirements_store import list_requirements, record_requirement

NOT_SURE = "Other"

# Per data/restricted_claims.yaml: not documented or certified for any
# current product. Still offered below so a real customer need can be
# recorded -- but flagged, since it can't be promised.
UNSUPPORTED_CERTIFICATIONS = {"ATEX", "IECEx", "CE"}

# Grounded in data/approved_docs/Industry Use Case Guide_MNST Product
# Categories.md -- the real set of industries/applications MNST's own
# Fixed, Portable, and Analyzer ranges are documented as serving, not a
# guessed list. Kept as the section headers from that guide (not every
# individual example under them) so the picker stays a quick multiselect;
# the free-text field alongside it covers anything more specific.
INDUSTRY_USE_CASES = [
    "Oil & Gas",
    "Power Generation & Transmission",
    "Petrochemicals",
    "Fertilizers",
    "Pharmaceuticals",
    "Chemicals",
    "Nuclear Research",
    "Steel",
    "Battery Rooms",
    "Automobiles (ICE / FCEV)",
    "Electrolysers (PEM / Alkaline / SOEC)",
    "Fuel Cells (PEM / PAFC / SOFC)",
    "Hydrogen Supply Chain (Pipelines / Storage / Tankers)",
    "Strategic Sectors (Space / Defence / R&D)",
    "New Age Applications (Ships / Aircraft / Backup Power)",
]


def _display_name(name: str) -> str:
    """Customer-facing product name with the trailing ordering-code
    placeholder ("XX"/"XX*") stripped -- e.g. "VISION H2 LD" instead of
    "VISION H2 LD XX". The registry keeps "XX" because it's the real,
    literal token in the product's own nomenclature (needed for exact
    alias/lookup matching); a form talking to a rep has no reason to
    show a raw code-template artifact."""
    return re.sub(r"\s+XX\*?$", "", name).strip()


def _selectable_segments(nomenclature) -> list[tuple[str, str, dict[str, str]]]:
    """Every nomenclature segment with real decoded values, in template
    order -- (segment_code, label, {value_code: display text}) tuples.
    Feeds one dropdown per real ordering choice the product actually has
    (e.g. "Select Output Signal" for VISION H2 LD, "Select Range" and
    "Select Background" for FIXaHY 4220MA), read directly from the
    registry, never guessed."""
    if not isinstance(nomenclature, dict):
        return []
    result = []
    for seg_code, segment in nomenclature.items():
        if isinstance(segment, NomenclatureSegment) and segment.values:
            display = {
                val_code: (str(v) if isinstance(v, ConcentrationRange) else str(v))
                for val_code, v in segment.values.items()
            }
            result.append((seg_code, segment.label, display))
    return result


_NUMERIC_CODE_RANGE_RE = re.compile(r"^(\d+)\s*(?:to|-|–)\s*(\d+)$", re.IGNORECASE)


def _numeric_code_range(value: str) -> tuple[int, int, int] | None:
    """(low, high, digit_width) if a segment's single documented value is
    a bare numeric code range like "01 to 50" -- a genuinely selectable
    field (the customer picks one industry/application code from that
    range at order time) that just doesn't have each individual code's
    meaning spelled out, unlike a real single fixed spec such as
    "04: LM6 Die Cast". digit_width preserves zero-padding (e.g. "01" ->
    width 2) so a chosen code matches the catalogue's own numbering.
    Proven necessary by testing: FIXaHY-4220MA's "Industry: 01 to 50"
    was being treated as a single fixed spec (like Enclosure) and hidden
    from the rep entirely, when it's actually 50 real, selectable
    options -- just ones the document doesn't individually name."""
    m = _NUMERIC_CODE_RANGE_RE.match(value.strip())
    if not m:
        return None
    low_s, high_s = m.group(1), m.group(2)
    return int(low_s), int(high_s), max(len(low_s), len(high_s))


def _suggested_code(product: ProductIndexEntry, segment_selections: dict[str, str]) -> str:
    """Reconstructs a concrete ordering code from the product's own
    nomenclature template, substituting each selectable segment's
    customer-chosen value into its position -- e.g. base "FIXaHY G/P/E
    4220MA RR* NN VV* II" becomes "FIXaHY P 4220MA 02 04 03 II" once
    real choices are made. A segment with only one documented value uses
    it automatically (extracting a leading "04:"-style code from its
    description when present); a genuinely unresolved multi-option
    segment keeps its raw placeholder token (e.g. "II") rather than
    guessing, so the output is always an honest code shape, never
    silently wrong. Only called once every real choosable segment has an
    answer (see render_requirements_page)."""
    if not isinstance(product.nomenclature, dict):
        return ""
    parts = []
    for seg_code, segment in product.nomenclature.items():
        if seg_code in segment_selections:
            parts.append(segment_selections[seg_code])
        elif isinstance(segment, NomenclatureSegment) and len(segment.values) == 1:
            only_code, only_value = next(iter(segment.values.items()))
            if only_code != "*":
                parts.append(only_code)
            else:
                m = re.match(r"^(\w{1,3}):", str(only_value))
                parts.append(m.group(1) if m else seg_code.rstrip("*"))
        else:
            parts.append(seg_code.rstrip("*"))
    return " ".join(parts)


def _format_option(code: str, value: str) -> str:
    """"G" instead of "G - Unknown" -- a segment option the catalogue
    lists as valid but never explains (e.g. FIXaHY-4220MA's Internal Code
    G/P/E) still needs to be selectable, just without implying there's a
    real description being withheld. Uses "code: value" (colon), not an
    en dash -- proven necessary by testing: "01 – 2000 ppm" reads as a
    range from 01 to 2000, when 01 is the option's code and 2000 ppm is
    its Span, two unrelated numbers."""
    return code if value == "Unknown" else f"{code}: {value}"


_WHITESPACE_RE = re.compile(r"\s+")
_SPAN_RE = re.compile(r"Span:\s*([^;]+)")


def _normalize_value_text(value: str) -> str:
    """Collapses a stray embedded newline (e.g. FIXaHY-4220MA's own
    Resolution cell text: "0.002%\\n(20 ppm)") into a single line -- the
    literal newline otherwise breaks up the option text mid-value."""
    return _WHITESPACE_RE.sub(" ", value).strip()


def _short_option_text(value: str, max_len: int = 70) -> str:
    """A compact identifier for a dropdown's closed/list view. A full
    Range description (Start/Span/Resolution/MDL/Accuracy all joined
    together) is easily 100+ characters -- far wider than the dropdown
    itself, so it read as cut off. The full text is always shown
    separately once a real choice is made (see the segment loop below),
    so nothing is actually lost by shortening it here.

    For a Range-shaped value (has a "Span:" figure), reuses
    ConcentrationRange's own "0 to 2,000 ppm (0% to 0.5% v/v)" formatting
    -- the exact same rendering AURIGA's Range Full Scale already gets
    (AURIGA's registry entry stores a real ConcentrationRange; FIXaHY-
    4220MA/PORTaHY's Range tables get parsed into plain text instead, see
    registry_builder._parse_selectable_table) -- so every product's Range
    dropdown reads the same way rather than each looking different
    depending on which extraction path happened to produce its data."""
    normalized = _normalize_value_text(value)
    m = _SPAN_RE.search(normalized)
    if m:
        span_range = parse_concentration_range(m.group(1).strip())
        if span_range is not None:
            return str(span_range)
        # Couldn't parse a concentration out of the Span text (shouldn't
        # happen for a real Range segment) -- fall back to labeling the
        # raw figure rather than showing nothing.
        return f"Span {m.group(1).strip()}"
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len].rstrip() + "..."


def _render_custom_field(f: dict, not_sure: str = NOT_SURE) -> str:
    """Renders one admin-added custom field (see app.requirements_fields)
    generically by its configured type -- unlike the core fields above,
    a custom field has no bespoke merge/validation logic to preserve."""
    key = f"custom-{f['key']}"
    ftype = f.get("type", "text")
    label = f["label"]
    options = f.get("options") or []
    if ftype == "textarea":
        return st.text_area(label, height=80, key=key)
    if ftype == "number":
        return str(st.number_input(label, value=0.0, key=key))
    if ftype == "select":
        chosen = st.selectbox(label, [not_sure] + options, key=key)
        return "" if chosen == not_sure else chosen
    if ftype == "multiselect":
        return ", ".join(st.multiselect(label, options, key=key))
    if ftype == "radio":
        return st.radio(label, options, horizontal=True, key=key) if options else ""
    return st.text_input(label, key=key)


def render_requirements_page() -> None:
    st.title("Customer Requirement Capture")
    st.caption("Capture a customer's technical requirements for a clean handoff to production.")

    try:
        products, _ = load_product_index()
    except (FileNotFoundError, OSError):
        products = []

    # Product, installation area, and hazard zone all drive the rest of
    # the form's options and need to react immediately -- form widgets
    # only update on submit, so these sit outside the form (same reason
    # installation_area/hazard_zone already did).
    display_to_product = {_display_name(p.product_name): p for p in products}
    product_choice = st.selectbox("Product", [NOT_SURE] + list(display_to_product.keys()))
    selected_product: ProductIndexEntry | None = display_to_product.get(product_choice)

    installation_area = st.radio("Installation Area", ["Safe", "Hazardous"], horizontal=True)
    hazard_zone = ""
    if installation_area == "Hazardous":
        hazard_zone = st.radio("Zone", ["Zone 0", "Zone 1", "Zone 2"], horizontal=True)

    # Every real, multi-option nomenclature choice for the selected
    # product becomes its own question below -- e.g. VISION H2 LD's
    # Output Signal, PORTaHY's Range Full Scale, FIXaHY 4220MA's Range/
    # Background/Internal Code, each with the exact documented options.
    # A segment documenting a bare numeric code range (e.g. 4220MA's
    # Industry: "01 to 50") is ALSO a real, selectable choice -- just one
    # whose individual codes aren't each individually named -- expanded
    # into one option per code so it gets its own dropdown too, not
    # hidden as if it were a single fixed spec. Only a segment with a
    # genuinely single, non-range value (e.g. Enclosure is always
    # "04: LM6 Die Cast") isn't a real decision -- shown as read-only
    # context instead of a pointless one-option dropdown.
    choosable_segments: list[tuple[str, str, dict[str, str]]] = []
    fixed_segments: list[tuple[str, str]] = []
    if selected_product is not None:
        for seg_code, label, display in _selectable_segments(selected_product.nomenclature):
            if len(display) >= 2:
                choosable_segments.append((seg_code, label, display))
            elif display:
                only_value = next(iter(display.values()))
                code_range = _numeric_code_range(only_value)
                if code_range:
                    low, high, width = code_range
                    # "Unknown" (not a real description) so it renders as
                    # a bare code via _format_option, same as a segment
                    # option the catalogue lists but never explains (e.g.
                    # Internal Code G/P/E) -- these codes genuinely have
                    # no individual meaning documented, just like those.
                    expanded = {f"{n:0{width}d}": "Unknown" for n in range(low, high + 1)}
                    choosable_segments.append((seg_code, label, expanded))
                else:
                    fixed_segments.append((label, only_value))

    # The product itself determines Fixed vs Portable -- it's not
    # something a customer configures at order time (there's no ordering
    # code for it, unlike Range/Background/etc.), so once a product with
    # a documented install_type is picked, this is a fact to show, not a
    # question to ask. Falls back to asking only when the product's
    # install_type isn't documented (e.g. FIXaHY Analyzer Series -- see
    # data/product_registry.json) or no product is selected yet.
    product_install_type = (
        selected_product.install_type
        if selected_product is not None and selected_product.install_type in ("Fixed", "Portable")
        else None
    )
    if selected_product is not None:
        if product_install_type:
            fixed_segments.append(("Install Type", product_install_type))

        if fixed_segments:
            with st.expander(f"{product_choice}'s fixed (non-selectable) specifications"):
                for label, value in fixed_segments:
                    st.caption(f"**{label}**: {value}")

        if not choosable_segments and not fixed_segments:
            st.caption(f"No documented ordering options are available for {product_choice}.")

    # One dropdown per real ordering choice the selected product has --
    # e.g. "Select Output Signal" for VISION H2 LD, with its exact
    # documented codes ("V - Analogue Voltage Output", ...), always
    # including a not-known fallback so a rep isn't forced to guess what
    # the customer hasn't decided yet. Grounded in the Product Index, not
    # admin-editable here -- see the module docstring. Kept outside the
    # form and right next to the fixed-specs expander above (not buried
    # among the generic questions below) -- proven necessary by testing:
    # a rep looking for FIXaHY 4220MA's "Select Industry" (its ordering
    # code) scrolled past the unrelated generic "Industry / Use Case"
    # field first and assumed the real one was missing.
    segment_selections: dict[str, str] = {}
    segment_summaries: list[str] = []
    if choosable_segments:
        st.caption(f"{product_choice}'s ordering options:")
        for seg_code, label, display in choosable_segments:
            option_labels = [_format_option(code, _short_option_text(val)) for code, val in display.items()]
            label_to_code = dict(zip(option_labels, display.keys()))
            chosen = st.selectbox(f"Select {label}", [NOT_SURE] + option_labels, key=f"seg-{seg_code}")
            if chosen != NOT_SURE:
                code = label_to_code[chosen]
                segment_selections[seg_code] = code
                value = display[code]
                segment_summaries.append(f"{label}: {code}" if value == "Unknown" else f"{label}: {value} ({code})")
                # The dropdown itself only shows a short identifier (e.g.
                # Range's Span) once a full description would run past
                # the widget's width -- the complete text always shows
                # here so nothing is actually hidden from the rep.
                if value != "Unknown" and _short_option_text(value) != _normalize_value_text(value):
                    st.caption(f"**{label} {code}**: {_normalize_value_text(value)}")

    # Every generic (non-nomenclature) question below is admin-editable
    # from the "Customer Requirements" tab in Admin (label, options,
    # required, visible) -- see app.requirements_fields. Customer Name and
    # Company stay always-shown regardless of their "visible" setting: the
    # save validation below requires both, and requirements_store.py's
    # database columns are NOT NULL for them.
    fcfg = field_map()

    def _visible(key: str) -> bool:
        return fcfg.get(key, {}).get("visible", True)

    def _label(key: str, fallback: str) -> str:
        return fcfg.get(key, {}).get("label") or fallback

    def _options(key: str, fallback: list[str]) -> list[str]:
        return fcfg.get(key, {}).get("options") or fallback

    custom_fields = [f for f in fcfg.values() if not f.get("core") and f.get("visible", True)]

    # enter_to_submit=False -- proven risky by inspection: this form has
    # 15+ fields and clear_on_submit=True, so pressing Enter out of habit
    # in an early field (Customer Name, say) would submit -- and wipe --
    # everything typed so far, or worse, silently save an incomplete
    # record if Customer Name/Company happen to already be filled. A
    # rep should only submit via the actual button.
    with st.form("requirement_form", clear_on_submit=True, enter_to_submit=False):
        col1, col2 = st.columns(2)
        customer_name = col1.text_input(_label("customer_name", "Customer Name"))
        company = col2.text_input(_label("company", "Company"))

        industries = (
            st.multiselect(_label("industries", "Industry / Use Case"), _options("industries", INDUSTRY_USE_CASES + ["Other"]))
            if _visible("industries") else []
        )
        application_details = (
            st.text_area(_label("application_details", "Application Details (specific scenario, if useful)"), height=80)
            if _visible("application_details") else ""
        )

        # Fixed vs Portable is a fact about the selected product (shown
        # above, in its fixed specifications), not a customer choice --
        # only asked here as a fallback when the product's install_type
        # isn't documented (see product_install_type above) or no product
        # is selected yet.
        if product_install_type:
            install_type = product_install_type
            num_detectors = st.number_input(
                _label("num_detectors", "Number of detectors"), min_value=1, step=1, value=1
            ) if _visible("num_detectors") else 1
        else:
            col1, col2 = st.columns(2)
            install_type = (
                col1.radio(_label("install_type", "Fixed or Portable"), _options("install_type", ["Fixed", "Portable"]), horizontal=True)
                if _visible("install_type") else ""
            )
            num_detectors = (
                col2.number_input(_label("num_detectors", "Number of detectors"), min_value=1, step=1, value=1)
                if _visible("num_detectors") else 1
            )

        comm_other = (
            st.text_input(_label("comm_other", "Other protocol / configuration requirement (if not listed above)"))
            if _visible("comm_other") else ""
        )

        certifications = (
            st.multiselect(_label("certifications", "Certifications Required"), _options("certifications", ["PESO", "ATEX", "IECEx", "CE", "Other"]))
            if _visible("certifications") else []
        )
        if UNSUPPORTED_CERTIFICATIONS & set(certifications):
            st.warning(
                "ATEX, IECEx, and CE are not currently documented or certified for any approved "
                "product. Recording this need is fine -- it must be escalated, not promised."
            )
        cert_other = (
            st.text_input(_label("cert_other", "Other certification (if any)")) if _visible("cert_other") else ""
        )

        col1, col2 = st.columns(2)
        temp_min = (
            col1.number_input(_label("temp_min", "Operating Temperature Min (°C)"), value=-20.0, step=1.0)
            if _visible("temp_min") else -20.0
        )
        temp_max = (
            col2.number_input(_label("temp_max", "Operating Temperature Max (°C)"), value=65.0, step=1.0)
            if _visible("temp_max") else 65.0
        )

        target_gas = (
            st.text_input(_label("target_gas", "Target / Background Gas (e.g. Hydrogen, with Nitrogen background)"))
            if _visible("target_gas") else ""
        )
        sensing_range = (
            st.text_input(_label("sensing_range", "Sensing Range Needed (e.g. 0-2000 ppm)")) if _visible("sensing_range") else ""
        )
        accuracy = st.text_input(_label("accuracy", "Accuracy Required")) if _visible("accuracy") else ""

        environmental_options = _options("environmental", ["Dust", "Water / Washdown", "Outdoor Exposure", "High Humidity"])
        if "Other" not in environmental_options:
            environmental_options = [*environmental_options, "Other"]
        environmental = (
            st.multiselect(_label("environmental", "Environmental Conditions"), environmental_options)
            if _visible("environmental") else []
        )
        environmental_other = (
            st.text_input("Other environmental condition (please specify)")
            if _visible("environmental") and "Other" in environmental else ""
        )

        # Grounded in the approved catalogues, not a guess: only AURIGA and
        # PORTaHY H2 LD (both "Portable") document a probe/cable at all --
        # none of the Fixed products (FIXaHY 4220MA, VISION H2 LD, FIXaHY
        # H2 LD) mention one, so asking it for a Fixed install is a
        # question with no real answer.
        probe_length = (
            st.text_input(_label("probe_length", "Probe / Cable Length (if applicable)"))
            if _visible("probe_length") and install_type == "Portable" else ""
        )

        additional = st.text_area(_label("additional", "Additional Requirements"), height=100) if _visible("additional") else ""

        custom_values: dict[str, str] = {}
        if custom_fields:
            st.divider()
            st.caption("Additional fields (added in Admin)")
            for f in custom_fields:
                custom_values[f["key"]] = _render_custom_field(f)

        submitted = st.form_submit_button("Save Requirement", type="primary")

    if submitted:
        if not customer_name or not company:
            st.error("Customer Name and Company are required.")
        else:
            application = ", ".join(industries)
            if application_details:
                application = f"{application} -- {application_details}" if application else application_details

            protocols = list(segment_summaries)
            if comm_other:
                protocols.append(comm_other)
            certs = [c for c in certifications if c != "Other"]
            if cert_other:
                certs.append(cert_other)
            env_conditions = [e for e in environmental if e != "Other"]
            if environmental_other:
                env_conditions.append(environmental_other)

            # A concrete, orderable product code, only once every real
            # choosable segment has an answer -- deterministic, built
            # from the registry's own segment order, never guessed at
            # for an unresolved position.
            suggested_code = ""
            if selected_product is not None and choosable_segments and len(segment_selections) == len(choosable_segments):
                suggested_code = _suggested_code(selected_product, segment_selections)

            record_requirement(
                customer_name=customer_name,
                company=company,
                application=application,
                product_family=selected_product.product_name if selected_product else "",
                install_type=install_type,
                num_detectors=int(num_detectors),
                comm_protocols=", ".join(protocols),
                certifications=", ".join(certs),
                installation_area=installation_area,
                hazard_zone=hazard_zone,
                temp_min=temp_min,
                temp_max=temp_max,
                target_gas=target_gas,
                sensing_range=sensing_range,
                accuracy=accuracy,
                environmental=", ".join(env_conditions),
                probe_length=probe_length,
                suggested_code=suggested_code,
                additional_requirements=additional,
                extra_fields=json.dumps({k: v for k, v in custom_values.items() if v}, ensure_ascii=False),
            )
            msg = f"Saved requirement for {customer_name} ({company})."
            if suggested_code:
                msg += f" Suggested product code: **{suggested_code}**."
            st.success(msg)

    st.divider()
    st.subheader("Recent Requirements")
    rows = list_requirements()
    if not rows:
        st.caption("No requirements captured yet.")
        return

    # Flatten each row's extra_fields JSON ({custom_key: answer}) into its
    # own columns so an admin-added custom field shows up like any other
    # column, in the table and in the Excel export -- not buried as raw
    # JSON text.
    flattened = []
    for r in rows:
        row = dict(r)
        raw_extra = row.pop("extra_fields", "") or ""
        try:
            extra = json.loads(raw_extra) if raw_extra else {}
        except json.JSONDecodeError:
            extra = {}
        flattened.append({**row, **extra})

    df = pd.DataFrame(flattened)
    st.dataframe(df, use_container_width=True, hide_index=True)

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    st.download_button(
        "Download as Excel",
        data=buffer.getvalue(),
        file_name="customer_requirements.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
