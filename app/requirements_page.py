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
import re

import pandas as pd
import streamlit as st

from app.concentration import ConcentrationRange
from app.product_index import NomenclatureSegment, ProductIndexEntry, load_product_index
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
                    expanded = {
                        f"{n:0{width}d}": f"code {n:0{width}d} of {low:0{width}d}-{high:0{width}d} (not individually documented)"
                        for n in range(low, high + 1)
                    }
                    choosable_segments.append((seg_code, label, expanded))
                else:
                    fixed_segments.append((label, only_value))

        if fixed_segments:
            with st.expander(f"{product_choice}'s fixed (non-selectable) specifications"):
                for label, value in fixed_segments:
                    st.caption(f"**{label}**: {value}")

        if not choosable_segments and not fixed_segments:
            st.caption(f"No documented ordering options are available for {product_choice}.")

    with st.form("requirement_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        customer_name = col1.text_input("Customer Name")
        company = col2.text_input("Company")

        industries = st.multiselect("Industry / Use Case", INDUSTRY_USE_CASES + ["Other"])
        application_details = st.text_area(
            "Application Details (specific scenario, if useful)", height=80
        )

        col1, col2 = st.columns(2)
        install_type = col1.radio("Fixed or Portable", ["Fixed", "Portable"], horizontal=True)
        num_detectors = col2.number_input("Number of detectors", min_value=1, step=1, value=1)

        # One dropdown per real ordering choice the selected product has
        # -- e.g. "Select Output Signal" for VISION H2 LD, with its exact
        # documented codes ("V - Analogue Voltage Output", ...), always
        # including a not-known fallback so a rep isn't forced to guess
        # what the customer hasn't decided yet.
        segment_selections: dict[str, str] = {}
        segment_summaries: list[str] = []
        for seg_code, label, display in choosable_segments:
            option_labels = [f"{code} – {val}" for code, val in display.items()]
            label_to_code = dict(zip(option_labels, display.keys()))
            chosen = st.selectbox(f"Select {label}", [NOT_SURE] + option_labels, key=f"seg-{seg_code}")
            if chosen != NOT_SURE:
                code = label_to_code[chosen]
                segment_selections[seg_code] = code
                segment_summaries.append(f"{label}: {display[code]} ({code})")

        comm_other = st.text_input("Other protocol / configuration requirement (if not listed above)")

        certifications = st.multiselect("Certifications Required", ["PESO", "ATEX", "IECEx", "CE", "Other"])
        if UNSUPPORTED_CERTIFICATIONS & set(certifications):
            st.warning(
                "ATEX, IECEx, and CE are not currently documented or certified for any approved "
                "product. Recording this need is fine -- it must be escalated, not promised."
            )
        cert_other = st.text_input("Other certification (if any)")

        col1, col2 = st.columns(2)
        temp_min = col1.number_input("Operating Temperature Min (°C)", value=-20.0, step=1.0)
        temp_max = col2.number_input("Operating Temperature Max (°C)", value=65.0, step=1.0)

        target_gas = st.text_input("Target / Background Gas (e.g. Hydrogen, with Nitrogen background)")
        sensing_range = st.text_input("Sensing Range Needed (e.g. 0-2000 ppm)")
        accuracy = st.text_input("Accuracy Required")

        environmental = st.multiselect(
            "Environmental Conditions", ["Dust", "Water / Washdown", "Outdoor Exposure", "High Humidity"]
        )
        probe_length = st.text_input("Probe / Cable Length (if applicable)")

        additional = st.text_area("Additional Requirements", height=100)

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
                environmental=", ".join(environmental),
                probe_length=probe_length,
                suggested_code=suggested_code,
                additional_requirements=additional,
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

    df = pd.DataFrame([dict(r) for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    st.download_button(
        "Download as Excel",
        data=buffer.getvalue(),
        file_name="customer_requirements.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
