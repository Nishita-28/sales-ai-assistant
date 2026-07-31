"""Customer requirement capture page: lets a sales rep record a customer's
technical needs during or after a call, for a clean handoff to production.
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app.requirements_store import list_requirements, record_requirement


def render_requirements_page() -> None:
    st.title("Customer Requirement Capture")
    st.caption("Capture a customer's technical requirements for a clean handoff to production.")

    # Installation area and the Zone follow-up need to react immediately --
    # form widgets only update on submit, so these sit outside the form.
    installation_area = st.radio("Installation Area", ["Safe", "Hazardous"], horizontal=True)
    hazard_zone = ""
    if installation_area == "Hazardous":
        hazard_zone = st.radio("Zone", ["Zone 0", "Zone 1", "Zone 2"], horizontal=True)

    with st.form("requirement_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        customer_name = col1.text_input("Customer Name")
        company = col2.text_input("Company")

        application = st.text_area("Application (e.g. battery room, electrolyzer)", height=80)

        col1, col2 = st.columns(2)
        install_type = col1.radio("Fixed or Portable", ["Fixed", "Portable"], horizontal=True)
        num_detectors = col2.number_input("Number of detectors", min_value=1, step=1, value=1)

        comm_protocols = st.multiselect(
            "Communication Protocol", ["4-20 mA", "RS485", "CAN", "Ethernet", "Other"]
        )
        comm_other = st.text_input("Other protocol (if any)")

        certifications = st.multiselect(
            "Certifications Required", ["PESO", "ATEX", "IECEx", "CE", "Other"]
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
            protocols = [p for p in comm_protocols if p != "Other"]
            if comm_other:
                protocols.append(comm_other)
            certs = [c for c in certifications if c != "Other"]
            if cert_other:
                certs.append(cert_other)

            record_requirement(
                customer_name=customer_name,
                company=company,
                application=application,
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
                additional_requirements=additional,
            )
            st.success(f"Saved requirement for {customer_name} ({company}).")

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
