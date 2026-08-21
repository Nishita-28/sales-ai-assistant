"""Admin-editable field definitions for the Customer Requirement Capture
page's generic questions (Customer Name, Industry, Certifications, etc.)
-- not the per-product nomenclature dropdowns, which stay fully driven by
the Product Index/approved catalogues since those are grounded facts,
not a form-design preference.

JSON-backed at data/requirements_fields.json, seeded with the form's
original hardcoded field set (DEFAULT_FIELDS) so first load reproduces
today's form.

Every default field is marked "core": True -- tied to a real, typed
column in requirements_store.py that can't just be dropped, so an admin
can rename/relabel/edit it but not delete it outright (see
_merge_core_fields). A custom field an admin adds has none of that
restriction, and its answers are stored generically.

Customer Name/Company are NOT in this list -- they come from the deal
picker now, not a form field an admin can relabel or hide.

Backed by Postgres (Neon) when app.db.is_postgres_enabled().
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from app import db

FIELDS_PATH = Path("data/requirements_fields.json")

FIELD_TYPES = ["text", "textarea", "number", "select", "multiselect", "radio"]

# The exact fields/options/labels the form has always had -- changing this
# list does not change already-saved data/requirements_fields.json, only
# what a fresh install (or a field an admin fully deletes) falls back to.
DEFAULT_FIELDS: list[dict[str, Any]] = [
    {"key": "industries", "label": "Industry / Use Case", "type": "multiselect", "options": [
        "Oil & Gas", "Power Generation & Transmission", "Petrochemicals", "Fertilizers", "Pharmaceuticals",
        "Chemicals", "Nuclear Research", "Steel", "Battery Rooms", "Automobiles (ICE / FCEV)",
        "Electrolysers (PEM / Alkaline / SOEC)", "Fuel Cells (PEM / PAFC / SOFC)",
        "Hydrogen Supply Chain (Pipelines / Storage / Tankers)", "Strategic Sectors (Space / Defence / R&D)",
        "New Age Applications (Ships / Aircraft / Backup Power)", "Other",
    ], "required": False, "visible": True, "core": True},
    {"key": "application_details", "label": "Application Details (specific scenario, if useful)", "type": "textarea", "options": [], "required": False, "visible": True, "core": True},
    {"key": "install_type", "label": "Fixed or Portable", "type": "radio", "options": ["Fixed", "Portable"], "required": False, "visible": True, "core": True},
    {"key": "num_detectors", "label": "Number of detectors", "type": "number", "options": [], "required": False, "visible": True, "core": True},
    {"key": "comm_other", "label": "Other protocol / configuration requirement (if not listed above)", "type": "text", "options": [], "required": False, "visible": True, "core": True},
    {"key": "certifications", "label": "Certifications Required", "type": "multiselect", "options": ["PESO", "Ex d (Flameproof)", "Ex ia (Intrinsically Safe)", "IP65", "IP66", "IP67", "IP68", "ATEX", "IECEx", "CE", "Other"], "required": False, "visible": True, "core": True},
    {"key": "cert_other", "label": "Other certification (if any)", "type": "text", "options": [], "required": False, "visible": True, "core": True},
    {"key": "temp_min", "label": "Operating Temperature Min (°C)", "type": "number", "options": [], "required": False, "visible": True, "core": True},
    {"key": "temp_max", "label": "Operating Temperature Max (°C)", "type": "number", "options": [], "required": False, "visible": True, "core": True},
    {"key": "target_gas", "label": "Target / Background Gas (e.g. Hydrogen, with Nitrogen background)", "type": "text", "options": [], "required": False, "visible": True, "core": True},
    {"key": "sensing_range", "label": "Sensing Range Needed (e.g. 0-2000 ppm)", "type": "text", "options": [], "required": False, "visible": True, "core": True},
    {"key": "accuracy", "label": "Accuracy Required", "type": "text", "options": [], "required": False, "visible": True, "core": True},
    {"key": "environmental", "label": "Environmental Conditions", "type": "multiselect", "options": ["Dust", "Water / Washdown", "Outdoor Exposure", "High Humidity", "Other"], "required": False, "visible": True, "core": True},
    {"key": "probe_length", "label": "Probe / Cable Length (if applicable)", "type": "text", "options": [], "required": False, "visible": True, "core": True},
    {"key": "additional", "label": "Additional Requirements", "type": "textarea", "options": [], "required": False, "visible": True, "core": True},
]

_DEFAULT_BY_KEY = {f["key"]: f for f in DEFAULT_FIELDS}


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "field"


def load_fields() -> list[dict[str, Any]]:
    """The current field list, seeding storage with DEFAULT_FIELDS on
    first call so it always reflects real, current state rather than an
    implicit fallback the admin page can't see."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        row = db.fetch_one("SELECT fields FROM requirements_fields WHERE id = 1")
        if row is None:
            save_fields(DEFAULT_FIELDS)
            return [dict(f) for f in DEFAULT_FIELDS]
        return row["fields"]

    if not FIELDS_PATH.exists():
        save_fields(DEFAULT_FIELDS)
        return [dict(f) for f in DEFAULT_FIELDS]
    return json.loads(FIELDS_PATH.read_text(encoding="utf-8"))


def save_fields(fields: list[dict[str, Any]]) -> None:
    if db.is_postgres_enabled():
        db.ensure_schema()
        db.execute(
            "INSERT INTO requirements_fields (id, fields) VALUES (1, %s) "
            "ON CONFLICT (id) DO UPDATE SET fields = EXCLUDED.fields",
            (Jsonb(fields),),
        )
        return

    FIELDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIELDS_PATH.write_text(json.dumps(fields, indent=2, ensure_ascii=False), encoding="utf-8")


def field_map() -> dict[str, dict[str, Any]]:
    """{key: field_dict}, with every DEFAULT_FIELDS core key guaranteed
    present (falling back to its built-in default) even if it's missing
    from the saved file entirely -- e.g. an admin deleted its row in the
    editor. Keeps requirements_page.py from crashing on a lookup for a
    field that underpins real validation (customer_name, company) or
    real logic (certifications' unsupported-certification warning)."""
    fields = {f["key"]: f for f in load_fields()}
    for key, default in _DEFAULT_BY_KEY.items():
        fields.setdefault(key, default)
    return fields


def unique_key(label: str, existing_key: str, taken: set[str]) -> str:
    """A stable key for a field row: keeps its existing key if it already
    has one (so renaming a label doesn't orphan previously-saved answers),
    otherwise slugifies the label, disambiguating against keys already
    assigned earlier in the same save."""
    key = existing_key.strip() if existing_key and existing_key.strip() else _slugify(label)
    base, i = key, 2
    while key in taken:
        key = f"{base}_{i}"
        i += 1
    return key


def merge_core_fields(edited: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconciles an admin's edited field list against what's already
    saved: any core field whose row the admin deleted from the editor is
    kept, forced to visible=False, rather than actually removed -- a core
    field is tied to a real typed database column (and, for a few, to
    validation/warning logic) that can't safely disappear. A non-core
    field with a deleted row is dropped for real."""
    existing = {f["key"]: f for f in load_fields()}
    edited_keys = {f["key"] for f in edited}
    result = list(edited)
    for key, f in existing.items():
        if f.get("core") and key not in edited_keys:
            result.append({**f, "visible": False})
    return result
