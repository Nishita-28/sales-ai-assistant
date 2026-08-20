"""Admin edits to a specific product's selectable fields -- adding a
brand-new one, editing a catalogue-extracted one's options, or hiding
one entirely (whether it came from the catalogue or was admin-added).

Deliberately kept OUT of data/product_registry.json: that file is fully
regenerated from the approved catalogues on every reindex/rebuild (see
registry_builder.rebuild_product_registry) -- anything written directly
into it would be silently wiped out the next time a document is added,
removed, or the index is rebuilt. This is a separate, persistent JSON
store, following the same pattern as app.requirements_fields, merged
in at read time by requirements_page.py (see effective_additional_params)
rather than baked into the registry itself.

Keyed by product_name (the registry's own canonical name, e.g. "VISION
H2 LD XX") -- not by source_catalogue or family, since that's exactly
what requirements_page.py already looks products up by.

Shape on disk: {product_name: {"fields": {label: {"type": str, "options":
{option: description}}}, "hidden": [label, ...]}}. "fields" covers both
a brand-new admin-added label AND an edit-override of a catalogue label
(same mechanism -- editing just means "fields" now has an entry for a
label the catalogue also produces, and that entry wins). "hidden" covers
deleting a field regardless of where it came from; a hidden label is
suppressed even if "fields" also has an entry for it (delete wins over a
stale edit). "type" is one of app.requirements_fields.FIELD_TYPES --
defaults to "select" for a catalogue-extracted label, since a real
"Selectable <X>" table is always a single choice.

Backed by Postgres (Neon) when app.db.is_postgres_enabled() -- see
app.deals_store's module docstring for why."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import json

from psycopg.types.json import Jsonb

from app import db

OVERRIDES_PATH = Path("data/product_field_overrides.json")

# get_fields()/get_hidden()/effective_additional_params() each call
# load_overrides() independently, and a single product's field render
# (admin_page._effective_product_fields) calls several of those in a
# row -- against Postgres, that's 4+ separate round trips for what's
# really one small table. A short TTL collapses those into one real query
# per render while still picking up a save within a couple of reruns --
# save_overrides() also clears it directly, so an admin's own edit is
# never waiting on the TTL to expire.
_CACHE_TTL_SECONDS = 3
_cache: tuple[float, dict[str, dict[str, Any]]] | None = None


def load_overrides() -> dict[str, dict[str, Any]]:
    global _cache
    if db.is_postgres_enabled():
        now = time.time()
        if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
            return _cache[1]
        db.ensure_schema()
        rows = db.fetch_all("SELECT product_name, data FROM product_field_overrides")
        result = {row["product_name"]: row["data"] for row in rows}
        _cache = (now, result)
        return result

    if not OVERRIDES_PATH.exists():
        return {}
    return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))


def save_overrides(overrides: dict[str, dict[str, Any]]) -> None:
    global _cache
    if db.is_postgres_enabled():
        db.ensure_schema()
        with db.get_connection() as conn:
            conn.execute("DELETE FROM product_field_overrides")
            for product_name, data in overrides.items():
                conn.execute(
                    "INSERT INTO product_field_overrides (product_name, data) VALUES (%s, %s)",
                    (product_name, Jsonb(data)),
                )
        _cache = None
        return

    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")


def _entry(overrides: dict[str, dict[str, Any]], product_name: str) -> dict[str, Any]:
    return overrides.setdefault(product_name, {"fields": {}, "hidden": []})


def get_fields(product_name: str) -> dict[str, dict[str, Any]]:
    """Admin-added-or-edited {label: {"type": str, "options": {option:
    description}}} for this product -- does NOT include catalogue-only
    fields that were never touched here."""
    return load_overrides().get(product_name, {}).get("fields", {})


def get_hidden(product_name: str) -> set[str]:
    return set(load_overrides().get(product_name, {}).get("hidden", []))


def set_field(product_name: str, label: str, options: dict[str, str], field_type: str = "select") -> None:
    """Adds a brand-new field, or edits an existing one (catalogue- or
    admin-sourced) -- both are the same operation: this label's
    effective type/options from now on are exactly `field_type`/
    `options`. Un-hides the label if it had been previously deleted,
    since editing it is a clear signal it should show again."""
    overrides = load_overrides()
    entry = _entry(overrides, product_name)
    entry["fields"][label] = {"type": field_type, "options": options}
    if label in entry["hidden"]:
        entry["hidden"].remove(label)
    save_overrides(overrides)


def remove_field(product_name: str, label: str) -> None:
    """Deletes this label entirely -- for an admin-added field this is
    real removal; for a catalogue-sourced field (no "fields" entry to
    begin with, or one that's just an edit-override) this instead
    records it as hidden, since the catalogue will keep re-producing it
    on every registry rebuild otherwise."""
    overrides = load_overrides()
    entry = _entry(overrides, product_name)
    entry["fields"].pop(label, None)
    if label not in entry["hidden"]:
        entry["hidden"].append(label)
    save_overrides(overrides)


def unhide_field(product_name: str, label: str) -> None:
    overrides = load_overrides()
    entry = _entry(overrides, product_name)
    if label in entry["hidden"]:
        entry["hidden"].remove(label)
        save_overrides(overrides)


def effective_additional_params(product: Any) -> dict[str, dict[str, Any]]:
    """The full, final {label: {"type": str, "options": {option:
    description}}} this product should show as freeform selectable specs
    (i.e. everything EXCEPT real ordering-code segments -- see
    requirements_page.py's own choosable_segments, which handles those
    separately and consults is_nomenclature_label_active/label_override
    below for the same hide/edit rules). Combines, in order: the
    catalogue's own additional_selectable_parameters (registry_builder's
    auto-extraction -- always type "select", the only shape a real
    "Selectable <X>" table produces), a catalogue ordering-code label
    that's been edited here (edit "promotes" it out of the order-code
    system -- see is_nomenclature_label_active), and any brand-new
    admin-added label -- each step skipping anything hidden, and letting
    a "fields" override replace the catalogue's own content for that
    label."""
    fields = get_fields(product.product_name)
    hidden = get_hidden(product.product_name)
    nomenclature_labels = (
        {v["label"] for v in product.nomenclature.values() if isinstance(v, dict)}
        if isinstance(product.nomenclature, dict) else set()
    )

    result: dict[str, dict[str, Any]] = {}
    for label, opts in product.additional_selectable_parameters.items():
        if label in hidden:
            continue
        result[label] = fields.get(label, {"type": "select", "options": opts})
    for label in nomenclature_labels:
        if label in hidden or label not in fields or label in result:
            continue
        result[label] = fields[label]
    for label, field in fields.items():
        if label in hidden or label in result:
            continue
        result[label] = field
    return result


def is_nomenclature_label_hidden_or_edited(product_name: str, label: str) -> bool:
    """True if a real ordering-code segment's label has been hidden or
    edited here -- requirements_page.py's choosable_segments loop must
    exclude it in either case: hidden means don't show it at all,
    edited means it's now rendered (with the override's content) via
    effective_additional_params instead, not as an ordering-code
    dropdown -- an admin-supplied option set can't be trusted to still
    contain the product's real order-code letters, so it stops
    contributing to the suggested product code once edited."""
    hidden = get_hidden(product_name)
    fields = get_fields(product_name)
    return label in hidden or label in fields
