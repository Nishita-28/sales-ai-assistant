"""Product Index -- the data model for Path 2 (deterministic product
identity), and a loader for it. This is Point 1 of the architecture only:
what a Product Index entry is, and how to read one from disk.

Deliberately NOT included here (separate, later work):
- Extraction -- populating data/product_registry.json from the catalogues
  (see the registry-builder script; this module only reads its output).
- The dispatcher -- deciding which questions should consult this index.
- Wiring into retriever.py/response_generator.py -- the live app doesn't
  use this yet; it's a standalone data model and loader until that's done.

Descriptive/explanatory product content (how something works, why it's
suitable) stays in the catalogues themselves, answered by semantic RAG --
this index only holds deterministic facts explicitly stated in each
catalogue's own title, "Technology:" field, and ordering nomenclature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from app.concentration import ConcentrationRange, HydrogenConcentration

REGISTRY_PATH = Path("data/product_registry.json")


@dataclass(frozen=True)
class NomenclatureSegment:
    """One selectable position in a product's ordering code -- a label
    (e.g. "Output Signal", "Range Full Scale") plus its decoded values.
    Values are plain strings for enum-like segments (a communication
    protocol choice is not a concentration) or ConcentrationRange for
    genuinely concentration-valued segments (Range / Range Full Scale) --
    see app.concentration.CONCENTRATION_RANGE_LABELS, applied at
    extraction time, not here."""

    label: str
    values: dict[str, Union[str, ConcentrationRange]]


# A nomenclature segment can be a fixed label (e.g. "FIXaHY" -> "Product
# Series Name", nothing to select) or a NomenclatureSegment (selectable,
# with decoded values). The whole nomenclature can also be the literal
# string "Unknown" -- the pattern wasn't found for that catalogue (e.g.
# FIXaHY-4220MA's different table structure, or the Analyzer's prose-based
# labels) -- never guessed at.
NomenclatureValue = Union[str, NomenclatureSegment]
Nomenclature = Union[dict[str, NomenclatureValue], str]


@dataclass(frozen=True)
class ProductIndexEntry:
    """One product's structured identity -- everything Path 2 needs to
    answer "what is this" and "which catalogue documents it" without
    retrieval. Fields that couldn't be confidently extracted are the
    literal string "Unknown", never a guess (see the registry-builder
    script's vocabulary-matching rules).

    additional_selectable_parameters holds real, documented selectable
    specs (e.g. VISION H2 LD's Range, Background Gas, Compatible
    Interfaces, Connector Option) that have NO position in the product's
    own ordering-code suffix at all -- {label: {option name: description}}
    -- kept separate from `nomenclature` (which only ever represents
    literal positions in the order code) rather than folding them in,
    since a caller reconstructing a real order code from `nomenclature`
    must never accidentally include one of these in it."""

    product_name: str
    family: str
    product_type: str
    technology: str
    install_type: str
    target_gas: str
    source_catalogue: str
    aliases: tuple[str, ...]
    nomenclature: Nomenclature
    additional_selectable_parameters: dict[str, dict[str, str]]


def _parse_concentration(d: dict[str, float]) -> HydrogenConcentration:
    return HydrogenConcentration(ppm=d["ppm"], percent_vv=d["percent_vv"], percent_lel=d["percent_lel"])


def _parse_range(d: dict[str, Any]) -> ConcentrationRange:
    return ConcentrationRange(start=_parse_concentration(d["start"]), end=_parse_concentration(d["end"]))


def _is_range_dict(value: Any) -> bool:
    return isinstance(value, dict) and "start" in value and "end" in value


def _parse_nomenclature(raw: Union[dict, str]) -> Nomenclature:
    if isinstance(raw, str):  # "Unknown"
        return raw
    parsed: dict[str, NomenclatureValue] = {}
    for code, value in raw.items():
        if isinstance(value, str):
            parsed[code] = value
            continue
        values: dict[str, Union[str, ConcentrationRange]] = {
            k: (_parse_range(v) if _is_range_dict(v) else v) for k, v in value.get("values", {}).items()
        }
        parsed[code] = NomenclatureSegment(label=value["label"], values=values)
    return parsed


def load_product_index(path: Path = REGISTRY_PATH) -> tuple[list[ProductIndexEntry], dict[str, str]]:
    """Loads data/product_registry.json into typed entries. Returns
    (products, technology_aliases) -- the latter a small, human-supplied
    map like {"SSEC": "Solid State Electrochemical"}, since that
    abbreviation never appears in any catalogue and can't be extracted
    (see app.concentration's module docstring for why). Raises if the
    file is missing or malformed; callers decide how to handle that --
    this doesn't guess at a fallback."""
    data = json.loads(path.read_text(encoding="utf-8"))
    products = [
        ProductIndexEntry(
            product_name=p["product_name"],
            family=p["family"],
            product_type=p["product_type"],
            technology=p["technology"],
            install_type=p["install_type"],
            target_gas=p["target_gas"],
            source_catalogue=p["source_catalogue"],
            aliases=tuple(p["aliases"]),
            nomenclature=_parse_nomenclature(p["nomenclature"]),
            # .get, not [] -- a registry built before this field existed
            # won't have it at all.
            additional_selectable_parameters=p.get("additional_selectable_parameters", {}),
        )
        for p in data["products"]
    ]
    return products, data.get("technology_aliases", {})


def by_name(products: list[ProductIndexEntry], name: str) -> Optional[ProductIndexEntry]:
    """Exact, case-insensitive match against a product's canonical name or
    any of its aliases. A basic index lookup, not a query-routing decision
    -- deciding *when* to call this belongs to the dispatcher, separate,
    later work."""
    name_lower = name.strip().lower()
    for p in products:
        if p.product_name.lower() == name_lower or name_lower in {a.lower() for a in p.aliases}:
            return p
    return None
