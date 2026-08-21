"""Hydrogen concentration as a domain value, not a primitive string or
number. 100% LEL = 4% H2 v/v = 40,000 ppm is a fixed physical constant
for hydrogen, identical across every product catalogue -- one conversion
ratio serves every product; this is not per-product data.

Needs to be a real conversion, not left to the LLM: the Product Registry
mixes units within one product's own table (e.g. "2,000 PPM H2 v/v"
alongside "4% H2 v/v"), unusable for a range comparison without a shared
representation; and a model can confidently miscompute the arithmetic
(e.g. 15,000 ppm as 6% instead of 1.5%).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Fixed for hydrogen specifically -- not configurable per product.
PERCENT_VV_PER_100_LEL = 4.0
PPM_PER_PERCENT_VV = 10_000.0


@dataclass(frozen=True)
class HydrogenConcentration:
    """A single point concentration -- always available in all three units
    regardless of which one it was constructed from. A concentration is a
    point, not a span; see ConcentrationRange for a product's full-scale
    interval, which is composed of two of these, not a separate type."""

    ppm: float
    percent_vv: float
    percent_lel: float

    @classmethod
    def from_ppm(cls, ppm: float) -> "HydrogenConcentration":
        percent_vv = ppm / PPM_PER_PERCENT_VV
        return cls(ppm=ppm, percent_vv=percent_vv, percent_lel=percent_vv * (100 / PERCENT_VV_PER_100_LEL))

    @classmethod
    def from_percent_vv(cls, percent_vv: float) -> "HydrogenConcentration":
        return cls(
            ppm=percent_vv * PPM_PER_PERCENT_VV,
            percent_vv=percent_vv,
            percent_lel=percent_vv * (100 / PERCENT_VV_PER_100_LEL),
        )

    @classmethod
    def from_percent_lel(cls, percent_lel: float) -> "HydrogenConcentration":
        percent_vv = percent_lel * (PERCENT_VV_PER_100_LEL / 100)
        return cls(ppm=percent_vv * PPM_PER_PERCENT_VV, percent_vv=percent_vv, percent_lel=percent_lel)

    def to_dict(self) -> dict[str, float]:
        return {"ppm": round(self.ppm, 4), "percent_vv": round(self.percent_vv, 6), "percent_lel": round(self.percent_lel, 4)}

    def __str__(self) -> str:
        return f"{self.ppm:,.0f} ppm = {self.percent_vv:g}% v/v = {self.percent_lel:g}% LEL"


@dataclass(frozen=True)
class ConcentrationRange:
    """A product's documented full-scale span -- an interval between two
    points. Deliberately composed from HydrogenConcentration rather than
    given its own separate ppm/percent fields, since a range's boundaries
    are themselves concentrations, not a distinct kind of value."""

    start: HydrogenConcentration
    end: HydrogenConcentration

    def contains(self, value: HydrogenConcentration) -> bool:
        return self.start.ppm <= value.ppm <= self.end.ppm

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    def __str__(self) -> str:
        return f"{self.start.ppm:,.0f} to {self.end.ppm:,.0f} ppm ({self.start.percent_vv:g}% to {self.end.percent_vv:g}% v/v)"


_PPM_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*ppm", re.IGNORECASE)
_PERCENT_RE = re.compile(r"([\d.]+)\s*%")


def parse_concentration(text: str) -> Optional[HydrogenConcentration]:
    """Parses a raw registry value string ("2,000 PPM H2 v/v", "4% H2 v/v")
    into a HydrogenConcentration. Returns None if no recognizable ppm or
    percent value is present -- never guesses at a value that isn't there."""
    m = _PPM_RE.search(text)
    if m:
        return HydrogenConcentration.from_ppm(float(m.group(1).replace(",", "")))
    m = _PERCENT_RE.search(text)
    if m:
        return HydrogenConcentration.from_percent_vv(float(m.group(1)))
    return None


def parse_concentration_range(text: str) -> Optional[ConcentrationRange]:
    """A registry "Range Full Scale" value documents a single full-scale
    endpoint (e.g. "2,000 PPM H2 v/v") -- 0 is the implicit start, per
    every product's own "Selectable range" table, which states "Start:
    0 ppm/0%" explicitly for each range option."""
    end = parse_concentration(text)
    if end is None:
        return None
    return ConcentrationRange(start=HydrogenConcentration.from_ppm(0.0), end=end)


# Segment labels that are genuinely concentration ranges, per the Product
# Registry's own extracted labels -- NOT every decoded segment is one (e.g.
# "Output Signal" decodes to a communication-protocol choice, not a
# concentration, and must never be parsed as one).
CONCENTRATION_RANGE_LABELS = {"range", "range full scale"}


def _normalize_unit(raw: str) -> Optional[str]:
    raw = raw.lower().replace(" ", "")
    if "lel" in raw:
        return "percent_lel"
    if raw == "ppm":
        return "ppm"
    if raw in ("%", "percent", "%v/v", "%vv", "percentv/v"):
        return "percent_vv"
    return None


_VALUE_UNIT_RE = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(ppm|%\s*lel|lel\s*%|percent\s*lel|%|percent)\b",
    re.IGNORECASE,
)
_TARGET_UNIT_RE = re.compile(
    r"\bin\s+(ppm|%\s*lel|lel\s*%|percent\s*lel|%|percent(?:\s*v\s*/\s*v)?)\b",
    re.IGNORECASE,
)


def detect_conversion_request(query: str) -> Optional[tuple[HydrogenConcentration, str]]:
    """If the query states a hydrogen concentration in one unit and asks
    for it in a different unit ("what is 15000 ppm in percent LEL"),
    returns (parsed value, target unit key). Returns None otherwise --
    deliberately narrow (an explicit "X in Y" conversion question), not a
    general-purpose concentration parser over arbitrary text."""
    value_match = _VALUE_UNIT_RE.search(query)
    target_match = _TARGET_UNIT_RE.search(query)
    if not value_match or not target_match:
        return None

    source_unit = _normalize_unit(value_match.group(2))
    target_unit = _normalize_unit(target_match.group(1))
    if source_unit is None or target_unit is None or source_unit == target_unit:
        return None

    raw_value = float(value_match.group(1).replace(",", ""))
    if source_unit == "ppm":
        concentration = HydrogenConcentration.from_ppm(raw_value)
    elif source_unit == "percent_vv":
        concentration = HydrogenConcentration.from_percent_vv(raw_value)
    else:
        concentration = HydrogenConcentration.from_percent_lel(raw_value)

    return concentration, target_unit


_TARGET_UNIT_LABELS = {"ppm": "ppm", "percent_vv": "% v/v", "percent_lel": "% LEL"}


def conversion_grounding_note(concentration: HydrogenConcentration, target_unit: str) -> str:
    """The exact fact to inject into the LLM's context so it states the
    pre-computed number instead of recalculating it -- the arithmetic
    happens here in code, not in the model."""
    return (
        f"Exact hydrogen concentration conversion (compute this yourself, do not recalculate): "
        f"{concentration.ppm:,.0f} ppm = {concentration.percent_vv:g}% H2 v/v = {concentration.percent_lel:g}% LEL. "
        f"The value requested in {_TARGET_UNIT_LABELS.get(target_unit, target_unit)} is "
        f"{getattr(concentration, target_unit):g}{'%' if target_unit != 'ppm' else ' ppm'}. "
        f"Use these exact numbers verbatim; do not perform your own unit conversion."
    )
