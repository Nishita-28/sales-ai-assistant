"""Builds data/product_registry.json from the approved catalogues
automatically. Wired into the same reindex triggers the vector index uses
(rag_pipeline.py --reindex, admin_page.py's rebuild/add/remove actions),
so the registry and the index never drift out of sync.

Extraction rule throughout: never guess. A field that can't be matched
against a small, bounded vocabulary is reported as "Unknown", not
inferred."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.concentration import CONCENTRATION_RANGE_LABELS, parse_concentration_range

REGISTRY_PATH = Path("data/product_registry.json")

# The only hand-authored input: small, bounded category vocabulary --
# grows when a genuinely new product category/technology is introduced,
# not with every new catalogue. Not per-product data.
PRODUCT_TYPE_VOCAB = ["Leak Detector", "Analyzer"]
TECHNOLOGY_VOCAB = ["Solid State Electrochemical", "CMOS MEMS", "MEMS"]
INSTALL_TYPE_VOCAB = ["Portable", "Fixed"]
NOMENCLATURE_HEADINGS = ["product ordering nomenclature", "product ordering information"]

# Human-supplied, not extractable -- these abbreviations never appear in
# any catalogue (confirmed: "SSEC" appears only in an unrelated historical
# sales record, never in a product document). Small, bounded, rarely
# changes -- unlike per-product facts.
TECHNOLOGY_ALIASES = {
    "SSEC": "Solid State Electrochemical",
    "MEMS": "MEMS",
}


def _iter_block_items(parent):
    for child in parent.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _match_vocab(text: str, vocab: list[str]) -> str:
    text_lower = text.lower()
    for term in sorted(vocab, key=len, reverse=True):
        if term.lower() in text_lower:
            return term
    return "Unknown"


def _extract_technology(full_text: str) -> str:
    m = re.search(r"Technology:\s*(.+)", full_text)
    if m:
        result = _match_vocab(m.group(1)[:200], TECHNOLOGY_VOCAB)
        if result != "Unknown":
            return result
    return _match_vocab(full_text[:4000], TECHNOLOGY_VOCAB)


def _is_selectable_code(code: str) -> bool:
    """True if a nomenclature code token is a customer-selectable position
    rather than the product's own fixed brand+model identity. Marked by a
    trailing "*", a slash-separated option list ("G/P/E"), or a short
    repeated-letter placeholder ("RR", "NN", "XX"). "FIXaHY" and "4220MA"
    match none of these -- always present, never a choice."""
    stripped = code.rstrip("*")
    if code.endswith("*"):
        return True
    if "/" in stripped:
        return True
    if stripped.isalpha() and 2 <= len(stripped) <= 3 and len(set(stripped.upper())) == 1:
        return True
    return False


def _normalize_range_code(code: str) -> str | None:
    """A nomenclature range code (e.g. "2K", "5K", "4%") into a form
    comparable against a table's "Span" row (e.g. "2000 ppm", "4%") --
    "2K" -> "2000", "4%" -> "4%" unchanged. None for a code that isn't
    range-shaped at all (e.g. an Output Signal letter)."""
    code = code.rstrip("*").strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*K$", code, re.IGNORECASE)
    if m:
        return str(int(float(m.group(1)) * 1000))
    if re.match(r"^\d+(?:\.\d+)?%$", code):
        return code
    return None


def _normalize_span_cell(text: str) -> str | None:
    """The matching normalization for a table's Span row cell -- "2000
    ppm" -> "2000", "4%" -> "4%", "5 vol%" -> "5%" (an optional unit word
    between the number and "%" stripped, since AURIGA's own table writes
    it that way while every other product just writes "4%") -- so it can
    be compared directly against _normalize_range_code's output."""
    text = text.strip()
    m = re.match(r"^([\d,]+)\s*ppm$", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:\w+\s*)?%$", text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}%"
    return None


def _parse_selectable_table(table: Table, codes: list[str] | None = None) -> dict[str, str]:
    """Parses a "Selectable <X>" table into {code: description}. Two ways
    to figure out which column is which nomenclature code: (1) match
    against the table's own "Span" row (code "2K" matches Span cell
    "2000 ppm"; "4%" matches "4%" directly) -- there's no single header
    convention across documents, but codes are derived from the Span
    value, so this works either way; (2) fall back to a parenthesized
    code in the header row (the 4220MA style) when there's no Span row.

    Further rows (Start/Span/Resolution/...) combine into a "Row Label:
    value" summary per matched code; a single-row table just uses its own
    column text as the value."""
    rows = [[c.text.strip() for c in row.cells] for row in table.rows]
    if not rows or len(rows[0]) < 2:
        return {}

    row_labels = [r[0] if r else "" for r in rows]
    span_idx = next((i for i, l in enumerate(row_labels) if l.strip().lower() == "span"), None)

    col_codes: dict[int, str] = {}
    if codes and span_idx is not None:
        normalized_codes = {c: n for c in codes if (n := _normalize_range_code(c)) is not None}
        for col_idx, cell in enumerate(rows[span_idx][1:], start=1):
            normalized_cell = _normalize_span_cell(cell)
            if normalized_cell is None:
                continue
            for code, norm in normalized_codes.items():
                if norm == normalized_cell:
                    col_codes[col_idx] = code
                    break

    names: dict[int, str] = {}
    if not col_codes:
        # 1-3 word chars only -- a real code ("01", "V", "DM") is always
        # short, unlike a parenthesized descriptive word (e.g.
        # "(Industrial)") that an unbounded \w+ would wrongly capture.
        for col_idx, cell in enumerate(rows[0][1:], start=1):
            m = re.search(r"\((\w{1,3})\)", cell)
            if m:
                col_codes[col_idx] = m.group(1)
                names[col_idx] = re.sub(r"\s*\(\w{1,3}\)\s*", " ", cell).strip()

    if not col_codes:
        return {}

    data_rows = rows[1:]
    # Keep everything through the LAST row that varies across the matched
    # code columns; drop the tail entirely -- some documents merge their
    # whole spec sheet into the Range table, so a trailing row identical
    # across every range (e.g. "Detection Certainty: 100%") shouldn't be
    # glued onto every code's description. A plain "keep only if it
    # varies" filter is too aggressive: a field like MDL can be identical
    # across ranges while still being a real per-range fact.
    if len(col_codes) >= 2:
        varies = [
            len({row[i] for i in col_codes if i < len(row) and row[i]}) > 1
            for row in data_rows
        ]
        if any(varies):
            last_varying = max(i for i, v in enumerate(varies) if v)
            data_rows = data_rows[:last_varying + 1]
        else:
            data_rows = []

    values: dict[str, str] = {}
    for col_idx, code in col_codes.items():
        if data_rows:
            parts = [
                f"{row[0]}: {row[col_idx]}"
                for row in data_rows
                if col_idx < len(row) and row[col_idx] and row[0]
            ]
            values[code] = "; ".join(parts) if parts else names.get(col_idx, "")
        else:
            values[code] = names.get(col_idx, "")
    return {k: v for k, v in values.items() if v}


def _parse_freeform_selectable_table(table: Table) -> dict[str, str]:
    """Parses a "Selectable <X>" table into {option name: description}
    when there's no short code to key by -- unlike _parse_selectable_table
    (which resolves a real ordering-code letter), this is for a table
    documenting a customer choice with no corresponding order-code
    position. The column's own header text is the only identifier these
    tables give, so that's the key. Some products' tables have neither a
    Span row nor a parenthesized code, so _parse_selectable_table alone
    would return {} and silently drop those parameters."""
    rows = [[c.text.strip() for c in row.cells] for row in table.rows]
    if not rows or len(rows[0]) < 2:
        return {}
    col_names = {i: cell for i, cell in enumerate(rows[0][1:], start=1) if cell}
    if not col_names:
        return {}

    data_rows = rows[1:]
    values: dict[str, str] = {}
    for col_idx, name in col_names.items():
        if data_rows:
            parts = [
                f"{row[0]}: {row[col_idx]}"
                for row in data_rows
                if col_idx < len(row) and row[col_idx] and row[0]
            ]
            value = "; ".join(parts) if parts else "Unknown"
        else:
            # Single-row table -- the header row IS the data, so the
            # option name itself is the complete answer.
            value = "Unknown"
        # Two columns can share the same header text (e.g. VISION's
        # Compatible Interfaces has two columns both labeled "0.5-3.5V or
        # RS485" with different data underneath) -- disambiguate with the
        # column position rather than letting the second overwrite the first.
        key = name if name not in values else f"{name} ({col_idx})"
        values[key] = value
    return values


def _extract_additional_selectable_parameters(
    doc: Document, resolved_labels: list[str]
) -> dict[str, dict[str, str]]:
    """Every "Selectable <X>" table not already captured as a real
    ordering-code segment -- e.g. VISION H2 LD's ordering code has one
    selectable position (Output Signal), but the document separately
    documents Range, Background Gas, and other real selectable specs with
    no code position at all. Without this they'd be silently dropped,
    since the nomenclature path only looks at a table when a matching
    code segment led it there. resolved_labels are the nomenclature
    labels that already consumed their own table, so they aren't
    re-added here as a duplicate."""
    # Identity by the underlying XML element (table._tbl), not the
    # python-docx Table wrapper -- doc.tables constructs a fresh wrapper
    # on every access, so id(table) would never match across calls.
    claimed_tables = {
        id(t._tbl) for label in resolved_labels if (t := _find_selectable_table(doc, label)) is not None
    }
    result: dict[str, dict[str, str]] = {}
    for table in doc.tables:
        if id(table._tbl) in claimed_tables:
            continue
        if not table.rows or not table.rows[0].cells:
            continue
        header_cell = table.rows[0].cells[0].text.strip()
        if not header_cell.lower().startswith("selectable"):
            continue
        label = re.sub(r"(?i)^selectable\s+", "", header_cell.split("\n")[0]).strip()
        if not label or label in result:
            continue
        values = _parse_freeform_selectable_table(table)
        if values:
            result[label] = values
    return result


def _find_selectable_table(doc: Document, label: str) -> Table | None:
    """Finds a "Selectable <X>" table elsewhere in the document matching a
    nomenclature segment's label (e.g. "Range" -> "Selectable range").
    Some products have real, detailed per-code tables sitting elsewhere in
    the document, far richer than the crude placeholder text inline in
    the nomenclature table itself."""
    # Trailing "s" stripped before comparing: some products' table headers
    # are plural ("Selectable Ranges") while others are singular
    # ("Selectable Range"), so an exact-word intersection would miss one
    # or the other and silently fall back to the bare ConcentrationRange
    # path instead of the richer table data.
    def _singularize(words: set[str]) -> set[str]:
        return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}

    label_words = _singularize(set(re.findall(r"[a-z]+", label.lower())))
    for table in doc.tables:
        if not table.rows or not table.rows[0].cells:
            continue
        header_text = table.rows[0].cells[0].text.lower()
        if "selectable" not in header_text:
            continue
        if _singularize(set(re.findall(r"[a-z]+", header_text))) & label_words:
            return table
    return None


def _find_fixed_value(doc: Document, label: str) -> str | None:
    """A "<Label>: <value>" sentence anywhere in the document body -- the
    fixed-spec counterpart to _find_selectable_table's elsewhere-in-the-
    document lookup. Only matches a paragraph that STARTS with the label
    (not any sentence merely containing it), so it can't accidentally
    grab an unrelated mention."""
    pattern = re.compile(rf"^{re.escape(label)}\s*:\s*(.+)$", re.IGNORECASE)
    for p in doc.paragraphs:
        m = pattern.match(p.text.strip())
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def _extract_nomenclature(doc: Document) -> dict[str, Any]:
    """Anchors on the "Product Ordering Nomenclature"/"...Information"
    heading (consistent across every catalogue checked), then takes
    whatever tab-separated line comes next as the code template and the
    line after that as labels -- by position, not exact wording, so a
    document phrasing the label row differently ("Series Name" vs
    "Product Series Name") still matches."""
    items = list(_iter_block_items(doc.element.body))
    heading_idx = None
    for i, item in enumerate(items):
        if isinstance(item, Paragraph) and any(h in item.text.lower() for h in NOMENCLATURE_HEADINGS):
            heading_idx = i
            break
    if heading_idx is None:
        return {"aliases": [], "segments": {}, "table_resolved_labels": []}

    template_line, label_line, template_table, label_line_idx, value_row = None, None, None, None, None
    for j in range(heading_idx + 1, min(heading_idx + 10, len(items))):
        item = items[j]
        if isinstance(item, Paragraph) and "\t" in item.text and item.text.strip():
            if template_line is None:
                template_line = item.text.strip()
            elif label_line is None:
                label_line = item.text.strip()
                label_line_idx = j
            elif value_row is None:
                # A third tab-separated line right after the label row,
                # giving each selectable segment's value/range inline --
                # a different decode shape than the "V - meaning" bullet
                # list some documents use instead of, or in addition to.
                value_row = item.text.strip()
                break
        elif isinstance(item, Table) and template_table is None:
            template_table = [c.text.strip() for c in item.rows[0].cells]

    # Bullet decode: the "V - Analogue Voltage Output" lines right after
    # the label row, when present.
    value_decode: dict[str, str] = {}
    if label_line_idx is not None:
        for j in range(label_line_idx + 1, min(label_line_idx + 8, len(items))):
            item = items[j]
            if not isinstance(item, Paragraph):
                break
            text = item.text.strip()
            m = re.match(r"^([A-Za-z0-9%]{1,4})\s*[-–]\s*(.+)$", text)
            if m:
                value_decode[m.group(1)] = m.group(2).strip()
            elif text and value_decode:
                break

    segments: dict[str, Any] = {}
    aliases: list[str] = []
    table_resolved_labels: list[str] = []
    codes = None
    if template_line and label_line:
        codes = [c.strip() for c in template_line.split("\t") if c.strip()]
        labels = [c.strip() for c in label_line.split("\t") if c.strip()]
        if len(codes) == len(labels):
            segments = dict(zip(codes, labels))
    elif template_table:
        codes = template_table

    if codes:
        base = " ".join(c.rstrip("*") for c in codes[:-1]) if len(codes) > 1 else codes[0]
        last = codes[-1].rstrip("*")
        aliases.append(codes[0].rstrip("*"))  # bare brand root, e.g. "PORTaHY" alone
        aliases.append(base)
        aliases.append(f"{base} {last}".strip())
        # Fully-coded aliases: base + each real selectable value, e.g.
        # "FIXaHY H2 LD V", "FIXaHY H2 LD C". Matching against these must
        # be done case-insensitively at lookup time, not by storing every
        # case variant here.
        for code_value in value_decode:
            aliases.append(f"{base} {code_value}".strip())

        # Decode every customer-selectable segment, not just the last --
        # e.g. FIXaHY-4220MA has several selectable segments. Fixed
        # brand+model identity codes are excluded by _is_selectable_code.
        selectable = [c for c in codes if _is_selectable_code(c)]

        inline_values: dict[str, str] = {}
        if value_row:
            row_values = [v.strip() for v in value_row.split("\t") if v.strip()]
            # Non-empty fields align with the TRAILING N selectable codes,
            # same convention the codes/labels rows use (a leading gap for
            # fixed identity segments with no value to show).
            if row_values and len(row_values) <= len(selectable):
                inline_values = dict(zip(selectable[-len(row_values):], row_values))

        for code in selectable:
            if code not in segments:
                continue
            label = segments[code]
            values: dict[str, Any] = {}

            table = _find_selectable_table(doc, label)
            table_values = (
                _parse_selectable_table(table, list(value_decode.keys()) or None)
                if table is not None else {}
            )

            if table_values:
                # A dedicated "Selectable <X>" table -- checked first
                # since it's far richer than the bullet list or inline
                # value-row text, which typically decode to only a bare
                # range. Falls through to those paths below when the
                # table can't be reliably matched to real codes.
                values = table_values
            elif code == codes[-1] and value_decode:
                if label.strip().lower() in CONCENTRATION_RANGE_LABELS:
                    for c2, raw in value_decode.items():
                        parsed = parse_concentration_range(raw)
                        values[c2] = parsed.to_dict() if parsed else raw
                else:
                    values = dict(value_decode)
            elif code in inline_values:
                raw = inline_values[code]
                if label.strip().lower() in CONCENTRATION_RANGE_LABELS:
                    parsed = parse_concentration_range(raw)
                    values["*"] = parsed.to_dict() if parsed else raw
                else:
                    values["*"] = raw
            elif "/" in code.rstrip("*"):
                # A slash-separated option list (e.g. "G/P/E") with no
                # decode found anywhere in the document -- checked:
                # FIXaHY-4220MA never explains what G, P, or E mean.
                # Never invent a meaning; still surface that these are
                # the real, valid options, per the "never guess" rule.
                for opt in code.rstrip("*").split("/"):
                    values[opt] = "Unknown"

            if values:
                segments[code] = {"label": label, "values": values}
                # Tracked regardless of which path resolved it, so
                # _extract_additional_selectable_parameters doesn't
                # re-capture the same label's table as a second parameter.
                table_resolved_labels.append(label)

        # A code position left as a bare label string is either a pure
        # identity token ("FIXaHY" -> "Series Name") or a fixed spec whose
        # value sits in a plain sentence elsewhere, not the nomenclature
        # table. Without this the rep would see no information instead of
        # a quotable fixed spec.
        for code, label in list(segments.items()):
            if not isinstance(label, str):
                continue
            fixed_value = _find_fixed_value(doc, label)
            if fixed_value:
                segments[code] = {"label": label, "values": {"*": fixed_value}}

    return {"aliases": aliases, "segments": segments, "table_resolved_labels": table_resolved_labels}


def _is_eligible_catalogue(path: Path) -> bool:
    """Only .docx product catalogues -- PDF text extraction is too lossy
    for this, and reference documents aren't real product catalogues.
    Reuses the same exclusion rule retriever.py relies on, so the
    registry and the vector index agree on what counts."""
    from app.retriever import _is_single_product_document

    return path.suffix.lower() == ".docx" and _is_single_product_document(path.name)


def build_product_registry(docs_dir: str | Path = "data/approved_docs") -> dict[str, Any]:
    """Rebuilds the full Product Registry from the approved catalogues.
    Product names are assigned with the same batch, collision-aware logic
    the vector index uses (chunker._assign_product_names), so the
    registry and the index always agree on what a product is called --
    not a second, independently-typed name that could drift from it."""
    from app.chunker import _assign_product_names
    from app.document_loader import load_document

    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return {"products": [], "technology_aliases": TECHNOLOGY_ALIASES}

    paths = sorted(p for p in docs_dir.iterdir() if p.is_file() and _is_eligible_catalogue(p))
    loaded_docs = [load_document(p) for p in paths]
    product_names = _assign_product_names(loaded_docs)

    products = []
    for path, loaded_doc in zip(paths, loaded_docs):
        product_name = product_names.get(path.name, path.stem)
        # full_text comes from the loader's own text, not a fresh raw
        # python-docx paragraph walk -- some catalogues declare their
        # category inside a floating text box, which plain doc.paragraphs
        # never sees. The loader already walks into text boxes, so this
        # reuses that instead of re-solving the same problem twice.
        full_text = loaded_doc["text"]
        doc = Document(path)  # still needed: _extract_nomenclature wants the raw docx structure
        nomenclature = _extract_nomenclature(doc)
        all_aliases = {product_name} | {a for a in nomenclature["aliases"] if a}
        # Selectable specs with their own "Selectable <X>" table but no
        # position in the ordering-code suffix (see
        # _extract_additional_selectable_parameters).
        additional_selectable_parameters = _extract_additional_selectable_parameters(
            doc, nomenclature["table_resolved_labels"]
        )

        products.append({
            "product_name": product_name,
            "family": product_name.split()[0] if product_name else "Unknown",
            "product_type": _match_vocab(full_text[:4000], PRODUCT_TYPE_VOCAB),
            "technology": _extract_technology(full_text),
            "install_type": _match_vocab(full_text[:4000], INSTALL_TYPE_VOCAB),
            "target_gas": "H2" if "H2" in full_text or "Hydrogen" in full_text else "Unknown",
            "source_catalogue": path.name,
            "aliases": sorted(all_aliases),
            "nomenclature": nomenclature["segments"] or "Unknown",
            "additional_selectable_parameters": additional_selectable_parameters,
        })

    return {"products": products, "technology_aliases": TECHNOLOGY_ALIASES}


def save_product_registry(registry: dict[str, Any], path: str | Path = REGISTRY_PATH) -> None:
    Path(path).write_text(json.dumps(registry, indent=2), encoding="utf-8")


def rebuild_product_registry(
    docs_dir: str | Path = "data/approved_docs", path: str | Path = REGISTRY_PATH
) -> int:
    """Build and save in one call -- what the reindex triggers actually
    call. Full regeneration every time, not incremental: the registry is
    cheap to rebuild from scratch, and always rebuilding from the current
    approved_docs is what keeps it from drifting out of sync, same as the
    vector index."""
    registry = build_product_registry(docs_dir)
    save_product_registry(registry, path)
    return len(registry["products"])
