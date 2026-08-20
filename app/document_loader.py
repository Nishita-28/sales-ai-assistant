"""Document extraction and parsing logic for all supported formats
(docx, pptx, pdf, csv, md, txt, and OCR'd images)."""
#python -m app.chunker "data/approved_docs/MNST_NC2. Catalogue_PORTaHY H2 LD (Leak Detector Series).docx"
#python -m app.document_loader "data/approved_docs/MNST_NC11. Catalogue_Auriga (Leak Detector Series).docx"

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

IsBoldFn = Callable[[int, int], bool]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize whitespace inside extracted text."""
    if text is None:
        return ""
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# OCR -- text baked into images (scanned PDF pages, screenshots/photos
# embedded in a document, or a standalone image file). Kept as one shared
# helper so every extractor that needs it (PDF page fallback, embedded
# images in DOCX/PPTX, standalone image files) goes through the same
# engine and the same cached instance -- initializing RapidOCR loads its
# models, which is too slow to repeat per call.
# ---------------------------------------------------------------------------

# Below this many characters, a PDF page's native text layer is treated as
# absent (scanned page) rather than just a short page of real text.
OCR_MIN_NATIVE_TEXT_CHARS = 20

_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def _run_ocr(image_bytes: bytes) -> str:
    """Runs OCR on raw image bytes and returns the recognized text, lines
    joined with newlines in reading order. Returns "" if nothing is
    recognized (e.g. a decorative image with no text), or if the bytes
    aren't a raster format Pillow can decode at all (embedded vector
    graphics like SVG/WMF/EMF are common in real DOCX/PPTX files and
    aren't something OCR applies to) -- either case is a normal, expected
    outcome for some embedded images, not an error."""
    import io

    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return ""

    result, _ = _get_ocr_engine()(np.array(image))
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def _no_bold(_row: int, _col: int) -> bool:
    """is_bold_fn for formats with no formatting info (CSV/TXT/plain MD)."""
    return False


def _row_nonempty(row: list[str]) -> list[str]:
    return [c for c in row if c]


def _row_raw_text_counts(row: list[str]) -> Counter[str]:
    """Counts a row's non-empty cells, treating a fully-merged row as one
    value so it isn't flagged as a dropped duplicate."""
    nonempty = _row_nonempty(row)
    if not nonempty:
        return Counter()
    if len(set(nonempty)) == 1 and len(nonempty) == len(row):
        return Counter({nonempty[0]: 1})
    return Counter(nonempty)


def _looks_like_header_cell(text: str) -> bool:
    if not text:
        return False
    if len(text) > 40:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    return digit_ratio < 0.4


def _row_extends_header(prev_row: list[str], row: list[str]) -> bool:
    """True if `row` likely continues a multi-level header -- it has a
    blank cell or repeats a value from the row above."""
    has_blank_cell = any(not c for c in row)
    repeats_row_above = bool(set(_row_nonempty(row)) & set(_row_nonempty(prev_row)))
    return has_blank_cell or repeats_row_above


def _detect_header_row_count(grid: list[list[str]], max_header_rows: int = 3) -> int:
    """Counts consecutive rows forming a multi-level header block; row 0
    always counts."""
    if not grid:
        return 1
    count = 1
    for row in grid[1 : max_header_rows + 1]:
        prev_row = grid[count - 1]
        nonempty = _row_nonempty(row)
        if len(nonempty) < 2:
            break
        if not all(_looks_like_header_cell(c) for c in nonempty):
            break
        if not _row_extends_header(prev_row, row):
            break
        count += 1
    return count


def _build_composite_columns(grid: list[list[str]], header_row_count: int) -> list[str]:
    if not grid:
        return []
    total_cols = len(grid[0])
    columns: list[str] = []
    for j in range(total_cols):
        parts: list[str] = []
        prev: str | None = None
        for i in range(header_row_count):
            text = grid[i][j] if j < len(grid[i]) else ""
            if text and text != prev:
                parts.append(text)
            if text:
                prev = text
        columns.append(" - ".join(parts) if parts else f"Column {j + 1}")

    seen: dict[str, int] = {}
    out: list[str] = []
    for name in columns:
        seen[name] = seen.get(name, 0) + 1
        out.append(f"{name} ({seen[name]})" if seen[name] > 1 else name)
    return out


def _is_section_header_row(row_idx: int, grid: list[list[str]], is_bold_fn: IsBoldFn) -> bool:
    """True only for a genuine section-separator row: a single or merged
    bold cell."""
    row = grid[row_idx]
    nonempty = _row_nonempty(row)
    if not nonempty:
        return False
    unique_texts = set(nonempty)
    is_single_or_merged = len(nonempty) == 1 or (len(unique_texts) == 1 and len(nonempty) == len(row))
    if not is_single_or_merged:
        return False
    return is_bold_fn(row_idx, 0)


def _is_option_row(row: list[str]) -> bool:
    return len(_row_nonempty(row)) == 1


def _row_is_full_header_row(row: list[str], row_idx: int, is_bold_fn: IsBoldFn) -> bool:
    """True only if every non-empty cell in the row is bold."""
    nonempty_cols = [c for c, text in enumerate(row) if text]
    if len(nonempty_cols) < 2:
        return False
    return all(is_bold_fn(row_idx, c) for c in nonempty_cols)


def _build_keyvalue_table(
    grid: list[list[str]], is_bold_fn: IsBoldFn, table_index: int, section_title: str
) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    current_subsection: str | None = None

    for row_idx, row in enumerate(grid):
        if not any(row):
            continue

        if row_idx == 0 and _row_is_full_header_row(row, row_idx, is_bold_fn):
            continue

        if _is_section_header_row(row_idx, grid, is_bold_fn):
            current_subsection = _row_nonempty(row)[0]
            continue

        label = row[0]
        if not label:
            continue
        value = row[1] if len(row) > 1 else ""

        rows_out.append({"label": label, "value": value, "subsection": current_subsection})

    return {
        "table_index": table_index,
        "section_title": section_title or None,
        "style": "keyvalue",
        "rows": rows_out,
        "raw_grid": grid,
    }


def _build_simple_table(grid: list[list[str]], table_index: int, section_title: str) -> dict[str, Any]:
    if not grid:
        return {
            "table_index": table_index,
            "section_title": section_title or None,
            "style": "header",
            "columns": [],
            "rows": [],
            "raw_grid": [],
        }

    header_row = grid[0]
    data_rows = grid[1:]
    raw_columns = [cell if cell else f"Column {i + 1}" for i, cell in enumerate(header_row)]

    seen: dict[str, int] = {}
    columns: list[str] = []
    for name in raw_columns:
        seen[name] = seen.get(name, 0) + 1
        columns.append(f"{name} ({seen[name]})" if seen[name] > 1 else name)

    rows_out: list[dict[str, Any]] = []
    for row in data_rows:
        if not any(row):
            continue
        row_label = row[0] if row else ""
        values = {columns[i]: row[i] for i in range(1, min(len(columns), len(row)))}
        rows_out.append({"row_label": row_label, "values": values})

    return {
        "table_index": table_index,
        "section_title": section_title or None,
        "style": "header",
        "columns": columns,
        "rows": rows_out,
        "raw_grid": grid,
    }


def _build_complex_table(
    grid: list[list[str]], is_bold_fn: IsBoldFn, table_index: int, section_title: str
) -> dict[str, Any]:
    if not grid:
        return {
            "table_index": table_index,
            "section_title": section_title or None,
            "style": "complex",
            "columns": [],
            "top_level_rows": [],
            "sections": [],
            "raw_grid": [],
        }

    header_row_count = _detect_header_row_count(grid)
    columns = _build_composite_columns(grid, header_row_count)

    sections: list[dict[str, Any]] = []
    top_level_rows: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None

    for row_idx in range(header_row_count, len(grid)):
        row = grid[row_idx]
        if not any(row):
            continue

        if _is_section_header_row(row_idx, grid, is_bold_fn):
            label = _row_nonempty(row)[0]
            current_section = {"section_title": label, "rows": [], "options": []}
            sections.append(current_section)
            continue

        if current_section is not None and _is_option_row(row):
            current_section["options"].append(_row_nonempty(row)[0])
            continue

        row_label = row[0] if row else ""
        values = {
            columns[i]: row[i]
            for i in range(1, min(len(columns), len(row)))
            if row[i]
        }
        row_entry = {"row_label": row_label, "values": values}
        (current_section["rows"] if current_section is not None else top_level_rows).append(row_entry)

    return {
        "table_index": table_index,
        "section_title": section_title or None,
        "style": "complex",
        "columns": columns,
        "top_level_rows": top_level_rows,
        "sections": sections,
        "raw_grid": grid,
    }


def _detect_table_style(grid: list[list[str]], is_bold_fn: IsBoldFn) -> str:
    if not grid:
        return "header"
    total_cols = len(grid[0])

    if total_cols == 2:
        # A single-row, 2-column table has no room for a header row plus a
        # data row underneath it -- "header" style is meaningless here, so
        # it's always a label/value pair.
        if len(grid) == 1:
            return "keyvalue"

        # A vertically-merged cell in column 0 (one shared label next to
        # several distinct per-row notes/options) has no merge information
        # by the time it reaches this grid -- resolve_table_grid() reads
        # python-docx cells directly, and python-docx returns the same
        # merged cell's text for every row it spans rather than exposing
        # the merge, so every constituent row looks like it has its own,
        # identical column-0 value; reading that as a real header row would
        # make the header itself double as a second, near-duplicate data row.
        #
        # Only trusted for a genuinely long, sentence-like column-0 value
        # (a real merged label reads like a description, not an
        # identifier) -- a short repeated value (a brand name, a code) is
        # not evidence of a merge at all, it's just a legitimate repeated
        # key across real, distinct rows (e.g. three rows that all
        # legitimately start with "MNST"). 40 chars comfortably separates a
        # short identifier from a multi-clause descriptive sentence.
        _MIN_MERGED_LABEL_LENGTH = 40
        col0_values = [row[0] for row in grid if row]
        merged_first_column = (
            bool(col0_values)
            and len(col0_values[0]) >= _MIN_MERGED_LABEL_LENGTH
            and len(set(col0_values)) == 1
        )
        if merged_first_column:
            return "keyvalue"

        bold_beyond_first = any(is_bold_fn(i, 0) for i in range(1, len(grid)))
        return "keyvalue" if bold_beyond_first else "header"

    header_row_count = _detect_header_row_count(grid)
    has_sections = any(
        _is_section_header_row(i, grid, is_bold_fn) for i in range(header_row_count, len(grid))
    )
    return "complex" if (header_row_count > 1 or has_sections) else "header"


def _collect_table_output_texts(block: dict[str, Any]) -> Counter[str]:
    """Counts every data value in the structured output, using counts
    rather than a set so a repeated value isn't covered by one survivor."""
    counts: Counter[str] = Counter()
    style = block.get("style", "header")

    if style == "keyvalue":
        for row in block.get("rows", []):
            if row.get("label"):
                counts[row["label"]] += 1
            if row.get("value"):
                counts[row["value"]] += 1
            if row.get("subsection"):
                counts[row["subsection"]] += 1
        # Row 0 is either real data or a skipped header row, so credit it either way.
        raw_grid = block.get("raw_grid", [])
        if raw_grid:
            counts.update(_row_raw_text_counts(raw_grid[0]))

    elif style == "complex":
        for row in block.get("top_level_rows", []):
            if row.get("row_label"):
                counts[row["row_label"]] += 1
            for v in row.get("values", {}).values():
                if v:
                    counts[v] += 1
        for section in block.get("sections", []):
            if section.get("section_title"):
                counts[section["section_title"]] += 1
            for row in section.get("rows", []):
                if row.get("row_label"):
                    counts[row["row_label"]] += 1
                for v in row.get("values", {}).values():
                    if v:
                        counts[v] += 1
            for o in section.get("options", []):
                if o:
                    counts[o] += 1

    else:  # "header" / simple
        for row in block.get("rows", []):
            if row.get("row_label"):
                counts[row["row_label"]] += 1
            for v in row.get("values", {}).values():
                if v:
                    counts[v] += 1

    return counts


def check_table_coverage(block: dict[str, Any]) -> list[str]:
    """Compares a table's raw data rows against its structured output and
    returns any values that were silently dropped."""
    raw_grid = block.get("raw_grid", [])
    style = block.get("style", "header")

    if style == "complex":
        header_row_count = _detect_header_row_count(raw_grid)
    elif style == "header":
        header_row_count = 1
    else:  # "keyvalue": no fixed header-row count -- row 0 handled specially above
        header_row_count = 0

    raw_counts: Counter[str] = Counter()
    for row in raw_grid[header_row_count:]:
        raw_counts.update(_row_raw_text_counts(row))

    output_counts = _collect_table_output_texts(block)

    missing_counts = raw_counts - output_counts  # Counter subtraction keeps only positive remainders
    missing: list[str] = []
    for text, count in missing_counts.items():
        missing.extend([text] * count)
    return sorted(missing)


# A real column/segment header is a short label -- a few words at most. A
# name this long is a strong signal the multi-row-header detector
# (_detect_header_row_count / _row_extends_header) merged unrelated body
# text into what it thinks is a header row, rather than that literal text
# actually dropping (check_table_coverage passes cleanly in this case --
# every character is still present, just organized wrong). Not proof the
# header is wrong -- a table could legitimately have a long name -- but
# cheap, high-signal, and worth flagging for review rather than shipping
# silently.
_MAX_REASONABLE_COLUMN_NAME_LENGTH = 100


def check_table_structure(block: dict[str, Any]) -> list[str]:
    """Flags column/segment names that are implausibly long for a real
    header. keyvalue-style tables have no column-name concept to check."""
    return [name for name in block.get("columns", []) if len(name) > _MAX_REASONABLE_COLUMN_NAME_LENGTH]


def _build_table_block(
    grid: list[list[str]], is_bold_fn: IsBoldFn, table_index: int, section_title: str
) -> dict[str, Any]:
    """Classify + build a table block, given any format's resolved grid."""
    style = _detect_table_style(grid, is_bold_fn)

    if style == "keyvalue":
        structured = _build_keyvalue_table(grid, is_bold_fn, table_index, section_title)
        block: dict[str, Any] = {"type": "table", "style": "keyvalue", "rows": structured["rows"]}
    elif style == "complex":
        structured = _build_complex_table(grid, is_bold_fn, table_index, section_title)
        block = {
            "type": "table",
            "style": "complex",
            "columns": structured["columns"],
            "top_level_rows": structured["top_level_rows"],
            "sections": structured["sections"],
        }
    else:
        structured = _build_simple_table(grid, table_index, section_title)
        block = {"type": "table", "style": "header", "columns": structured["columns"], "rows": structured["rows"]}

    block["raw_grid"] = grid

    dropped = check_table_coverage(block)
    if dropped:
        block["dropped_cells"] = dropped

    suspicious = check_table_structure(block)
    if suspicious:
        block["suspicious_columns"] = suspicious

    return block


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _cell_is_bold_docx(cell) -> bool:
    runs = [r for p in cell.paragraphs for r in p.runs if r.text.strip()]
    if not runs:
        return False
    return all(r.bold for r in runs)


def resolve_table_grid(table) -> list[list[str]]:
    """Merged cells are already text-duplicated by python-docx's row.cells."""
    grid = [[_clean_text(cell.text) for cell in row.cells] for row in table.rows]
    max_len = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([""] * (max_len - len(row)))
    return grid


# Word wraps each text box in two identical copies of the same content
# (a modern version and a legacy fallback) -- read only one or text boxes
# get duplicated.
_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_MC_ALTERNATE_CONTENT = f"{{{_MC_NS}}}AlternateContent"
_MC_CHOICE = f"{{{_MC_NS}}}Choice"
_MC_FALLBACK = f"{{{_MC_NS}}}Fallback"


def _iter_textbox_paragraph_elements(paragraph_element):
    """Yields paragraph elements from text boxes anchored to this paragraph."""
    from docx.oxml.ns import qn

    for alt in paragraph_element.iter(_MC_ALTERNATE_CONTENT):
        source = alt.find(_MC_CHOICE)
        if source is None:
            source = alt.find(_MC_FALLBACK)
        if source is None:
            continue
        for txbx in source.iter(qn("w:txbxContent")):
            yield from txbx.iter(qn("w:p"))


def _paragraph_type_docx(paragraph) -> str:
    style_name = ""
    if paragraph.style is not None and paragraph.style.name:
        style_name = paragraph.style.name.lower()
    if style_name.startswith("heading") or style_name == "title":
        return "heading"
    if "list" in style_name:
        return "list_item"

    text = paragraph.text.strip()
    runs = [r for r in paragraph.runs if r.text.strip()]
    if text and runs and len(text) <= 100:
        all_bold = all(r.bold for r in runs)
        sizes = [r.font.size.pt for r in runs if r.font.size is not None]
        looks_larger = bool(sizes) and min(sizes) >= 13
        if all_bold or looks_larger:
            return "heading"
    return "paragraph"


def extract_docx_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    from docx import Document
    from docx.text.paragraph import Paragraph

    path = Path(file_path)
    filename = path.name
    document = Document(path)

    blocks: list[dict[str, Any]] = []
    table_index = 0
    current_section = ""
    block_index = 0

    for element in document.element.body.iterchildren():
        tag = element.tag.split("}")[-1]

        if tag == "p":
            paragraph = Paragraph(element, document)
            text = _clean_text(paragraph.text)
            if text:
                block_type = _paragraph_type_docx(paragraph)
                if block_type == "heading":
                    current_section = text
                blocks.append({
                    "type": block_type,
                    "text": text,
                    "metadata": {"filename": filename, "block_index": block_index, "section_title": current_section or None},
                })
                block_index += 1

            for tb_element in _iter_textbox_paragraph_elements(element):
                tb_paragraph = Paragraph(tb_element, document)
                tb_text = _clean_text(tb_paragraph.text)
                if not tb_text:
                    continue
                # Text boxes are floating captions, so they don't set current_section.
                blocks.append({
                    "type": _paragraph_type_docx(tb_paragraph),
                    "text": tb_text,
                    "metadata": {"filename": filename, "block_index": block_index, "section_title": current_section or None},
                })
                block_index += 1

        elif tag == "tbl":
            table = document.tables[table_index]
            grid = resolve_table_grid(table)
            is_bold_fn = lambda r, c, t=table: _cell_is_bold_docx(t.rows[r].cells[c])
            block = _build_table_block(grid, is_bold_fn, table_index, current_section)
            block["metadata"] = {
                "filename": filename, "block_index": block_index, "section_title": current_section or None,
                "table_index": table_index, "num_rows": len(grid), "num_columns": len(grid[0]) if grid else 0,
            }
            blocks.append(block)
            table_index += 1
            block_index += 1

    # Embedded pictures (screenshots, scanned certificates, nameplate
    # photos) aren't part of the paragraph/table walk above at all -- python-
    # docx exposes them only via the part relationships, not as inline text.
    # This is a flat pass over every embedded image in the file, not
    # positioned relative to the section it visually appears under -- good
    # enough to make the text searchable/retrievable, not a layout-accurate
    # placement.
    for rel in document.part.rels.values():
        if "image" not in rel.reltype:
            continue
        text = _run_ocr(rel.target_part.blob)
        for para in re.split(r"\n\s*\n", text):
            cleaned = _clean_text(para)
            if not cleaned:
                continue
            blocks.append({
                "type": _paragraph_type_plaintext(cleaned),
                "text": cleaned,
                "metadata": {"filename": filename, "block_index": block_index, "section_title": None, "extraction_method": "ocr"},
            })
            block_index += 1

    return blocks


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------

def _resolve_pptx_table_grid(table) -> list[list[str]]:
    """Re-expands merged cells so the grid matches the table's visible shape."""
    n_rows, n_cols = len(table.rows), len(table.columns)
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for r in range(n_rows):
        for c in range(n_cols):
            cell = table.cell(r, c)
            if cell.is_spanned:
                continue  # filled in when we process its merge origin below
            text = _clean_text(cell.text)
            span_h = cell.span_height if cell.is_merge_origin else 1
            span_w = cell.span_width if cell.is_merge_origin else 1
            for dr in range(span_h):
                for dc in range(span_w):
                    grid[r + dr][c + dc] = text
    return grid


def _cell_is_bold_pptx(cell) -> bool:
    runs = [run for p in cell.text_frame.paragraphs for run in p.runs if run.text.strip()]
    if not runs:
        return False
    return all(run.font.bold for run in runs)


def extract_pptx_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    path = Path(file_path)
    filename = path.name
    prs = Presentation(path)

    blocks: list[dict[str, Any]] = []
    table_index = 0
    current_section = ""
    block_index = 0

    for slide_num, slide in enumerate(prs.slides, start=1):

        def process_shapes(shapes) -> None:
            nonlocal table_index, current_section, block_index
            for shape in shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    # Groups have no table/text of their own; recurse into their children.
                    process_shapes(shape.shapes)
                    continue

                if shape.has_table:
                    table = shape.table
                    grid = _resolve_pptx_table_grid(table)
                    is_bold_fn = lambda r, c, t=table: _cell_is_bold_pptx(t.cell(r, c))
                    block = _build_table_block(grid, is_bold_fn, table_index, current_section)
                    block["metadata"] = {
                        "filename": filename, "block_index": block_index, "section_title": current_section or None,
                        "table_index": table_index, "slide_number": slide_num,
                        "num_rows": len(grid), "num_columns": len(grid[0]) if grid else 0,
                    }
                    blocks.append(block)
                    table_index += 1
                    block_index += 1
                    continue

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    text = _run_ocr(shape.image.blob)
                    for para in re.split(r"\n\s*\n", text):
                        cleaned = _clean_text(para)
                        if not cleaned:
                            continue
                        blocks.append({
                            "type": _paragraph_type_plaintext(cleaned),
                            "text": cleaned,
                            "metadata": {
                                "filename": filename, "block_index": block_index,
                                "section_title": current_section or None, "slide_number": slide_num,
                                "extraction_method": "ocr",
                            },
                        })
                        block_index += 1
                    continue

                if not shape.has_text_frame:
                    continue

                is_title = shape == slide.shapes.title
                for paragraph in shape.text_frame.paragraphs:
                    text = _clean_text(paragraph.text)
                    if not text:
                        continue

                    if is_title:
                        block_type = "heading"
                        current_section = text
                    else:
                        runs = [r for r in paragraph.runs if r.text.strip()]
                        all_bold = bool(runs) and all(r.font.bold for r in runs)
                        block_type = "heading" if (all_bold and len(text) <= 100) else "paragraph"
                        if block_type == "heading":
                            current_section = text

                    blocks.append({
                        "type": block_type,
                        "text": text,
                        "metadata": {
                            "filename": filename, "block_index": block_index,
                            "section_title": current_section or None, "slide_number": slide_num,
                        },
                    })
                    block_index += 1

        process_shapes(slide.shapes)

    return blocks


# ---------------------------------------------------------------------------
# PDF -- text only, no font/bold info, so headings use a plaintext
# heuristic and tables aren't detected.
# ---------------------------------------------------------------------------

def _paragraph_type_plaintext(text: str) -> str:
    if len(text) <= 80 and not text.endswith((".", ",", ";", ":")):
        if text.isupper() or text.istitle():
            return "heading"
    if re.match(r"^\s*([-*\u2022]|\d+[.)])\s+", text):
        return "list_item"
    return "paragraph"


def _ocr_pdf_page(fitz_doc, page_index: int) -> str:
    """Rasterizes one page (0-indexed) and OCRs it -- the fallback for a
    page with no usable native text layer (a scanned page)."""
    pixmap = fitz_doc[page_index].get_pixmap(dpi=200)
    return _run_ocr(pixmap.tobytes("png"))


def extract_pdf_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    from pypdf import PdfReader

    path = Path(file_path)
    filename = path.name
    reader = PdfReader(path)

    blocks: list[dict[str, Any]] = []
    current_section = ""
    block_index = 0
    fitz_doc = None  # opened lazily, only if some page actually needs OCR

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        used_ocr = False
        if len(raw_text.strip()) < OCR_MIN_NATIVE_TEXT_CHARS:
            import fitz

            if fitz_doc is None:
                fitz_doc = fitz.open(path)
            ocr_text = _ocr_pdf_page(fitz_doc, page_num - 1)
            if ocr_text.strip():
                raw_text = ocr_text
                used_ocr = True

        for para in re.split(r"\n\s*\n", raw_text):
            text = _clean_text(para)
            if not text:
                continue
            block_type = _paragraph_type_plaintext(text)
            if block_type == "heading":
                current_section = text
            metadata = {"filename": filename, "block_index": block_index, "section_title": current_section or None, "page_number": page_num}
            if used_ocr:
                metadata["extraction_method"] = "ocr"
            blocks.append({"type": block_type, "text": text, "metadata": metadata})
            block_index += 1

    if fitz_doc is not None:
        fitz_doc.close()

    return blocks


# ---------------------------------------------------------------------------
# Standalone image files -- the whole file is one OCR pass, since there's
# no independent structure (headings, tables) to detect the way there is
# in a text document.
# ---------------------------------------------------------------------------

def extract_image_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    filename = path.name

    text = _run_ocr(path.read_bytes())
    blocks: list[dict[str, Any]] = []
    block_index = 0
    for para in re.split(r"\n\s*\n", text):
        cleaned = _clean_text(para)
        if not cleaned:
            continue
        blocks.append({
            "type": _paragraph_type_plaintext(cleaned),
            "text": cleaned,
            "metadata": {"filename": filename, "block_index": block_index, "section_title": None, "extraction_method": "ocr"},
        })
        block_index += 1

    return blocks


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def extract_csv_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    """A CSV is one table, always 'header' style -- no formatting info to
    detect section rows."""
    path = Path(file_path)
    filename = path.name

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        grid = [[_clean_text(cell) for cell in row] for row in reader if any(cell.strip() for cell in row)]

    max_len = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([""] * (max_len - len(row)))

    block = _build_table_block(grid, _no_bold, table_index=0, section_title="")
    block["metadata"] = {
        "filename": filename, "block_index": 0, "section_title": None,
        "table_index": 0, "num_rows": len(grid), "num_columns": len(grid[0]) if grid else 0,
    }
    return [block]


# ---------------------------------------------------------------------------
# XLSX title/table-region detection -- a leading row above a real table
# (a report title, a subtitle, a section label) is common in real-world
# spreadsheets. Left undetected, it gets misread as the header row itself,
# corrupting every column name below it.
#
# Deliberately not a "sparse row = title" rule: a row with exactly one
# populated cell can just as easily be real data (e.g. a narrow table
# whose first column repeats the same category value down every row).
# Two independent signals must BOTH agree before a row is reclassified
# out of the table body:
#
# 1. Coverage gap against the table's own "core columns" (the columns
#    populated in every row of a consistent, regular block) -- a
#    candidate row must populate at most _MAX_TITLE_POPULATED_CORE_
#    COLUMNS of them. An absolute count, not a percentage, since a
#    percentage threshold doesn't generalize across table widths.
# 2. Non-recurrence -- none of the candidate row's populated core-column
#    values may reappear in that same column anywhere in the table body.
#    A title is a one-off label; real row data recurs.
#
# A row that doesn't clearly satisfy both stays part of the table,
# untouched. This can misjudge a real title as ordinary data, but it can
# never misjudge real data as a title and discard it -- the failure mode
# that actually matters for a knowledge base an AI answers questions from.
# ---------------------------------------------------------------------------

# How many consecutive rows must share a consistent set of populated
# columns before that's trusted as "a real table starts here" rather than
# coincidence.
_TABLE_REGION_WINDOW = 3

# A real table's core columns are the ones populated in every row of that
# consistent block; a candidate leading row must leave at least half of
# them empty to even be considered for title reclassification.
_TABLE_REGION_MIN_CORE_FRACTION = 0.5

_MAX_TITLE_POPULATED_CORE_COLUMNS = 1


def _find_table_region_start(
    rows: list[list[str]],
    window: int = _TABLE_REGION_WINDOW,
    min_core_frac: float = _TABLE_REGION_MIN_CORE_FRACTION,
) -> tuple[int, set[int], bool]:
    """Scans forward for the first index where `window` consecutive rows
    share a consistent, substantial set of populated columns -- trusted
    as where a real table begins. Returns (start_index, core_columns,
    found); found=False means no such consistent region was located, and
    the other two values fall back to row 0 as-is.

    The "how wide should a real table be" yardstick is the widest
    POPULATED row count anywhere in `rows`, not len(rows[0]) -- openpyxl
    can report a sheet as wider than it really is due to leftover
    formatting on cells that were never used, which would otherwise make
    the core-size bar unreachable and silently disable detection."""
    n = len(rows)
    if n == 0:
        return 0, set(), False
    real_width = max((sum(1 for c in row if c) for row in rows), default=0)
    for h in range(n):
        group = rows[h : h + window]
        # A single-row "group" trivially intersects with itself, which is
        # no corroboration at all -- never accept fewer than 2 rows of
        # agreement, even if the sheet is running out of rows to check.
        if len(group) < min(window, 2):
            break
        col_sets = [{i for i, c in enumerate(r) if c} for r in group]
        core = set.intersection(*col_sets) if col_sets else set()
        if len(core) >= max(2, real_width * min_core_frac):
            return h, core, True
    return 0, {i for i, c in enumerate(rows[0]) if c}, False


def _classify_leading_title_rows(
    rows: list[list[str]],
    table_start: int,
    core_columns: set[int],
    max_populated_core: int = _MAX_TITLE_POPULATED_CORE_COLUMNS,
) -> list[int]:
    """Walks backward from table_start, reclassifying a leading row as a
    confident title only when it satisfies both signals described above.
    Stops at the first row that fails either check -- every row from
    there up stays part of the table body, ambiguous and untouched."""
    confident_titles: list[int] = []
    for r in range(table_start - 1, -1, -1):
        row = rows[r]
        populated_core = {i for i, c in enumerate(row) if c} & core_columns
        recurs = any(
            rows[body_row][col] == row[col]
            for col in populated_core
            for body_row in range(table_start, len(rows))
        )
        if len(populated_core) <= max_populated_core and not recurs:
            confident_titles.append(r)
        else:
            break
    confident_titles.reverse()
    return confident_titles


class _XlsxRegion:
    """One logical unit within a worksheet: an optional run of confident
    title rows, plus the table body they sit above. rows/cell_rows cover
    just this region -- index 0 is the region's own first row, whether
    that's a title row or the table itself."""

    __slots__ = ("rows", "cell_rows", "title_row_indexes", "table_start")

    def __init__(
        self, rows: list[list[str]], cell_rows: list[list[Any]], title_row_indexes: list[int], table_start: int
    ) -> None:
        self.rows = rows
        self.cell_rows = cell_rows
        self.title_row_indexes = title_row_indexes
        self.table_start = table_start


def _segment_xlsx_grid(
    rows: list[list[str]], cell_rows: list[list[Any]], gap_before: list[bool]
) -> list["_XlsxRegion"]:
    """Splits a worksheet's rows into one or more regions at blank-row gaps.
    A sheet can legitimately hold several separate tables (a title, a
    table, blank rows, another title, another table); without this,
    treating the whole sheet as a single table would bury the second
    table's real header inside the first table's data rows, as if it were
    just another entry.

    A blank gap only becomes a real boundary once the rows before it
    already form a genuine, self-sufficient table on their own (i.e.
    _find_table_region_start succeeds within that chunk alone) --
    otherwise those rows are merged forward into the next chunk instead of
    being finalized as their own title-only, tableless region. This also
    keeps a title+subtitle pair sitting above its own table across a blank
    spacer row from being wrongly split into two pieces."""
    raw_chunks: list[tuple[list[list[str]], list[list[Any]]]] = []
    cur_rows, cur_cells = [rows[0]], [cell_rows[0]]
    for i in range(1, len(rows)):
        if gap_before[i]:
            raw_chunks.append((cur_rows, cur_cells))
            cur_rows, cur_cells = [], []
        cur_rows.append(rows[i])
        cur_cells.append(cell_rows[i])
    raw_chunks.append((cur_rows, cur_cells))

    regions: list[_XlsxRegion] = []
    pending_rows: list[list[str]] = []
    pending_cells: list[list[Any]] = []
    for chunk_rows, chunk_cells in raw_chunks:
        combined_rows = pending_rows + chunk_rows
        combined_cells = pending_cells + chunk_cells
        table_start, core, found = _find_table_region_start(combined_rows)
        if found:
            titles = _classify_leading_title_rows(combined_rows, table_start, core)
            # table_start from _find_table_region_start is only the right
            # slice point when rows above it were actually confirmed as
            # titles -- otherwise a real, meaningful row (e.g. a two-row
            # grouped header, correctly left out of `titles`) could get
            # silently excluded by slicing at the detected region-start
            # index regardless of what was actually confirmed as a title.
            # When titles is empty, nothing was confidently reclassified,
            # so the whole region -- starting at its own row 0 -- is the table.
            effective_start = (max(titles) + 1) if titles else 0
            regions.append(_XlsxRegion(combined_rows, combined_cells, titles, effective_start))
            pending_rows, pending_cells = [], []
        else:
            pending_rows, pending_cells = combined_rows, combined_cells
    if pending_rows:
        # Nothing ever followed to complete it -- surface as-is (treat
        # row 0 as the header, today's exact pre-existing behavior)
        # rather than silently drop it.
        regions.append(_XlsxRegion(pending_rows, pending_cells, [], 0))
    return regions


# ---------------------------------------------------------------------------
# XLSX -- one or more table blocks per worksheet, with real bold
# formatting available.
# ---------------------------------------------------------------------------

def _read_xlsx_sheet_grid(worksheet) -> tuple[list[list[str]], list[list[Any]], list[bool]]:
    """Returns (text_grid, cell_rows, gap_before), dropping blank rows
    while keeping bold formatting lookups intact. gap_before[i] is True
    when text_grid[i] was preceded by one or more dropped blank rows --
    used by _segment_xlsx_grid to detect separate tables on one sheet."""
    text_rows: list[list[str]] = []
    cell_rows: list[list[Any]] = []
    gap_before: list[bool] = []
    pending_gap = False

    for row in worksheet.iter_rows(values_only=False):
        texts = [_clean_text(str(cell.value)) if cell.value is not None else "" for cell in row]
        if any(texts):
            text_rows.append(texts)
            cell_rows.append(list(row))
            gap_before.append(pending_gap)
            pending_gap = False
        else:
            pending_gap = True

    max_len = max((len(row) for row in text_rows), default=0)
    for row in text_rows:
        row.extend([""] * (max_len - len(row)))

    return text_rows, cell_rows, gap_before


def _cell_is_bold_xlsx(cell_rows: list[list[Any]], row_idx: int, col_idx: int) -> bool:
    if row_idx >= len(cell_rows) or col_idx >= len(cell_rows[row_idx]):
        return False
    font = cell_rows[row_idx][col_idx].font
    return bool(font and font.bold)


def extract_xlsx_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    """One or more blocks per worksheet, reading computed values rather
    than formula text. A confidently-detected title row (see
    _classify_leading_title_rows) becomes its own heading block -- text
    preserved, never discarded, just no longer misread as column
    headers -- and each self-sufficient table region (see
    _segment_xlsx_grid) becomes its own table block."""
    from openpyxl import load_workbook

    path = Path(file_path)
    filename = path.name
    workbook = load_workbook(path, data_only=True)

    blocks: list[dict[str, Any]] = []
    table_index = 0
    block_index = 0

    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        grid, cell_rows, gap_before = _read_xlsx_sheet_grid(worksheet)
        if not grid:
            continue

        for region in _segment_xlsx_grid(grid, cell_rows, gap_before):
            for title_idx in region.title_row_indexes:
                title_text = next((c for c in region.rows[title_idx] if c), "")
                if not title_text:
                    continue
                blocks.append({
                    "type": "heading",
                    "text": title_text,
                    "metadata": {"filename": filename, "block_index": block_index, "section_title": sheet_name},
                })
                block_index += 1

            table_rows = region.rows[region.table_start :]
            table_cells = region.cell_rows[region.table_start :]
            if not table_rows:
                continue
            is_bold_fn = lambda r, c, cr=table_cells: _cell_is_bold_xlsx(cr, r, c)
            block = _build_table_block(table_rows, is_bold_fn, table_index, sheet_name)
            block["metadata"] = {
                "filename": filename, "block_index": block_index, "section_title": sheet_name,
                "table_index": table_index, "num_rows": len(table_rows),
                "num_columns": len(table_rows[0]) if table_rows else 0,
            }
            blocks.append(block)
            table_index += 1
            block_index += 1

    return blocks


# ---------------------------------------------------------------------------
# Markdown -- headings (#), list items (-/*), pipe tables (| a | b |)
# ---------------------------------------------------------------------------

def extract_markdown_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    filename = path.name
    lines = path.read_text(encoding="utf-8").splitlines()

    blocks: list[dict[str, Any]] = []
    current_section = ""
    block_index = 0
    table_index = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            text = _clean_text(heading_match.group(2))
            current_section = text
            blocks.append({"type": "heading", "text": text, "metadata": {"filename": filename, "block_index": block_index, "section_title": current_section}})
            block_index += 1
            i += 1
            continue

        if re.match(r"^\|.*\|\s*$", stripped):
            table_lines = []
            while i < len(lines) and re.match(r"^\|.*\|\s*$", lines[i].strip()):
                table_lines.append(lines[i].strip())
                i += 1
            grid = []
            for row_num, tline in enumerate(table_lines):
                if row_num == 1 and re.match(r"^\|[\s:|-]+\|\s*$", tline):
                    continue  # markdown's "|---|---|" alignment separator row
                cells = [c.strip() for c in tline.strip("|").split("|")]
                grid.append([_clean_text(c) for c in cells])

            block = _build_table_block(grid, _no_bold, table_index, current_section)
            block["metadata"] = {
                "filename": filename, "block_index": block_index, "section_title": current_section or None,
                "table_index": table_index, "num_rows": len(grid), "num_columns": len(grid[0]) if grid else 0,
            }
            blocks.append(block)
            table_index += 1
            block_index += 1
            continue

        if re.match(r"^([-*+]|\d+[.)])\s+", stripped):
            text = _clean_text(re.sub(r"^([-*+]|\d+[.)])\s+", "", stripped))
            blocks.append({"type": "list_item", "text": text, "metadata": {"filename": filename, "block_index": block_index, "section_title": current_section or None}})
            block_index += 1
            i += 1
            continue

        text = _clean_text(stripped)
        blocks.append({"type": "paragraph", "text": text, "metadata": {"filename": filename, "block_index": block_index, "section_title": current_section or None}})
        block_index += 1
        i += 1

    return blocks


# ---------------------------------------------------------------------------
# TXT -- blank-line-separated paragraphs, no tables
# ---------------------------------------------------------------------------

def extract_txt_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    filename = path.name
    raw_text = path.read_text(encoding="utf-8")

    blocks: list[dict[str, Any]] = []
    current_section = ""
    block_index = 0

    for para in re.split(r"\n\s*\n", raw_text):
        text = _clean_text(para)
        if not text:
            continue
        block_type = _paragraph_type_plaintext(text)
        if block_type == "heading":
            current_section = text
        blocks.append({"type": block_type, "text": text, "metadata": {"filename": filename, "block_index": block_index, "section_title": current_section or None}})
        block_index += 1

    return blocks


# ---------------------------------------------------------------------------
# Document-level coverage check -- catches content the block loop never
# visited at all, not just values dropped inside a found table.
# ---------------------------------------------------------------------------

DOCUMENT_COVERAGE_WARNING_THRESHOLD = 0.90  # warn if under 90% of raw words made it out


def _raw_word_count(file_path: str | Path) -> Optional[int]:
    """Independently counts words in the raw source file. Returns None if
    there's no counter for this format yet."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".docx":
        from docx import Document
        from docx.oxml.ns import qn
        from docx.text.paragraph import Paragraph

        doc = Document(path)
        # Joins each paragraph's runs before counting words, since Word
        # sometimes splits a single word across two runs.
        count = sum(len(p.text.split()) for p in doc.paragraphs)
        count += sum(len(cell.text.split()) for t in doc.tables for row in t.rows for cell in row.cells)
        for p_element in doc.element.body.iterchildren():
            if p_element.tag != qn("w:p"):
                continue
            for tb_element in _iter_textbox_paragraph_elements(p_element):
                count += len(Paragraph(tb_element, doc).text.split())
        return count

    if suffix == ".pptx":
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        prs = Presentation(path)
        count = 0

        def walk(shapes) -> None:
            nonlocal count
            for shape in shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    walk(shape.shapes)
                    continue
                if shape.has_text_frame:
                    count += len(shape.text_frame.text.split())
                if shape.has_table:
                    for row in shape.table.rows:
                        for cell in row.cells:
                            count += len(cell.text.split())

        for slide in prs.slides:
            walk(slide.shapes)
        return count

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return sum(len((page.extract_text() or "").split()) for page in reader.pages)

    if suffix == ".csv":
        # csv.reader avoids misreading empty comma-separated cells as words.
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            return sum(len(cell.split()) for row in reader for cell in row)

    if suffix == ".xlsx":
        from openpyxl import load_workbook

        # read_only is fine here since only cell values are needed, not formatting.
        workbook = load_workbook(path, data_only=True, read_only=True)
        count = 0
        for sheet_name in workbook.sheetnames:
            for row in workbook[sheet_name].iter_rows(values_only=True):
                for value in row:
                    if value is not None:
                        count += len(str(value).split())
        return count

    if suffix in (".md", ".markdown", ".txt"):
        return len(path.read_text(encoding="utf-8").split())

    return None


def _extracted_word_count(blocks: list[dict[str, Any]]) -> int:
    """Counts words in the extracted blocks, including every raw table cell."""
    count = 0
    for block in blocks:
        if block.get("type") == "table":
            for row in block.get("raw_grid", []):
                for cell in row:
                    if cell:
                        count += len(cell.split())
        else:
            count += len((block.get("text") or "").split())
    return count


def check_document_coverage(file_path: str | Path, blocks: list[dict[str, Any]]) -> Optional[str]:
    """Coarse check: does roughly as much text end up in the blocks as
    exists in the source? A large shortfall means content was missed."""
    raw_count = _raw_word_count(file_path)
    if not raw_count:
        return None

    extracted_count = _extracted_word_count(blocks)
    coverage = extracted_count / raw_count

    if coverage < DOCUMENT_COVERAGE_WARNING_THRESHOLD:
        return (
            f"{Path(file_path).name}: only {extracted_count}/{raw_count} words "
            f"({coverage:.0%}) made it into extracted blocks -- some content may "
            f"never be reaching the extractor's block loop at all, not just being "
            f"lost inside a table."
        )
    return None


_EXTRACTORS: dict[str, Callable[[str | Path], list[dict[str, Any]]]] = {
    ".docx": extract_docx_blocks,
    ".pptx": extract_pptx_blocks,
    ".pdf": extract_pdf_blocks,
    ".csv": extract_csv_blocks,
    ".xlsx": extract_xlsx_blocks,
    ".md": extract_markdown_blocks,
    ".markdown": extract_markdown_blocks,
    ".txt": extract_txt_blocks,
    ".png": extract_image_blocks,
    ".jpg": extract_image_blocks,
    ".jpeg": extract_image_blocks,
    ".bmp": extract_image_blocks,
    ".tiff": extract_image_blocks,
}


def extract_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    """Dispatches to the right extractor based on file extension. All
    extractors return the same block schema."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {suffix} ({path.name})")
    return extractor(path)


_load_document_cache: dict[Path, tuple[float, int, dict[str, Any]]] = {}


def load_document(file_path: str | Path, verbose: bool = True) -> dict[str, Any]:
    """Loads a file into readable text plus ordered blocks, along with any
    coverage warnings. Parsing (especially OCR on embedded images) is slow
    enough that callers can end up loading the same untouched file twice in
    one process -- e.g. a full reindex loads every document, then a
    registry rebuild loads the same catalogues again -- so results are
    cached per process, keyed by path + mtime + size, and reused as long as
    the file hasn't changed on disk."""
    path = Path(file_path)
    try:
        stat = path.stat()
        cache_key = path.resolve()
        cached = _load_document_cache.get(cache_key)
        if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]
    except OSError:
        stat = None
        cache_key = None

    blocks = extract_blocks(path)

    readable_parts: list[str] = []
    warnings: list[str] = []
    for block in blocks:
        if block["type"] == "table":
            t_idx = block["metadata"].get("table_index", 0)
            readable_parts.append(f"[Table {t_idx + 1}: {len(block['raw_grid'])} rows]")
            if block.get("dropped_cells"):
                warnings.append(
                    f"Table {t_idx + 1} in {path.name}: possibly dropped cells "
                    f"{block['dropped_cells']}"
                )
            if block.get("suspicious_columns"):
                worst = max(block["suspicious_columns"], key=len)
                preview = worst[:80] + ("..." if len(worst) > 80 else "")
                warnings.append(
                    f"Table {t_idx + 1} in {path.name}: header detection may have misfired -- "
                    f"found an unusually long column name ({len(worst)} characters, starts "
                    f"with \"{preview}\"). This table's structure may need manual review."
                )
        else:
            readable_parts.append(block["text"])

    document_warning = check_document_coverage(path, blocks)
    if document_warning:
        warnings.append(document_warning)

    if verbose:
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)

    result = {
        "filename": path.name,
        "file_path": str(path),
        "text": "\n".join(readable_parts),
        "blocks": blocks,
        "warnings": warnings,
    }

    if stat is not None and cache_key is not None:
        _load_document_cache[cache_key] = (stat.st_mtime, stat.st_size, result)

    return result


# Backward-compatible alias for older callers.
def load_docx(file_path: str | Path, verbose: bool = True) -> dict[str, Any]:
    return load_document(file_path, verbose=verbose)


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path-to-file> [--out output.json]")
        raise SystemExit(1)

    file_arg = sys.argv[1]
    out_path = None
    if "--out" in sys.argv:
        idx = sys.argv.index("--out")
        if idx + 1 < len(sys.argv):
            out_path = sys.argv[idx + 1]

    document = load_document(file_arg, verbose=True)
    blocks = document["blocks"]
    output_text = json.dumps(blocks, indent=2, ensure_ascii=False)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Written to {out_path} (UTF-8)")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(output_text)
