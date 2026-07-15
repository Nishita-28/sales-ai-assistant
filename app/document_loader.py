"""Document extraction and parsing logic for all supported formats
(docx, pptx, pdf, csv, md, txt).

Previously split across app/loaders/shared.py, app/loaders/docx_loader.py,
and app/document_loader.py -- merged into one file since there was no
longer a good reason for the package split. app/loaders/ can be deleted
once this file replaces it.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Callable

IsBoldFn = Callable[[int, int], bool]


# ---------------------------------------------------------------------------
# Shared helpers (formerly app/loaders/shared.py)
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize whitespace inside extracted text."""
    if text is None:
        return ""
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _no_bold(_row: int, _col: int) -> bool:
    """is_bold_fn for formats with no formatting info (CSV/TXT/plain MD)."""
    return False


def _row_nonempty(row: list[str]) -> list[str]:
    return [c for c in row if c]


def _looks_like_header_cell(text: str) -> bool:
    if not text:
        return False
    if len(text) > 40:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    return digit_ratio < 0.4


def _detect_header_row_count(grid: list[list[str]], max_header_rows: int = 3) -> int:
    """Count consecutive rows from the top forming a (possibly
    multi-level) header block. Row 0 always counts."""
    if not grid:
        return 1
    count = 1
    for row in grid[1 : max_header_rows + 1]:
        nonempty = _row_nonempty(row)
        if len(nonempty) < 2:
            break
        if not all(_looks_like_header_cell(c) for c in nonempty):
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
    """True only for a genuine section-separator row: a single non-empty
    cell, or a merged row where every cell repeats the same text -- AND
    that cell is bold. A normal two-column data row (label + value both
    non-empty) never qualifies here, even if its label happens to be bold."""
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
    """True only if every non-empty cell in the row is bold -- the typical
    formatting signature of a genuine column-header row like
    'Feature | Value'. A real data row like 'Operating Temperature | -20C'
    almost always has only its label bold (if anything), not its value, so
    this doesn't fire for it -- unlike a blanket 'row 0 is always the
    header' assumption, which drops real data when a table has no header
    row at all."""
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
            # Only skip row 0 when it's actually formatted like a header
            # (both cells bold) -- e.g. "Feature | Value". A table with no
            # header row at all, where row 0 is real data, is left alone.
            continue

        # A subsection separator is a genuine single-cell/merged bold row
        # (e.g. "Display Enclosure" spanning the row with no value cell) --
        # not just "column 0 happens to be bold," which a normal
        # label/value data row can also be (e.g. "Start-Up Time | 5 seconds"
        # with a bold label). Using is_bold_fn(row_idx, 0) alone here was
        # what caused ordinary bold-labeled data rows to be misread as
        # subsection headers and silently dropped.
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
        bold_beyond_first = any(is_bold_fn(i, 0) for i in range(1, len(grid)))
        return "keyvalue" if bold_beyond_first else "header"

    header_row_count = _detect_header_row_count(grid)
    has_sections = any(
        _is_section_header_row(i, grid, is_bold_fn) for i in range(header_row_count, len(grid))
    )
    return "complex" if (header_row_count > 1 or has_sections) else "header"


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
    return block


# ---------------------------------------------------------------------------
# DOCX (formerly app/loaders/docx_loader.py)
# ---------------------------------------------------------------------------

def _cell_is_bold_docx(cell) -> bool:
    runs = [r for p in cell.paragraphs for r in p.runs if r.text.strip()]
    if not runs:
        return False
    return all(r.bold for r in runs)


def resolve_table_grid(table) -> list[list[str]]:
    """docx: merges are already text-duplicated by python-docx's row.cells."""
    grid = [[_clean_text(cell.text) for cell in row.cells] for row in table.rows]
    max_len = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([""] * (max_len - len(row)))
    return grid


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
            if not text:
                continue
            block_type = _paragraph_type_docx(paragraph)
            if block_type == "heading":
                current_section = text
            blocks.append({
                "type": block_type,
                "text": text,
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

    return blocks


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------

def _resolve_pptx_table_grid(table) -> list[list[str]]:
    """python-pptx does NOT auto-duplicate merged-cell text (unlike
    python-docx) -- it exposes cell.is_merge_origin / is_spanned /
    span_height / span_width instead. Re-expand manually so spanned
    positions carry the origin cell's text, matching docx's grid shape."""
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

    path = Path(file_path)
    filename = path.name
    prs = Presentation(path)

    blocks: list[dict[str, Any]] = []
    table_index = 0
    current_section = ""
    block_index = 0

    for slide_num, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
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

    return blocks


# ---------------------------------------------------------------------------
# PDF (pypdf) -- text only; pypdf has no reliable font/bold or table
# extraction, so headings use the same short-line heuristic as the
# docx fallback, and tables are NOT detected.
# ---------------------------------------------------------------------------

def _paragraph_type_plaintext(text: str) -> str:
    if len(text) <= 80 and not text.endswith((".", ",", ";", ":")):
        if text.isupper() or text.istitle():
            return "heading"
    if re.match(r"^\s*([-*\u2022]|\d+[.)])\s+", text):
        return "list_item"
    return "paragraph"


def extract_pdf_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    from pypdf import PdfReader

    path = Path(file_path)
    filename = path.name
    reader = PdfReader(path)

    blocks: list[dict[str, Any]] = []
    current_section = ""
    block_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        for para in re.split(r"\n\s*\n", raw_text):
            text = _clean_text(para)
            if not text:
                continue
            block_type = _paragraph_type_plaintext(text)
            if block_type == "heading":
                current_section = text
            blocks.append({
                "type": block_type,
                "text": text,
                "metadata": {"filename": filename, "block_index": block_index, "section_title": current_section or None, "page_number": page_num},
            })
            block_index += 1

    return blocks


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def extract_csv_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    """A CSV is one table, always classified as 'header' style since
    there is no formatting metadata to detect bold section rows."""
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
# Dispatcher + generic loader
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, Callable[[str | Path], list[dict[str, Any]]]] = {
    ".docx": extract_docx_blocks,
    ".pptx": extract_pptx_blocks,
    ".pdf": extract_pdf_blocks,
    ".csv": extract_csv_blocks,
    ".md": extract_markdown_blocks,
    ".markdown": extract_markdown_blocks,
    ".txt": extract_txt_blocks,
}


def extract_blocks(file_path: str | Path) -> list[dict[str, Any]]:
    """Dispatch to the right extractor based on file extension.
    All extractors return the same block schema (see extract_docx_blocks)."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {suffix} ({path.name})")
    return extractor(path)


def load_document(file_path: str | Path) -> dict[str, Any]:
    """Load any supported file into readable text plus ordered blocks."""
    path = Path(file_path)
    blocks = extract_blocks(path)

    readable_parts: list[str] = []
    for block in blocks:
        if block["type"] == "table":
            t_idx = block["metadata"].get("table_index", 0)
            readable_parts.append(f"[Table {t_idx + 1}: {len(block['raw_grid'])} rows]")
        else:
            readable_parts.append(block["text"])

    return {"filename": path.name, "file_path": str(path), "text": "\n".join(readable_parts), "blocks": blocks}


# Backward-compatible alias -- existing callers using load_docx() keep working.
def load_docx(file_path: str | Path) -> dict[str, Any]:
    return load_document(file_path)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path-to-file> [--out output.json]")
        raise SystemExit(1)

    file_arg = sys.argv[1]
    out_path = None
    if "--out" in sys.argv:
        idx = sys.argv.index("--out")
        if idx + 1 < len(sys.argv):
            out_path = sys.argv[idx + 1]

    blocks = extract_blocks(file_arg)
    output_text = json.dumps(blocks, indent=2, ensure_ascii=False)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Written to {out_path} (UTF-8)")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(output_text)