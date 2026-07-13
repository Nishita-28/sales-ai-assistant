from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Callable

from app.loaders.docx_loader import extract_docx_blocks
from app.loaders.shared import (
    IsBoldFn,
    _build_table_block,
    _clean_text,
    _detect_header_row_count,
    _detect_table_style,
    _is_option_row,
    _is_section_header_row,
    _no_bold,
    _row_nonempty,
)


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
# docx fallback, and tables are NOT detected (see note in the reply).
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
