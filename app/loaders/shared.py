"""Shared utilities for document extraction across all formats."""
from __future__ import annotations

import re
from typing import Any, Callable

IsBoldFn = Callable[[int, int], bool]


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


def _build_keyvalue_table(
    grid: list[list[str]], is_bold_fn: IsBoldFn, table_index: int, section_title: str
) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    current_subsection: str | None = None

    for row_idx, row in enumerate(grid):
        if not any(row):
            continue
        label = row[0]
        value = row[1] if len(row) > 1 else ""

        if label and is_bold_fn(row_idx, 0):
            current_subsection = label
            continue
        if not label:
            continue

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
