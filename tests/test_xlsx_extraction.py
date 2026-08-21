"""Tests for app.document_loader's XLSX title/table-region detection --
see the comment above _find_table_region_start for the full design
rationale. Two groups: regression (a plain header-first table, and the
real approved Historical Sales workbook must keep working), and new
flexible formats (title/subtitle rows, multiple tables per sheet, and the
cases a naive "sparse row = title" rule gets wrong).

Every "not a title" case asserts NO heading block was produced and the
real data survived untouched -- this mechanism may under-detect a title,
but must never misclassify real data as a title and discard it.
"""
from pathlib import Path

import pytest
from openpyxl import Workbook
from openpyxl.styles import Font

from app.document_loader import extract_xlsx_blocks


def _save(wb: Workbook, tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    wb.save(path)
    return path


def _headings(blocks: list[dict]) -> list[str]:
    return [b["text"] for b in blocks if b["type"] == "heading"]


def _tables(blocks: list[dict]) -> list[dict]:
    return [b for b in blocks if b["type"] == "table"]


def _rows(table_block: dict) -> list[dict]:
    return table_block.get("rows") or table_block.get("top_level_rows") or []


# ---------------------------------------------------------------------------
# Regression: already-correct sheets must stay correct
# ---------------------------------------------------------------------------

def test_plain_header_first_table_unaffected(tmp_path):
    """A normal sheet with no title row -- header is already row 1 --
    must produce zero heading blocks and the same columns as before."""
    wb = Workbook()
    ws = wb.active
    ws.append(["Sr No", "Company", "Sector"])
    ws.append(["1", "Acme", "Oil & Gas"])
    ws.append(["2", "Globex", "Power"])
    path = _save(wb, tmp_path, "plain.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == []
    tables = _tables(blocks)
    assert len(tables) == 1
    assert tables[0]["columns"] == ["Sr No", "Company", "Sector"]
    assert _rows(tables[0])[0] == {"row_label": "1", "values": {"Company": "Acme", "Sector": "Oil & Gas"}}


def test_real_historical_sales_workbook():
    """The actual, currently-approved production file. Both sheets have a
    title (or title+subtitle) row above their real header, and the
    second sheet's reported column width (25) is inflated well beyond
    the 9 columns that ever actually hold data -- exercises the
    populated-width-not-nominal-width logic in _find_table_region_start."""
    path = Path("data/approved_docs/Copy of Historical Sales_Customer and Use Case with Value.xlsx")
    if not path.exists():
        pytest.skip("real approved_docs file not present in this environment")

    blocks = extract_xlsx_blocks(path)
    headings = _headings(blocks)
    assert "Client and Project Details for Work Executed in since April 2022" in headings
    assert "India Business" in headings
    assert "International Business through Distributor - 21Senses Inc." in headings

    tables = _tables(blocks)
    assert len(tables) == 2

    india = tables[0]
    assert set(["Sr No", "FY", "Company Name", "Sector", "Purpose", "Location", "Use Case"]) <= set(india["columns"])
    first_row = _rows(india)[0]
    assert first_row["values"]["Company Name"] == "Reliance Industries Limited (through Solutions 4 Hydrogen)"
    assert first_row["values"]["Sector"] == "Oil & Gas"

    intl = tables[1]
    assert set(["Company Name", "Sector", "Purpose", "Location", "Country"]) <= set(intl["columns"])
    shell_row = _rows(intl)[0]
    assert shell_row["values"]["Company Name"] == "Shell"


# ---------------------------------------------------------------------------
# New: title/subtitle rows above a table
# ---------------------------------------------------------------------------

def test_single_banner_title_row(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Quarterly Report"
    for i, h in enumerate(["Sr No", "Company", "Sector", "Purpose"], 1):
        ws.cell(row=2, column=i, value=h)
    ws.append(["1", "Acme", "Oil & Gas", "Monitoring"])
    ws.append(["2", "Globex", "Power", "Monitoring"])
    path = _save(wb, tmp_path, "banner.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == ["Quarterly Report"]
    tables = _tables(blocks)
    assert len(tables) == 1
    assert tables[0]["columns"] == ["Sr No", "Company", "Sector", "Purpose"]
    assert _rows(tables[0])[0]["values"]["Company"] == "Acme"


def test_stacked_title_and_subtitle_with_blank_row(tmp_path):
    """Title + subtitle + a blank spacer row before the real header --
    the exact shape of the real Historical Sales 'India Business' sheet."""
    wb = Workbook()
    ws = wb.active
    ws["B1"] = "Client Project Details"
    ws["B2"] = "India Business"
    for i, h in enumerate(["Sr No", "FY", "Company Name", "Sector"], 2):
        ws.cell(row=4, column=i, value=h)
    ws.append(["", "1", "2023", "Reliance", "Oil & Gas"])
    path = _save(wb, tmp_path, "stacked.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == ["Client Project Details", "India Business"]
    tables = _tables(blocks)
    assert len(tables) == 1
    assert "Company Name" in tables[0]["columns"]


# ---------------------------------------------------------------------------
# Multiple separate tables on one sheet
# ---------------------------------------------------------------------------

def test_multiple_tables_with_titles_on_one_sheet(tmp_path):
    """Title / Table 1 / blank rows / Title / Table 2 -- each table and
    its own title must be segmented separately, not read as one table
    with the second title/header buried inside the first table's data."""
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Q1 Regional Sales"
    ws["A2"] = "Region"; ws["B2"] = "Rep"; ws["C2"] = "Total"
    ws.append(["North", "Alice", "1000"])
    ws.append(["South", "Bob", "1500"])
    ws["A8"] = "Q2 Product Breakdown"
    ws["A9"] = "Product"; ws["B9"] = "Category"; ws["C9"] = "Units"; ws["D9"] = "Price"; ws["E9"] = "Revenue"
    ws.append(["Widget", "Hardware", "50", "10", "500"])
    path = _save(wb, tmp_path, "multi_table.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == ["Q1 Regional Sales", "Q2 Product Breakdown"]
    tables = _tables(blocks)
    assert len(tables) == 2
    assert set(["Region", "Rep", "Total"]) <= set(tables[0]["columns"])
    assert _rows(tables[0])[0]["row_label"] == "North"
    assert _rows(tables[0])[0]["values"]["Rep"] == "Alice"
    assert set(["Product", "Category", "Units", "Price", "Revenue"]) <= set(tables[1]["columns"])
    assert _rows(tables[1])[0]["row_label"] == "Widget"
    assert _rows(tables[1])[0]["values"]["Category"] == "Hardware"


# ---------------------------------------------------------------------------
# New, and most important: cases a naive "sparse row = title" rule gets
# wrong -- these must NOT produce a heading block, and the row must
# survive as real, intact data.
# ---------------------------------------------------------------------------

def test_two_row_grouped_header_is_not_mistaken_for_a_title(tmp_path):
    """A sparse row that's actually a real, meaningful group-label row
    (merged 'Q1'/'Q2' spanning sub-columns) must be left alone, not
    stripped out as if it were junk."""
    wb = Workbook()
    ws = wb.active
    ws["B1"] = "Q1"; ws["D1"] = "Q2"
    ws["A2"] = "Company"; ws["B2"] = "Revenue"; ws["C2"] = "Cost"; ws["D2"] = "Revenue"; ws["E2"] = "Cost"
    ws.append(["Acme", "100", "50", "120", "55"])
    path = _save(wb, tmp_path, "grouped_header.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == []
    tables = _tables(blocks)
    assert len(tables) == 1
    # Nothing was silently dropped -- the Q1/Q2 row's real text values are
    # still present somewhere in the extracted table's raw grid.
    raw_text = str(tables[0]["raw_grid"])
    assert "Q1" in raw_text and "Q2" in raw_text


def test_repeating_value_is_treated_as_data_not_a_title(tmp_path):
    """The decisive case: a sparse-looking value ('MNST') that recurs as
    real data in later rows must never be reclassified as a title, even
    though a single leading occurrence looks exactly like one."""
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "MNST"; ws["B1"] = "Product A"
    ws.append(["MNST", "Product B"])
    ws.append(["MNST", "Product C"])
    path = _save(wb, tmp_path, "recurring_value.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == []
    tables = _tables(blocks)
    assert len(tables) == 1
    row_labels = [r["row_label"] for r in _rows(tables[0])]
    # All three "MNST" rows survive as real data rows, not just two
    # (which would mean the first was wrongly stripped as a title).
    assert row_labels.count("MNST") == 2  # row 0 becomes the header in this 2-row table; rows 1-2 are data
    assert all(r["row_label"] == "MNST" for r in _rows(tables[0]))


def test_sparse_but_real_first_data_row_is_untouched(tmp_path):
    """A real data row missing a couple of optional fields (matching the
    real Shell row in the actual Historical Sales sheet) must not be
    mistaken for a title just because it's less than fully populated."""
    wb = Workbook()
    ws = wb.active
    for i, h in enumerate(["Sr No", "Company", "Sector", "Purpose", "Location", "Use Case", "Country"], 1):
        ws.cell(row=1, column=i, value=h)
    ws.append(["1", "Shell", "Oil & Gas", "", "H2 Lab", "", "USA"])
    ws.append(["2", "UL", "Testing", "In-Line", "Chamber", "H2 quant", "USA"])
    path = _save(wb, tmp_path, "sparse_real_row.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == []
    tables = _tables(blocks)
    assert _rows(tables[0])[0]["values"]["Company"] == "Shell"


def test_freak_wide_row_far_below_does_not_distort_detection(tmp_path):
    """A single unusually long row far from the top must not inflate the
    'how wide should this table be' measurement and wrongly strip real
    leading rows. Status/Owner vary per row so this test isolates
    title/width detection from the separate _detect_header_row_count
    multi-row-header logic."""
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Status", "Owner"])
    for i in range(12):
        ws.append([f"Item {i}", f"Status{i}", f"Owner{i}"])
    ws.append(["Item X", "StatusX", "OwnerX", "note0", "note1", "note2", "note3", "note4", "note5", "note6"])
    path = _save(wb, tmp_path, "wide_row.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == []
    tables = _tables(blocks)
    assert len(tables) == 1
    # No genuine title here, so detection correctly declines to touch
    # anything -- checked as "nothing was lost," not exact column naming.
    raw_text = str(tables[0]["raw_grid"])
    assert "Name" in raw_text and "Item 0" in raw_text and "Item 11" in raw_text


def test_nominal_sheet_width_wider_than_actual_data_does_not_block_detection(tmp_path):
    """openpyxl can report a sheet as wider than any row ever actually
    populates (old formatting left on unused cells). The width used for
    detection must come from the widest POPULATED row, not the sheet's
    nominal dimensions, or detection silently fails across the whole
    sheet."""
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Report Title"
    for i, h in enumerate(["Sr No", "Company", "Sector"], 1):
        ws.cell(row=2, column=i, value=h)
    ws.append(["1", "Acme", "Oil & Gas"])
    ws.append(["2", "Globex", "Power"])
    # Styling a distant, otherwise-empty cell forces openpyxl to report
    # max_column well beyond column 3.
    ws["T50"].font = Font(bold=True)
    path = _save(wb, tmp_path, "phantom_width.xlsx")

    blocks = extract_xlsx_blocks(path)
    assert _headings(blocks) == ["Report Title"]
    tables = _tables(blocks)
    assert len(tables) == 1
    assert "Company" in tables[0]["columns"]
