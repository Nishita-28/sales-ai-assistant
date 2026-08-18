"""Tests for check_table_structure (app/document_loader.py) -- flags a
table whose detected header/column name is implausibly long, the signal
for a real, confirmed failure mode: the multi-row-header detector
merging unrelated body text into what it thinks is a header, silently
burying real data underneath it. Proven on real approved catalogues --
see the module-level comment above check_table_structure for the exact
case (a ~300-character footnote paragraph became the "name" of every
column in a product spec table).

check_table_coverage (dropped-cell detection) does NOT catch this class
of problem -- every character is still present, just organized wrong --
which is why this is a separate check, not an extension of that one.
"""
from app.document_loader import _build_table_block, _no_bold, check_table_structure


def test_normal_short_headers_produce_no_warning():
    grid = [
        ["Sr No", "Company", "Sector"],
        ["1", "Acme", "Oil & Gas"],
        ["2", "Globex", "Power"],
    ]
    block = _build_table_block(grid, _no_bold, table_index=0, section_title="")
    assert "suspicious_columns" not in block
    assert check_table_structure(block) == []


def test_long_merged_header_is_flagged():
    """Simulates the real failure: a long non-header paragraph absorbed
    as a composite column name across several columns."""
    footnote = (
        "Hydrogen Concentration Terminology Conversion Table for Ease of Reference "
        "Hydrogen Lower Explosive Limit (LEL) - 4% H2 v/v in Air 250% H2 LEL = 10% H2 v/v"
    )
    grid = [
        ["", footnote, footnote, footnote],
        ["Selectable Range", "Range 1", "Range 2", "Range 3"],
        ["Start", "0 ppm", "0 ppm", "0%"],
    ]
    block = _build_table_block(grid, _no_bold, table_index=0, section_title="")
    suspicious = check_table_structure(block)
    assert suspicious, f"expected the long footnote text to be flagged as a suspicious column name (style={block['style']!r}, columns={block.get('columns')!r})"
    assert all(len(name) > 100 for name in suspicious)


def test_keyvalue_style_table_is_never_flagged():
    """A 2-column keyvalue table has no column-name concept -- nothing
    for this check to look at, regardless of how long its values are."""
    grid = [
        ["Warranty", "Standard product warranty is 12 months from the date of invoice " * 5],
        ["AMC", "Comprehensive AMC covers periodic calibration " * 5],
    ]
    from app.document_loader import _cell_is_bold_docx

    def is_bold(r, c):
        return c == 0  # first column bold -- triggers keyvalue style detection

    block = _build_table_block(grid, is_bold, table_index=0, section_title="")
    assert block["style"] == "keyvalue"
    assert check_table_structure(block) == []
