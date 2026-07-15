#python -m app.chunker "data/approved_docs/MNST_NC11. Catalogue_Auriga (Leak Detector Series).docx"
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 50

# When a chunk's word-count cutoff would land mid-sentence, look up to this
# many words further for a cleaner stopping point (see _find_boundary_end).
SENTENCE_LOOKAHEAD_WORDS = 20

MIN_TRAILING_CHUNK_WORDS = 30


@dataclass
class Chunk:
    """A single retrieval-ready unit of document text plus its
    provenance metadata."""

    text: str
    document_name: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    section_heading: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convenience for JSON serialization / passing to other modules
        that expect plain dicts rather than dataclass instances."""
        return asdict(self)



def _table_block_to_text(block: dict[str, Any]) -> str:
    """Serialize a table block (any of the loader's three table
    styles) into a compact, readable sentence-per-row form."""
    style = block.get("style", "header")
    parts: list[str] = []

    def _row_text(label: str, values: dict[str, str]) -> str:
        value_text = ", ".join(f"{k}: {v}" for k, v in values.items() if v)
        return f"{label}: {value_text}" if value_text else label

    if style == "header":
        for row in block.get("rows", []):
            parts.append(_row_text(row.get("row_label", ""), row.get("values", {})))

    elif style == "keyvalue":
        for row in block.get("rows", []):
            subsection = row.get("subsection")
            prefix = f"{subsection} - " if subsection else ""
            label, value = row.get("label", ""), row.get("value", "")
            parts.append(f"{prefix}{label}: {value}" if value else f"{prefix}{label}")

    elif style == "complex":
        for row in block.get("top_level_rows", []):
            parts.append(_row_text(row.get("row_label", ""), row.get("values", {})))
        for section in block.get("sections", []):
            section_title = section.get("section_title", "")
            for row in section.get("rows", []):
                row_text = _row_text(row.get("row_label", ""), row.get("values", {}))
                parts.append(f"{section_title} - {row_text}")
            if section.get("options"):
                parts.append(f"{section_title} options: {', '.join(section['options'])}")

    return ". ".join(p for p in parts if p)


def _block_to_text(block: dict[str, Any]) -> str:
    """Convert one loader block into plain text for chunking."""
    block_type = block.get("type")
    if block_type in ("heading", "paragraph", "list_item"):
        return block.get("text", "") or ""
    if block_type == "table":
        return _table_block_to_text(block)
    return ""


# ---------------------------------------------------------------------------
# Boundary-respecting grouping
# ---------------------------------------------------------------------------

def _split_into_boundary_units(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Groups a document's blocks into boundary-respecting units instead of
    one long word stream that a fixed-size window slides over blindly.

    Two boundaries are respected here:
      - A change in section_title (i.e. crossing into a new heading) starts
        a new unit, so a chunk is scoped to one section and the sliding
        window can't straddle two unrelated sections.
      - A table block is always its own unit, regardless of whether the
        section changed, so a chunk boundary never lands mid-row.

    Falling back to a plain word-count split only happens later, inside
    chunk_document(), and only for a unit that's itself bigger than
    chunk_size -- e.g. one very long section, or an unusually large table.
    """
    units: list[dict[str, Any]] = []
    current_words: list[tuple[str, dict[str, Any]]] = []
    current_section: Optional[str] = None

    def flush() -> None:
        nonlocal current_words
        if current_words:
            units.append({"section_heading": current_section, "words": current_words})
            current_words = []

    for block in document.get("blocks", []):
        text = _block_to_text(block)
        if not text:
            continue

        meta = block.get("metadata", {}) or {}
        section_title = meta.get("section_title")

        if block.get("type") == "table":
            # Always its own unit so table rows never get split by a text
            # chunk boundary landing in the middle of them.
            flush()
            word_meta = {
                "page_number": meta.get("page_number"),
                "slide_number": meta.get("slide_number"),
                "section_heading": section_title or current_section,
            }
            units.append(
                {
                    "section_heading": word_meta["section_heading"],
                    "words": [(word, word_meta) for word in text.split()],
                }
            )
            continue

        if section_title != current_section:
            # Crossed into a new section (this also naturally fires on the
            # heading block itself, since a heading's own section_title is
            # the new heading text) -- whatever was accumulated belongs to
            # the section we're leaving.
            flush()
            current_section = section_title

        word_meta = {
            "page_number": meta.get("page_number"),
            "slide_number": meta.get("slide_number"),
            "section_heading": current_section,
        }
        for word in text.split():
            current_words.append((word, word_meta))

    flush()
    return units


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r'[.;][\'")\]]?$')


def _find_boundary_end(
    words: list[tuple[str, dict[str, Any]]],
    target_end: int,
    lookahead: int = SENTENCE_LOOKAHEAD_WORDS,
) -> int:
    """Given a word-count cutoff at target_end, looks up to `lookahead`
    words further for a cleaner stopping point -- a word ending in '.' or
    ';' (optionally followed by a closing quote/bracket) -- so a chunk
    doesn't end mid-sentence like "...probe length can" when
    "...probe length can be customized as per requirement." is only a few
    words further on. Table rows end up covered by the same rule for free:
    _table_block_to_text joins rows with ". ", so a row's last word already
    carries a real trailing period once split into words.

    Falls back to target_end unchanged if no boundary shows up within the
    lookahead window (e.g. one very long run-on sentence/row) -- this is a
    nicer stopping point when one's available, not a guarantee that the
    cut always lands cleanly."""
    n = len(words)
    limit = min(target_end + lookahead, n)
    for i in range(target_end, limit):
        if _SENTENCE_END_RE.search(words[i][0]):
            return i + 1  # include the word carrying the punctuation
    return target_end


def _chunk_word_stream(
    words: list[tuple[str, dict[str, Any]]],
    chunk_size: int,
    overlap: int,
) -> list[list[tuple[str, dict[str, Any]]]]:
    """Slides a window over one boundary unit's words, preferring to end
    each chunk at a sentence or table-row boundary near chunk_size rather
    than cutting exactly at the word count -- see _find_boundary_end. This
    is the fallback for when a unit is bigger than chunk_size, not the
    primary way chunks get split (that's the heading/table boundaries in
    _split_into_boundary_units)."""
    raw_chunks: list[list[tuple[str, dict[str, Any]]]] = []
    start = 0
    n = len(words)
    while start < n:
        target_end = min(start + chunk_size, n)
        end = target_end if target_end >= n else _find_boundary_end(words, target_end)
        raw_chunks.append(words[start:end])
        if end >= n:
            break
        # Next window starts `overlap` words before this chunk's actual end
        # (which may be past chunk_size if a sentence boundary pushed it),
        # not before the original word-count target -- keeps overlap
        # consistent with what was really just emitted. max(..., start + 1)
        # guards against overlap >= the words just added, which would
        # otherwise stall progress.
        start = max(end - overlap, start + 1)

    if len(raw_chunks) > 1 and len(raw_chunks[-1]) < MIN_TRAILING_CHUNK_WORDS:
        raw_chunks[-2].extend(raw_chunks[-1])
        raw_chunks.pop()

    return raw_chunks


def chunk_document(
    document: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a single loaded document (the dict returned by the
    document loader's load_document()/load_docx()) into Chunk objects.

    Chunking now respects logical boundaries first: each section (delimited
    by headings) and each table is chunked on its own, so a chunk can't
    straddle a heading or cut a table in half. A fixed word-count sliding
    window is only used as a fallback within a unit that's itself larger
    than chunk_size.

    Args:
        document: loader output with "filename" and "blocks" keys.
        chunk_size: target words per chunk (keep within ~400-800).
        overlap: words repeated at the start of the next chunk, when a
            section/table is large enough to need splitting.

    Returns:
        Ordered list of Chunk objects for this document.
    """
    boundary_units = _split_into_boundary_units(document)
    if not boundary_units:
        return []

    document_name = document.get("filename", "unknown")

    raw_chunks: list[list[tuple[str, dict[str, Any]]]] = []
    for unit in boundary_units:
        raw_chunks.extend(_chunk_word_stream(unit["words"], chunk_size, overlap))

    # A unit's own trailing remainder is already folded into it by
    # _chunk_word_stream, but a short standalone unit (e.g. a heading with
    # only a line or two under it, or a heading-only stub) can still end up
    # as its own tiny chunk. A heading describes the content that *follows*
    # it, not what precedes it, so small chunks are carried forward and
    # prepended to the next substantial chunk -- that keeps a heading
    # attached to its own section instead of bleeding into an unrelated
    # previous one. The only exception is a small chunk with nothing left
    # after it (the very end of the document): there's no "next" to attach
    # to, so that trailing remainder falls back to merging into the chunk
    # before it.
    merged_chunks: list[list[tuple[str, dict[str, Any]]]] = []
    carry: list[tuple[str, dict[str, Any]]] = []
    for chunk_words in raw_chunks:
        if len(chunk_words) < MIN_TRAILING_CHUNK_WORDS:
            carry.extend(chunk_words)
            continue
        if carry:
            chunk_words = carry + list(chunk_words)
            carry = []
        merged_chunks.append(list(chunk_words))

    if carry:
        # Trailing small chunk(s) with no following chunk to attach to.
        if merged_chunks:
            merged_chunks[-1].extend(carry)
        else:
            merged_chunks.append(carry)

    total_chunks = len(merged_chunks)
    chunks: list[Chunk] = []
    for index, unit_group in enumerate(merged_chunks):
        text = " ".join(word for word, _ in unit_group)
        first_word_meta = unit_group[0][1]
        chunks.append(
            Chunk(
                text=text,
                document_name=document_name,
                chunk_index=index,
                total_chunks=total_chunks,
                page_number=first_word_meta.get("page_number"),
                slide_number=first_word_meta.get("slide_number"),
                section_heading=first_word_meta.get("section_heading"),
            )
        )

    return chunks


def chunk_documents(
    documents: dict[str, Any] | list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Main entry point. Accepts either a single loader output dict or
    a list of them (e.g. from loading every file in a folder) and
    returns one flat list of Chunk objects across all documents.

    Args:
        documents: single loader output dict, or a list of them.
        chunk_size: target words per chunk.
        overlap: words of overlap between consecutive chunks, when a
            section/table needs splitting.

    Returns:
        Flat list of Chunk objects, in document then chunk order.
    """
    if isinstance(documents, dict):
        documents = [documents]

    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Ensure Unicode characters (e.g. ℃) print correctly on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Try to chunk a real file via the existing loader if a path was
    # given on the command line; otherwise fall back to a small
    # synthetic document so this file is runnable with no arguments.
    document: dict[str, Any]

    if len(sys.argv) > 1:
        from app.document_loader import load_document  # existing loader, not modified

        document = load_document(sys.argv[1])
    else:
        sample_paragraph = " ".join(f"word{i}" for i in range(900))
        document = {
            "filename": "sample_demo.txt",
            "file_path": "sample_demo.txt",
            "blocks": [
                {
                    "type": "heading",
                    "text": "Introduction",
                    "metadata": {
                        "section_title": "Introduction",
                        "page_number": 1,
                    },
                },
                {
                    "type": "paragraph",
                    "text": sample_paragraph,
                    "metadata": {
                        "section_title": "Introduction",
                        "page_number": 1,
                    },
                },
            ],
        }

    chunks = chunk_documents(document)

    print(f"Total chunks: {len(chunks)}\n")

    for chunk in chunks:
        print(f"--- Chunk {chunk.chunk_index + 1}/{chunk.total_chunks} ---")
        print(
            f"document_name={chunk.document_name} "
            f"page_number={chunk.page_number} "
            f"slide_number={chunk.slide_number} "
            f"section_heading={chunk.section_heading}"
        )
        print(chunk.text)
        print("-" * 80)