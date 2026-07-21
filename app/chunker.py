#python -m app.chunker "data/approved_docs/MNST_NC11. Catalogue_Auriga (Leak Detector Series).docx"
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

# Chunks pack whole blocks, so this means "how many short facts share a
# chunk" rather than a raw word ceiling.
DEFAULT_CHUNK_SIZE = 50
DEFAULT_OVERLAP = 10

# How far past a chunk's target cutoff to look for a sentence boundary
# instead of cutting mid-sentence.
SENTENCE_LOOKAHEAD_WORDS = 10

MIN_TRAILING_CHUNK_WORDS = 10


@dataclass
class Chunk:
    """A single retrieval-ready chunk of text plus its provenance metadata."""

    text: str
    document_name: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    section_heading: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _document_title(document_name: str) -> str:
    """Strips the file extension for use as a chunk-text prefix, so a
    query naming the product can still match."""
    return os.path.splitext(document_name)[0]


def _table_block_to_text(block: dict[str, Any]) -> str:
    """Serializes any of the loader's three table styles into a compact,
    sentence-per-row form."""
    style = block.get("style", "header")
    parts: list[str] = []

    def _row_text(label: str, values: dict[str, str]) -> str:
        # Collapse repeated identical values across columns to one mention.
        unique_values = set(values.values())
        if len(values) > 1 and len(unique_values) == 1:
            value_text = next(iter(unique_values))
        else:
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
    """Groups blocks so a section heading or table always starts a new unit."""
    units: list[dict[str, Any]] = []
    current_blocks: list[list[tuple[str, dict[str, Any]]]] = []
    current_section: Optional[str] = None

    def flush() -> None:
        nonlocal current_blocks
        if current_blocks:
            units.append({"section_heading": current_section, "blocks": current_blocks})
            current_blocks = []

    for block in document.get("blocks", []):
        text = _block_to_text(block)
        if not text:
            continue

        meta = block.get("metadata", {}) or {}
        section_title = meta.get("section_title")

        if block.get("type") == "table":
            flush()  # a table is always its own unit
            word_meta = {
                "page_number": meta.get("page_number"),
                "slide_number": meta.get("slide_number"),
                "section_heading": section_title or current_section,
            }
            units.append(
                {
                    "section_heading": word_meta["section_heading"],
                    "blocks": [[(word, word_meta) for word in text.split()]],
                }
            )
            continue

        if section_title != current_section:
            flush()
            current_section = section_title

        word_meta = {
            "page_number": meta.get("page_number"),
            "slide_number": meta.get("slide_number"),
            "section_heading": current_section,
        }
        current_blocks.append([(word, word_meta) for word in text.split()])

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
    """Looks past target_end for a sentence ending; falls back to
    target_end if none turns up."""
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
    """Fallback for a block bigger than chunk_size -- slides a window,
    preferring a sentence boundary over a hard cut."""
    raw_chunks: list[list[tuple[str, dict[str, Any]]]] = []
    start = 0
    n = len(words)
    while start < n:
        target_end = min(start + chunk_size, n)
        end = target_end if target_end >= n else _find_boundary_end(words, target_end)
        raw_chunks.append(words[start:end])
        if end >= n:
            break
        # Guards against overlap stalling progress.
        start = max(end - overlap, start + 1)

    if len(raw_chunks) > 1 and len(raw_chunks[-1]) < MIN_TRAILING_CHUNK_WORDS:
        raw_chunks[-2].extend(raw_chunks[-1])
        raw_chunks.pop()

    return raw_chunks


def _pack_blocks_into_chunks(
    blocks: list[list[tuple[str, dict[str, Any]]]],
    chunk_size: int,
    overlap: int,
) -> list[list[tuple[str, dict[str, Any]]]]:
    """Greedily packs whole blocks into chunks up to chunk_size, never
    splitting one block across two chunks."""
    chunks: list[list[tuple[str, dict[str, Any]]]] = []
    current: list[tuple[str, dict[str, Any]]] = []

    for block_words in blocks:
        if not block_words:
            continue
        if len(block_words) > chunk_size:
            if current:
                chunks.append(current)
                current = []
            chunks.extend(_chunk_word_stream(block_words, chunk_size, overlap))
            continue

        if current and len(current) + len(block_words) > chunk_size:
            chunks.append(current)
            current = []
        current.extend(block_words)

    if current:
        chunks.append(current)

    return chunks


def chunk_document(
    document: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunks one loaded document into Chunk objects, respecting section
    and table boundaries."""
    boundary_units = _split_into_boundary_units(document)
    if not boundary_units:
        return []

    document_name = document.get("filename", "unknown")

    raw_chunks: list[list[tuple[str, dict[str, Any]]]] = []
    for unit in boundary_units:
        raw_chunks.extend(_pack_blocks_into_chunks(unit["blocks"], chunk_size, overlap))

    # A tiny chunk gets carried forward into the next substantial one
    # instead of bleeding into the previous, unrelated chunk.
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
        if merged_chunks:
            merged_chunks[-1].extend(carry)
        else:
            merged_chunks.append(carry)

    document_title = _document_title(document_name)

    total_chunks = len(merged_chunks)
    chunks: list[Chunk] = []
    for index, unit_group in enumerate(merged_chunks):
        text = f"{document_title}: " + " ".join(word for word, _ in unit_group)
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
    """Chunks a single loader output dict, or a list of them, into one flat
    list of Chunk objects."""
    if isinstance(documents, dict):
        documents = [documents]

    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Chunk a real file if given on the command line, otherwise a small sample.
    document: dict[str, Any]

    if len(sys.argv) > 1:
        from app.document_loader import load_document

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
