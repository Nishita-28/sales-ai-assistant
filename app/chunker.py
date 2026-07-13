
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 50


MIN_TRAILING_CHUNK_WORDS = 100


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
# Word-level flattening
# ---------------------------------------------------------------------------

def _words_with_metadata(document: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Flatten every block's text into a single ordered list of
    (word, source_metadata) pairs.

    Tagging metadata at the word level (rather than the chunk level)
    means a sliding window can cut through the middle of a document
    at any point and each resulting chunk still knows which page,
    slide, and section its *first* word came from -- without needing
    a second pass to re-derive that after windowing.
    """
    units: list[tuple[str, dict[str, Any]]] = []
    for block in document.get("blocks", []):
        text = _block_to_text(block)
        if not text:
            continue

        meta = block.get("metadata", {}) or {}
        word_meta = {
            "page_number": meta.get("page_number"),
            "slide_number": meta.get("slide_number"),
            "section_heading": meta.get("section_title"),
        }
        for word in text.split():
            units.append((word, word_meta))

    return units


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(
    document: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a single loaded document (the dict returned by the
    document loader's load_document()/load_docx()) into overlapping
    word-bounded Chunk objects.

    Args:
        document: loader output with "filename" and "blocks" keys.
        chunk_size: target words per chunk (keep within ~400-800).
        overlap: words repeated at the start of the next chunk.

    Returns:
        Ordered list of Chunk objects for this document.
    """
    units = _words_with_metadata(document)
    if not units:
        return []

    document_name = document.get("filename", "unknown")
    step = max(chunk_size - overlap, 1)  # guard against overlap >= chunk_size

    raw_chunks: list[list[tuple[str, dict[str, Any]]]] = []
    start = 0
    n = len(units)
    while start < n:
        end = min(start + chunk_size, n)
        raw_chunks.append(units[start:end])
        if end == n:
            break
        start += step

    # Fold a too-small trailing chunk into the previous one so the
    # last chunk isn't a near-empty sliver.
    if len(raw_chunks) > 1 and len(raw_chunks[-1]) < MIN_TRAILING_CHUNK_WORDS:
        raw_chunks[-2].extend(raw_chunks[-1])
        raw_chunks.pop()

    total_chunks = len(raw_chunks)
    chunks: list[Chunk] = []
    for index, unit_group in enumerate(raw_chunks):
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
        overlap: words of overlap between consecutive chunks.

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

if __name__ == "__main__":
    import sys

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
                    "metadata": {"section_title": "Introduction", "page_number": 1},
                },
                {
                    "type": "paragraph",
                    "text": sample_paragraph,
                    "metadata": {"section_title": "Introduction", "page_number": 1},
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
        print(f"preview: {chunk.text[:200]!r}\n")