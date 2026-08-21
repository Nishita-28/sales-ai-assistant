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
    product_name: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    section_heading: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _document_title(document_name: str) -> str:
    """Strips the file extension -- the fallback product identity for a
    document with no headings at all (e.g. a table-only spreadsheet)."""
    return os.path.splitext(document_name)[0]


def _document_headings(document: dict[str, Any]) -> list[str]:
    return [
        (b.get("text") or "").strip()
        for b in document.get("blocks", [])
        if b.get("type") == "heading" and (b.get("text") or "").strip()
    ]


# Fallback only, for filenames that don't follow the "Catalogue_<model>
# (<category>)" convention (see _distinctive_filename_tokens) -- generic
# words that would otherwise falsely count as a "distinctive" match.
_GENERIC_FILENAME_WORDS = {
    "catalogue", "catalog", "series", "leak", "detector", "detectors",
    "portable", "fixed", "hydrogen", "sensor", "sensors", "peso",
    "approved", "draft", "mnst", "nc1", "nc2", "nc11", "c1", "of", "the",
    "multi", "nano", "sense", "in",
}


def _distinctive_filename_tokens(filename: str) -> set[str]:
    """MNST's catalogue filenames follow a real, consistent convention:
    "Catalogue_<exact model name> (<generic category descriptor>)" -- e.g.
    "Catalogue_FIXaHY H2 LD (Leak Detector Series)". The parenthesized
    part is always the generic descriptor, never the model, so stripping
    it (rather than maintaining a hand-typed stopword list) gives a
    precise signal for what actually distinguishes this document. Falls
    back to a small stopword list for the one filename that doesn't
    follow this convention (no "Catalogue_" marker, no parentheses)."""
    name = os.path.splitext(filename)[0]
    marker = re.search(r"catalogu?e_", name, re.IGNORECASE)
    if marker:
        core = name[marker.end():]
        core = re.sub(r"\([^)]*\)", "", core)  # drop the generic descriptor
        return set(re.findall(r"[a-zA-Z0-9]+", core.lower()))
    tokens = set(re.findall(r"[a-zA-Z0-9]+", name.lower()))
    return {t for t in tokens if t not in _GENERIC_FILENAME_WORDS and len(t) >= 3}


def _reorder_headings_by_filename_match(headings: list[str], filename: str) -> list[str]:
    """A document can contain more than one plausible "heading" -- e.g. a
    floating text box's caption extracted ahead of the real title purely
    because of where it's anchored in the file, not because it's what a
    reader would see first on the page. Plain first-heading order can
    silently assign the wrong product identity (e.g. a header banner like
    "PORTABLE H2 LEAK DETECTOR" extracted before its own title paragraph,
    "AURIGA").

    Sorts by how many filename tokens each heading shares, not merely
    whether it shares any: within one product family (e.g. "FIXaHY"),
    several real headings all contain the shared brand word, including an
    over-generic series-level banner ("FIXaHY LEAK DETECTOR SERIES"). A
    boolean any-match can't tell that apart from the actual specific model
    heading ("FIXaHY H2 LD XX"); an overlap count can, since the specific
    heading shares every token (brand + target gas + type) while the
    generic banner only shares the brand. Stable otherwise, so a document
    with no match at all keeps its original order."""
    distinctive = _distinctive_filename_tokens(filename)
    if not distinctive:
        return headings

    def overlap(heading: str) -> int:
        heading_tokens = set(re.findall(r"[a-zA-Z0-9]+", heading.lower()))
        return len(heading_tokens & distinctive)

    scored = sorted(enumerate(headings), key=lambda pair: (-overlap(pair[1]), pair[0]))
    return [h for _, h in scored] if any(overlap(h) for h in headings) else headings


def _is_long_placeholder_blob(token: str) -> bool:
    """True for a token made entirely of repeated-letter pairs, 4+ chars
    long -- e.g. "RRNNVVII" (the glued-together "RR* NN VV* II" ordering-
    code placeholders). Deliberately requires 4+ chars so a short, genuine
    suffix like "XX" is never touched -- several real product titles end
    in "XX" as their own printed name (e.g. "FIXaHY H2 LD XX"), not a
    placeholder Claude is guessing at; only a long blob like this is
    unambiguously code, never a real word."""
    token = token.rstrip("*")
    return (
        token.isalpha() and len(token) >= 4 and len(token) % 2 == 0
        and all(token[i] == token[i + 1] for i in range(0, len(token), 2))
    )


def _clean_ordering_code_heading(heading: str) -> str:
    """A document's own chosen title heading can itself be the ordering-
    code template, rendered with hyphens instead of the tabs used in the
    "Product Ordering Nomenclature" section further down -- e.g. the
    4220MA catalogue's title heading is literally
    "FIXaHY-G/P/E-4220MA-RRNNVVII", not a human-written product name.
    Only cleans up when the heading contains an unmistakable code marker
    (a slash-separated option list like "G/P/E", or a long repeated-
    letter placeholder blob like "RRNNVVII") -- otherwise returns the
    heading unchanged, so every other document's real printed title
    (including ones that end in a short "XX") is left exactly as-is."""
    tokens = re.split(r"[\s-]+", heading)
    if not any("/" in t or _is_long_placeholder_blob(t) for t in tokens):
        return heading
    kept = [t for t in tokens if "/" not in t and not _is_long_placeholder_blob(t)]
    return " ".join(kept) if kept else heading


def assign_product_name(document: dict[str, Any], other_documents: tuple[dict[str, Any], ...] = ()) -> str:
    """Uses the document's own first heading as its product identity,
    since catalogues put the product name first -- after reordering
    candidates so a heading matching the filename's distinctive brand
    token wins over one that merely happens to be extracted first (see
    _reorder_headings_by_filename_match). When another document shares
    the same heading at the same position -- e.g. two model variants
    under one family name -- walks forward to the first heading that
    actually differs between them, so the two don't collide onto the
    same identity. Falls back to the filename when a document has no
    headings at all."""
    filename = document.get("filename", "unknown")
    own_headings = _reorder_headings_by_filename_match(_document_headings(document), filename)
    other_heading_seqs = [
        _reorder_headings_by_filename_match(_document_headings(d), d.get("filename", "unknown"))
        for d in other_documents
    ]

    for i, heading in enumerate(own_headings):
        if not any(i < len(seq) and seq[i] == heading for seq in other_heading_seqs):
            return _clean_ordering_code_heading(heading)

    if own_headings:
        return _clean_ordering_code_heading(own_headings[-1])
    return _document_title(document.get("filename", "unknown"))


def assign_product_name_avoiding(document: dict[str, Any], taken_names: set[str]) -> str:
    """Simpler variant for adding one new document to an already-indexed
    corpus, where only the already-assigned product names are available
    (not the other documents' full heading sequences). Same filename-match
    reordering as assign_product_name, for the same reason."""
    filename = document.get("filename", "unknown")
    headings = _reorder_headings_by_filename_match(_document_headings(document), filename)
    for heading in headings:
        if heading not in taken_names:
            return _clean_ordering_code_heading(heading)
    return _document_title(filename)


def _assign_product_names(documents: list[dict[str, Any]]) -> dict[str, str]:
    """Collision-aware product-name assignment across a batch of documents
    being (re)indexed together. An admin correction (see
    app.product_name_overrides) wins over the auto-derived name for any
    document it covers."""
    from app.product_name_overrides import load_overrides as load_product_name_overrides

    overrides = load_product_name_overrides()
    names: dict[str, str] = {}
    for i, document in enumerate(documents):
        filename = document.get("filename", "unknown")
        if filename in overrides:
            names[filename] = overrides[filename]
            continue
        others = tuple(documents[:i] + documents[i + 1 :])
        names[filename] = assign_product_name(document, others)
    return names


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
    product_name: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunks one loaded document into Chunk objects, respecting section
    and table boundaries. product_name is used as the chunk-text prefix
    and stored per chunk; defaults to assign_product_name(document) when
    not given (e.g. when chunking a single document in isolation)."""
    boundary_units = _split_into_boundary_units(document)
    if not boundary_units:
        return []

    document_name = document.get("filename", "unknown")
    if product_name is None:
        product_name = assign_product_name(document)

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

    total_chunks = len(merged_chunks)
    chunks: list[Chunk] = []
    for index, unit_group in enumerate(merged_chunks):
        text = f"{product_name}: " + " ".join(word for word, _ in unit_group)
        first_word_meta = unit_group[0][1]
        chunks.append(
            Chunk(
                text=text,
                document_name=document_name,
                product_name=product_name,
                chunk_index=index,
                total_chunks=total_chunks,
                page_number=first_word_meta.get("page_number"),
                slide_number=first_word_meta.get("slide_number"),
                section_heading=first_word_meta.get("section_heading"),
            )
        )

    return chunks


def chunk_approved_claims(document_name: str, bullets: list[str]) -> list[Chunk]:
    """One chunk per bullet -- deliberately bypasses chunk_document's
    general word-count packing (see DEFAULT_CHUNK_SIZE's own comment:
    "how many short facts share a chunk"). That packing is a good default
    for a document's prose, where nearby short facts usually share
    context, but Approved Claims bullets are independent, atomic facts
    spanning unrelated topics (certifications, warranty, pricing, ...):
    packing them together lets one bullet's embedding dominate the shared
    chunk, burying the others from their own queries."""
    total = len(bullets)
    return [
        Chunk(
            text=f"Approved Claims: {bullet}",
            document_name=document_name,
            product_name="Approved Claims",
            chunk_index=i,
            total_chunks=total,
            section_heading="Approved Claims",
        )
        for i, bullet in enumerate(bullets)
    ]


def chunk_documents(
    documents: dict[str, Any] | list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunks a single loader output dict, or a list of them, into one flat
    list of Chunk objects. Product names are assigned collision-aware
    across the whole batch (see assign_product_name)."""
    if isinstance(documents, dict):
        documents = [documents]

    product_names = _assign_product_names(documents)

    all_chunks: list[Chunk] = []
    for document in documents:
        name = product_names[document.get("filename", "unknown")]
        all_chunks.extend(chunk_document(document, name, chunk_size=chunk_size, overlap=overlap))
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
