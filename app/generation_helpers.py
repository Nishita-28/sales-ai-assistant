"""Shared helpers for building LLM context blocks and parsing structured
LLM replies -- used by discovery_generator.py and sales_aid_generator.py,
which both retrieve KB excerpts and ask for a "Header:\\n<content>"-style
reply.
"""
from __future__ import annotations

import re
from typing import Any


def build_context_block(matches: list[dict[str, Any]]) -> str:
    """Formats retrieved chunks as a labeled, numbered context block for
    the LLM prompt, including product_name so cross-product comparisons
    are unambiguous."""
    if not matches:
        return "(no approved excerpts were retrieved)"
    blocks = []
    for i, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        document_name = metadata.get("document_name", "unknown")
        product_name = metadata.get("product_name", "")
        label = f"{document_name} (product: {product_name})" if product_name else document_name
        blocks.append(f"[{i}] {label}:\n{match.get('text', '')}")
    return "\n\n".join(blocks)


def dedupe_sources(matches: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """First-seen-order, deduped (document_name, "") pairs for display."""
    sources: list[tuple[str, str]] = []
    seen = set()
    for match in matches:
        name = (match.get("metadata") or {}).get("document_name", "unknown")
        if name not in seen:
            seen.add(name)
            sources.append((name, ""))
    return sources


# Matches a leading "- " bullet or a "1. " / "1) " numbered-list marker.
_LIST_ITEM_RE = re.compile(r"^(?:-|\d+[.)])\s*(.+)$")


def extract_list_items(block: str) -> list[str]:
    """Pulls out bullet/numbered list items, deduped by text (case-
    insensitive) while preserving first-seen order -- a safety net for
    models that restate the same list twice under a second header."""
    items = []
    seen = set()
    for line in block.splitlines():
        match = _LIST_ITEM_RE.match(line.strip())
        if not match:
            continue
        item = match.group(1).strip()
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def extract_lines(block: str) -> list[str]:
    """Returns every non-blank line, stripped, in order -- unlike
    extract_list_items, this doesn't require a bullet/number prefix, so it
    preserves a markdown table's header/separator/data rows intact for
    rendering as one block."""
    return [line.strip() for line in block.splitlines() if line.strip()]


def split_sections(raw_text: str, headers: list[str]) -> dict[str, str]:
    """Splits raw_text into {header_lower: content} for each "Header:" line
    that appears on its own line, matching headers case-insensitively."""
    pattern = re.compile(
        r"^(" + "|".join(re.escape(h) for h in headers) + r"):\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    sections: dict[str, str] = {}
    matches = list(pattern.finditer(raw_text))
    for i, match in enumerate(matches):
        header = match.group(1).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        sections[header] = raw_text[start:end].strip()
    return sections
