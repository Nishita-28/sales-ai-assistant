"""Document format-specific loaders."""
from app.loaders.docx_loader import extract_docx_blocks
from app.loaders.shared import IsBoldFn

__all__ = ["extract_docx_blocks", "IsBoldFn"]
