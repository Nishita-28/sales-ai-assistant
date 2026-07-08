from pathlib import Path

from docx import Document


def load_approved_document_text() -> str:
    """Read and return text from the approved Word document."""
    project_root = Path(__file__).resolve().parent.parent
    document_path = project_root / "data" / "approved_docs" / "MNST_NC2. Catalogue_PORTaHY H2 LD (Leak Detector Series).docx"

    doc = Document(document_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


if __name__ == "__main__":
    text = load_approved_document_text()
    print(text)
