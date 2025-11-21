from pathlib import Path
from typing import Optional
from PyPDF2 import PdfReader
from .utils import clean_whitespace


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract selectable text from a PDF using PyPDF2 (no OCR).

    Returns cleaned text. Assumes digitally-created PDF (no OCR).
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(p))
    parts = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        parts.append(text)

    raw = "\n\n".join(parts)
    return clean_whitespace(raw)


def save_text(text: str, out_path: str) -> None:
    Path(out_path).write_text(text, encoding="utf-8")
