import re
from typing import List

def clean_whitespace(text: str) -> str:
    if text is None:
        return ""
    # replace multiple spaces, normalize newlines
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    # lightweight sentence splitter using punctuation heuristics
    if not text:
        return []
    # Ensure we keep legal abbreviations simple — do not attempt heavy parsing
    parts = re.split(r'(?<=[.!?;])\s+(?=[A-Z0-9\(])', text)
    return [p.strip() for p in parts if p.strip()]
