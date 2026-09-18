import re


def clean_text(text: str) -> str:
    """
    Normalize and clean raw extracted text from documents.

    - Removes null bytes and unprintable control characters.
    - Normalizes line breaks (CRLF, CR -> LF).
    - Collapses excessive horizontal whitespace (tabs, consecutive spaces).
    - Preserves logical paragraph breaks while collapsing 3+ newlines.
    - Strips leading and trailing whitespace.
    """
    if not text:
        return ""

    # Remove null bytes
    cleaned = text.replace("\x00", "")

    # Normalize line breaks
    cleaned = re.sub(r"\r\n|\r", "\n", cleaned)

    # Collapse consecutive horizontal whitespace (spaces and tabs)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Clean leading/trailing spaces on individual lines
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    # Limit consecutive newlines to maximum of 2 (preserving paragraph breaks)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
