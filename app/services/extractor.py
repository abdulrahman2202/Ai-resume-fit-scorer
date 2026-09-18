import io
from pathlib import Path
from typing import BinaryIO, Optional, Union

import pypdf
import pypdf.errors

from app.config import get_minimum_resume_chars
from app.utils.text import clean_text

# Supported resume file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class ResumeExtractionError(Exception):
    """Base exception for document extraction errors."""
    pass


class UnsupportedFileTypeError(ResumeExtractionError):
    """Raised when an unsupported file type is provided (only .pdf and .txt supported)."""
    pass


class UnreadableDocumentError(ResumeExtractionError):
    """Raised when a document is empty, corrupt, scanned, or cannot be parsed."""
    pass


class DocumentTooShortError(ResumeExtractionError):
    """Raised when extracted text does not meet the minimum character threshold."""
    pass


def _resolve_stream(
    file_input: Union[str, Path, bytes, BinaryIO]
) -> tuple[io.BytesIO, Optional[str]]:
    """
    Helper to convert various file input types into a seekable BytesIO stream
    and extract an optional filename.
    """
    filename: Optional[str] = None

    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        if not path.is_file():
            raise UnreadableDocumentError(f"File not found or not a valid file: {path}")
        filename = path.name
        content = path.read_bytes()
        stream = io.BytesIO(content)
    elif isinstance(file_input, bytes):
        stream = io.BytesIO(file_input)
    elif hasattr(file_input, "read"):
        # File-like object (BinaryIO, BytesIO, SpooledTemporaryFile)
        if hasattr(file_input, "name") and isinstance(file_input.name, str):
            filename = Path(file_input.name).name
        content = file_input.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        stream = io.BytesIO(content)
        # Reset original pointer if seekable
        if hasattr(file_input, "seek") and callable(file_input.seek):
            try:
                file_input.seek(0)
            except Exception:
                pass
    else:
        raise UnreadableDocumentError(f"Unsupported input type: {type(file_input)}")

    return stream, filename


def extract_text_from_pdf(source: Union[str, Path, bytes, BinaryIO]) -> str:
    """
    Extract text from a PDF file using pypdf.

    Handles:
    - Empty or 0-byte PDF streams
    - Malformed or corrupted PDFs
    - Scanned/image-only PDFs where text extraction yields no text
    - Normalizes and cleans the extracted text

    Raises:
        UnreadableDocumentError: If PDF cannot be read or contains no extractable text.
    """
    stream, _ = _resolve_stream(source)
    stream_size = stream.getbuffer().nbytes
    if stream_size == 0:
        raise UnreadableDocumentError("PDF file is empty (0 bytes).")

    try:
        reader = pypdf.PdfReader(stream)
    except pypdf.errors.PdfReadError as e:
        raise UnreadableDocumentError(f"Corrupted or invalid PDF document: {e}") from e
    except Exception as e:
        raise UnreadableDocumentError(f"Failed to read PDF document: {e}") from e

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
            if decrypted == 0:
                raise UnreadableDocumentError("PDF is encrypted with a password and cannot be read.")
        except Exception as e:
            raise UnreadableDocumentError(f"Encrypted PDF could not be decrypted: {e}") from e

    if len(reader.pages) == 0:
        raise UnreadableDocumentError("PDF document contains 0 pages.")

    page_texts: list[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                page_texts.append(page_text)
        except Exception as e:
            # Continue trying other pages if an individual page has issues
            continue

    raw_text = "\n\n".join(page_texts)
    cleaned = clean_text(raw_text)

    if not cleaned:
        raise UnreadableDocumentError(
            "PDF contains no extractable text. The file may be empty, image-based, or scanned "
            "(OCR is currently not supported)."
        )

    return cleaned


def extract_text_from_txt(source: Union[str, Path, bytes, BinaryIO]) -> str:
    """
    Extract text from a plain text file.

    Handles:
    - UTF-8 and Latin-1 encodings
    - Empty files
    - Normalizes and cleans the text

    Raises:
        UnreadableDocumentError: If file is empty or cannot be decoded.
    """
    stream, _ = _resolve_stream(source)
    raw_bytes = stream.getvalue()

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise UnreadableDocumentError("Text file is empty.")

    text: str = ""
    # Try decoding with UTF-8 first, fallback to Latin-1
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if not text:
        text = raw_bytes.decode("utf-8", errors="replace")

    cleaned = clean_text(text)
    if not cleaned:
        raise UnreadableDocumentError("Text file contains only whitespace or unreadable content.")

    return cleaned


def validate_resume_text(text: str, min_chars: Optional[int] = None) -> str:
    """
    Validate that extracted resume text meets the minimum character length requirement
    defined in config.yaml.

    Raises:
        DocumentTooShortError: If text is shorter than the configured threshold.
    """
    if min_chars is None:
        min_chars = get_minimum_resume_chars()

    char_count = len(text)
    if char_count < min_chars:
        raise DocumentTooShortError(
            f"Extracted resume text is too short ({char_count} chars). "
            f"Minimum required is {min_chars} characters."
        )

    return text


def extract_resume(
    file_input: Union[str, Path, bytes, BinaryIO],
    filename: Optional[str] = None,
    min_chars: Optional[int] = None,
) -> str:
    """
    Extract and validate text from a resume document (.pdf or .txt).

    Args:
        file_input: File path, bytes, or file-like object.
        filename: Optional filename to determine format if file_input is bytes or stream.
        min_chars: Optional character minimum threshold. Defaults to config.yaml value.

    Returns:
        Cleaned, extracted resume text.

    Raises:
        UnsupportedFileTypeError: If file extension is not .pdf or .txt.
        UnreadableDocumentError: If document is empty, corrupt, or has no extractable text.
        DocumentTooShortError: If extracted text is shorter than minimum threshold.
    """
    stream, detected_filename = _resolve_stream(file_input)
    actual_filename = filename or detected_filename

    # Determine extension
    ext = ""
    if actual_filename:
        ext = Path(actual_filename).suffix.lower()

    # Magic byte check if filename has no extension
    if not ext:
        header = stream.getvalue()[:8]
        if header.startswith(b"%PDF"):
            ext = ".pdf"

    if ext not in SUPPORTED_EXTENSIONS:
        display_ext = ext if ext else "unknown"
        raise UnsupportedFileTypeError(
            f"Unsupported file format '{display_ext}'. Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        extracted = extract_text_from_pdf(stream)
    elif ext == ".txt":
        extracted = extract_text_from_txt(stream)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file format: {ext}")

    validated = validate_resume_text(extracted, min_chars=min_chars)
    return validated


def extract_job_description(
    source: Union[str, Path, bytes, BinaryIO],
    filename: Optional[str] = None,
    min_chars: int = 20,
) -> str:
    """
    Extract and validate text for a job description.

    Accepts:
    - Raw text string
    - File path (.txt or .pdf)
    - Raw bytes or stream with optional filename

    Raises:
        UnreadableDocumentError: If job description is empty or too short.
        UnsupportedFileTypeError: If file format is unsupported.
    """
    # If source is a string and not an existing file path, treat as raw text
    if isinstance(source, str) and not Path(source).is_file():
        cleaned = clean_text(source)
    else:
        stream, detected_filename = _resolve_stream(source)
        actual_name = filename or detected_filename
        ext = Path(actual_name).suffix.lower() if actual_name else ""

        if not ext and stream.getvalue()[:4].startswith(b"%PDF"):
            ext = ".pdf"
        elif not ext:
            ext = ".txt"

        if ext == ".pdf":
            cleaned = extract_text_from_pdf(stream)
        elif ext == ".txt":
            cleaned = extract_text_from_txt(stream)
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported job description file format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    if len(cleaned) < min_chars:
        raise UnreadableDocumentError(
            f"Job description is too short ({len(cleaned)} chars). Minimum required is {min_chars} characters."
        )

    return cleaned
