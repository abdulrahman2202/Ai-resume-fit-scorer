import io
from pathlib import Path
import pytest
import pypdf

from app.config import get_config, get_minimum_resume_chars
from app.services.extractor import (
    DocumentTooShortError,
    ResumeExtractionError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    extract_job_description,
    extract_resume,
    extract_text_from_pdf,
    extract_text_from_txt,
    validate_resume_text,
)
from app.utils.text import clean_text

# Paths
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SAMPLE_TXT_PATH = SAMPLES_DIR / "sample_resume.txt"
SAMPLE_PDF_PATH = SAMPLES_DIR / "sample_resume.pdf"
SAMPLE_JD_PATH = SAMPLES_DIR / "sample_job_description.txt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_valid_text() -> str:
    return (
        "Alice Smith - Senior Software Engineer\n"
        "Email: alice.smith@example.com | Phone: 555-987-6543\n\n"
        "Summary:\n"
        "Experienced Backend Developer with 7+ years of experience in Python, FastAPI, "
        "microservices architecture, Docker, and PostgreSQL. Proven track record of designing "
        "and scaling high-throughput distributed systems.\n\n"
        "Skills:\n"
        "Python, FastAPI, Django, Docker, Kubernetes, PostgreSQL, Redis, PyTest, AWS."
    )


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    """Generate in-memory valid PDF with a single blank page (no text)."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests: Valid TXT extraction
# ---------------------------------------------------------------------------
def test_extract_txt_from_filepath():
    """Verify extracting valid resume from a .txt file path."""
    assert SAMPLE_TXT_PATH.exists(), "Sample resume txt must exist"
    text = extract_resume(SAMPLE_TXT_PATH)
    assert len(text) >= 100
    assert "Jane Doe" in text
    assert "FastAPI" in text


def test_extract_txt_from_bytes(sample_valid_text):
    """Verify extracting valid resume from raw bytes with a .txt filename."""
    data = sample_valid_text.encode("utf-8")
    extracted = extract_resume(data, filename="applicant_resume.txt")
    assert "Alice Smith" in extracted
    assert "PostgreSQL" in extracted
    assert len(extracted) >= 100


def test_extract_txt_from_bytesio(sample_valid_text):
    """Verify extracting valid resume from a BytesIO stream."""
    stream = io.BytesIO(sample_valid_text.encode("utf-8"))
    stream.name = "resume.txt"
    extracted = extract_resume(stream)
    assert "Alice Smith" in extracted


# ---------------------------------------------------------------------------
# Tests: Valid PDF extraction
# ---------------------------------------------------------------------------
def test_extract_pdf_from_filepath():
    """Verify extracting valid resume from a .pdf file path."""
    assert SAMPLE_PDF_PATH.exists(), "Sample resume pdf must exist"
    text = extract_resume(SAMPLE_PDF_PATH)
    assert len(text) >= 100
    assert "Alex Morgan" in text
    assert "Machine Learning" in text


def test_extract_pdf_from_bytes():
    """Verify extracting valid resume from raw PDF bytes."""
    raw_bytes = SAMPLE_PDF_PATH.read_bytes()
    text = extract_resume(raw_bytes, filename="alex_resume.pdf")
    assert len(text) >= 100
    assert "Alex Morgan" in text


# ---------------------------------------------------------------------------
# Tests: Unsupported file types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_filename", [
    "resume.docx",
    "resume.doc",
    "resume.png",
    "resume.jpg",
    "resume.csv",
    "resume.json",
    "resume.exe",
    "resume",
])
def test_unsupported_file_types(invalid_filename):
    """Verify unsupported extensions raise UnsupportedFileTypeError."""
    dummy_bytes = b"Some sample dummy content that is long enough to pass length validation."
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        extract_resume(dummy_bytes, filename=invalid_filename)
    
    assert "Unsupported file format" in str(exc_info.value)
    # Check inheritance
    assert isinstance(exc_info.value, ResumeExtractionError)


# ---------------------------------------------------------------------------
# Tests: Unreadable or empty resumes
# ---------------------------------------------------------------------------
def test_empty_txt_file_raises_error(tmp_path):
    """Verify empty text file raises UnreadableDocumentError."""
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("", encoding="utf-8")

    with pytest.raises(UnreadableDocumentError):
        extract_resume(empty_txt)


def test_empty_bytes_raises_error():
    """Verify 0-byte input raises UnreadableDocumentError."""
    with pytest.raises(UnreadableDocumentError):
        extract_resume(b"", filename="empty.txt")

    with pytest.raises(UnreadableDocumentError):
        extract_resume(b"", filename="empty.pdf")


def test_corrupted_pdf_raises_error():
    """Verify corrupted PDF bytes raise UnreadableDocumentError."""
    corrupt_data = b"NOT_A_REAL_PDF_HEADER_OR_CONTENT"
    with pytest.raises(UnreadableDocumentError):
        extract_resume(corrupt_data, filename="corrupt.pdf")


def test_blank_scanned_pdf_raises_error(blank_pdf_bytes):
    """Verify blank or scanned PDF with no extractable text raises UnreadableDocumentError."""
    with pytest.raises(UnreadableDocumentError) as exc_info:
        extract_resume(blank_pdf_bytes, filename="scanned_resume.pdf")

    assert "no extractable text" in str(exc_info.value)


def test_too_short_resume_raises_error(tmp_path):
    """Verify text shorter than minimum_resume_chars raises DocumentTooShortError."""
    short_file = tmp_path / "short_resume.txt"
    short_file.write_text("Hello world, this is a very short text.", encoding="utf-8")

    with pytest.raises(DocumentTooShortError) as exc_info:
        extract_resume(short_file)

    assert "too short" in str(exc_info.value)
    assert isinstance(exc_info.value, ResumeExtractionError)


def test_nonexistent_file_raises_error():
    """Verify non-existent file path raises UnreadableDocumentError."""
    with pytest.raises(UnreadableDocumentError):
        extract_resume("non_existent_file_path_12345.pdf")


# ---------------------------------------------------------------------------
# Tests: Configuration threshold validation
# ---------------------------------------------------------------------------
def test_config_minimum_resume_chars():
    """Verify minimum_resume_chars is correctly loaded from config.yaml."""
    min_chars = get_minimum_resume_chars()
    assert min_chars == 100

    cfg = get_config()
    assert "thresholds" in cfg
    assert cfg["thresholds"]["minimum_resume_chars"] == 100


def test_validate_resume_text_custom_threshold():
    """Verify validate_resume_text respects custom thresholds."""
    text = "Valid text with exactly 40 chars............"
    # Threshold 30 -> valid
    assert validate_resume_text(text, min_chars=30) == text
    # Threshold 100 -> raises error
    with pytest.raises(DocumentTooShortError):
        validate_resume_text(text, min_chars=100)


# ---------------------------------------------------------------------------
# Tests: Job description extraction
# ---------------------------------------------------------------------------
def test_extract_job_description_from_string():
    """Verify extracting job description from direct string."""
    jd_str = "We are seeking a Senior Python Engineer with experience in FastAPI and machine learning pipelines."
    extracted = extract_job_description(jd_str)
    assert "Senior Python Engineer" in extracted


def test_extract_job_description_from_file():
    """Verify extracting job description from text file."""
    assert SAMPLE_JD_PATH.exists()
    extracted = extract_job_description(SAMPLE_JD_PATH)
    assert "CloudScale AI" in extracted
    assert len(extracted) > 100


def test_extract_job_description_too_short():
    """Verify too short job description raises error."""
    with pytest.raises(UnreadableDocumentError):
        extract_job_description("Too short")


# ---------------------------------------------------------------------------
# Tests: Text cleaning utility
# ---------------------------------------------------------------------------
def test_clean_text_utility():
    """Verify clean_text removes null bytes and collapses excessive whitespace."""
    raw = "Hello\x00 World!   \r\n\r\n\r\nThis is a    test.  \n\n\n\nEnd."
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "   " not in cleaned
    assert "Hello World!\n\nThis is a test.\n\nEnd." == cleaned
