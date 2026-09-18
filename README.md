# Resume Job Fit Scorer

An applied-AI evaluation tool designed to analyze resumes against job descriptions, extract structured evaluation criteria, compute semantic and keyword match scores, and provide actionable feedback.

---

## Current Status: Step 1 – Document & Job-Description Extraction

The first phase implements robust document extraction and input validation for both candidate resumes and job descriptions.

### Features Implemented

1. **Multi-Format Document Ingestion**:
   - **PDF Extraction**: Uses `pypdf` to extract text from multi-page PDF documents.
   - **Plain Text (TXT) Extraction**: Supports UTF-8 and Latin-1 encoded text documents.
   - **Job Description Extraction**: Accepts direct string inputs or text/PDF files.

2. **Error Handling & Failure Protection**:
   - Custom exception hierarchy rooted at `ResumeExtractionError`:
     - `UnsupportedFileTypeError`: Enforces supported formats (`.pdf`, `.txt`) and rejects unsupported files (e.g., `.docx`, images, executables).
     - `UnreadableDocumentError`: Protects against corrupted PDFs, zero-byte files, password-protected files, and image-only/scanned documents where no extractable text is present (OCR is not supported yet).
     - `DocumentTooShortError`: Validates that extracted text satisfies minimum character thresholds.

3. **Configurable Thresholds**:
   - Configured via `config.yaml` using `thresholds.minimum_resume_chars` (default: 100 characters).

4. **Text Cleaning & Normalization**:
   - Strips null bytes and control characters.
   - Normalizes line breaks (`CRLF` / `CR` to `LF`).
   - Normalizes excessive horizontal whitespace while preserving paragraph semantics.

---

## Project Structure (Step 1)

```
app/
├── __init__.py
├── config.py             # Configuration loader for config.yaml & thresholds
├── main.py
├── models/
│   └── schemas.py
├── services/
│   ├── __init__.py
│   ├── extractor.py      # Core document extraction & validation logic
│   ├── criterion_extractor.py
│   ├── matcher.py
│   └── scorer.py
└── utils/
    ├── __init__.py
    └── text.py           # Text normalization and sanitization utilities

tests/
└── test_parser.py        # Unit test suite for extraction and validation

samples/
├── sample_resume.txt     # Sample software engineer resume in TXT
├── sample_resume.pdf     # Sample ML engineer resume in PDF
└── sample_job_description.txt # Sample job description

config.yaml               # Application scoring and threshold configuration
requirements.txt          # Python dependencies
```

---

## Running the Unit Tests

Run the test suite using `pytest`:

```bash
pytest -v
```

All 25 test cases covering valid TXT/PDF, corrupted/empty files, scanned PDFs, unsupported formats, and threshold validations should pass.
