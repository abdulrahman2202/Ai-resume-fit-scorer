# Resume Job Fit Scorer

An applied-AI evaluation tool designed to analyze resumes against job descriptions, extract structured evaluation criteria, compute semantic and keyword match scores, and provide actionable feedback.

---

## Current Status: Step 2 – Gemini Job Requirement Extraction

The first two phases of the pipeline are complete:
- **Step 1**: Multi-format document ingestion and validation for resumes and job descriptions.
- **Step 2**: Gemini-powered extraction of structured hiring criteria from job descriptions with strict Pydantic schema validation.

### Architecture & Isolation Mandate

```
Job Description
      ↓
Gemini API (gemini-2.5-flash)
      ↓
Structured Hiring Criteria
      ↓
Pydantic Schema Validation
```

> **Strict Non-Scoring Mandate**: Gemini is solely responsible for extracting hiring criteria from job descriptions. Gemini does **NOT** score resumes, calculate overall scores, rank candidates, or evaluate candidate suitability. All scoring logic is reserved for subsequent deterministic scoring components.

---

### Step 2 Features Implemented

1. **Structured Criteria Extraction (`app/services/criterion_extractor.py`)**:
   - Uses the official Google GenAI Python SDK (`google-genai`).
   - Configured via environment variables (`GEMINI_API_KEY`, `GEMINI_MODEL=gemini-2.5-flash` in `.env`).
   - Temperature pinned to `0.0` for deterministic requirement extraction.
   - Automatically handles JSON response format and strips markdown code fences (` ```json ... ``` `).

2. **Strict Pydantic Schemas (`app/models/schemas.py`)**:
   - **`CriterionCategory`**: Enforces the 5 allowed categories:
     - `technical_skills`: Languages, architectures, core principles, technical domains.
     - `experience`: Years of experience, seniority level, industry background.
     - `education`: Degrees, academic majors, required credentials.
     - `responsibilities`: Core day-to-day duties, deliveries, project management.
     - `tools`: Specific applications, cloud platforms (AWS, GCP, Azure), databases, CI/CD tools.
   - **`Criterion`**:
     - `name`: Requirement title (min length 1, trimmed).
     - `category`: Validated against `CriterionCategory`.
     - `description`: Detailed requirement description (min length 1, trimmed).
     - `keywords`: Cleaned, trimmed, and deduplicated matching keywords.
   - **`CriterionExtractionResponse`**:
     - `criteria`: Required list of `Criterion` objects.
     - Automatic deduplication and keyword merging across duplicate criteria.

3. **Robust Error Handling Hierarchy**:
   - `CriterionExtractionError` (base)
     - `ConfigurationError`: Missing or placeholder `GEMINI_API_KEY`.
     - `InvalidJobDescriptionError`: Empty or blank job description input.
     - `GeminiAPIError`: Network, HTTP, or Gemini API communication failures.
       - `GeminiTimeoutError`: Request timeouts from Gemini or network layers.
     - `EmptyGeminiResponseError`: Empty, null, or candidate-free model responses.
     - `InvalidExtractionJSONError`: Malformed JSON or unexpected non-JSON structures.
     - `ExtractionSchemaValidationError`: Pydantic validation failures (missing required fields, invalid categories).

4. **Zero Live Network Calls in Tests**:
   - Tests mock the Gemini client and SDK interfaces completely.
   - No active API keys or external network connections are needed to run tests.

---

### Step 1 Features Implemented

1. **Multi-Format Document Ingestion**:
   - **PDF Extraction**: Uses `pypdf` to extract text from multi-page PDF documents.
   - **Plain Text (TXT) Extraction**: Supports UTF-8 and Latin-1 encoded text documents.
   - **Job Description Extraction**: Accepts direct string inputs or text/PDF files.

2. **Error Handling & Failure Protection**:
   - Custom exception hierarchy rooted at `ResumeExtractionError`:
     - `UnsupportedFileTypeError`: Enforces supported formats (`.pdf`, `.txt`) and rejects unsupported files.
     - `UnreadableDocumentError`: Protects against corrupted PDFs, zero-byte files, password-protected files, and scanned PDFs.
     - `DocumentTooShortError`: Validates that extracted text satisfies minimum character thresholds.

3. **Configurable Thresholds**:
   - Configured via `config.yaml` using `thresholds.minimum_resume_chars` (default: 100 characters).

4. **Text Cleaning & Normalization**:
   - Strips null bytes and control characters.
   - Normalizes line breaks (`CRLF` / `CR` to `LF`).
   - Normalizes excessive horizontal whitespace while preserving paragraph semantics.

---

## Project Structure

```
app/
├── __init__.py
├── config.py                 # Configuration loader for config.yaml & .env
├── main.py
├── models/
│   ├── __init__.py
│   └── schemas.py            # Pydantic schemas (Criterion, CriterionCategory, etc.)
├── services/
│   ├── __init__.py
│   ├── extractor.py          # Document parsing & text extraction (Step 1)
│   ├── criterion_extractor.py # Gemini JD criteria extraction & validation (Step 2)
│   ├── matcher.py
│   └── scorer.py
└── utils/
    ├── __init__.py
    └── text.py               # Text normalization and cleaning utilities

tests/
├── test_parser.py            # Step 1 unit tests (25 tests)
└── test_criterion_extractor.py # Step 2 unit tests (23 tests)

samples/
├── sample_resume.txt         # Sample software engineer resume in TXT
├── sample_resume.pdf         # Sample ML engineer resume in PDF
└── sample_job_description.txt # Sample job description

config.yaml                   # Application scoring and threshold configuration
requirements.txt              # Python dependencies
.env                          # Gemini API credentials & model settings
```

---

## Running the Unit Tests

Run the complete test suite using `pytest`:

```bash
pytest -v
```

Or using the local virtual environment:

```powershell
.\venv\Scripts\python -m pytest -v
```

All 48 test cases (25 parser tests + 23 criterion extractor tests) will execute and pass without requiring external API calls.
