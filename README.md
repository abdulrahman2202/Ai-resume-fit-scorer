# Resume Fit Scorer

An applied-AI evaluation engine designed to analyze candidate resumes against job descriptions, extract structured hiring criteria, perform dual-signal semantic and keyword matching, and produce explainable, calibrated fit scores.

---

## Overview

The **Resume Fit Scorer** evaluates candidate suitability against job openings through a transparent, reproducible pipeline:

- **Accepts a Job Description and Resume**: Ingests raw job descriptions alongside candidate resumes in supported formats (`.pdf` and `.txt`).
- **Extracts Structured Requirements via Gemini**: Utilizes the Google Gemini API to analyze the job description and extract categorized hiring criteria (technical skills, experience, education, responsibilities, tools) with associated keywords.
- **Extracts Text from Supported Resume Documents**: Parses, cleans, and validates resume text from PDF and TXT documents.
- **Matches Resume Content**: Evaluates candidate text against each extracted criterion using two independent signals: dense semantic embeddings (`all-MiniLM-L6-v2`) and deterministic keyword matching.
- **Calculates a Deterministic Candidate Score**: Aggregates criterion scores using configured category weights and missing-category renormalization.
- **Provides Criterion-Level Evidence and Reasoning**: Generates factual, reproducible explanations referencing exact resume excerpts.
- **Exposes the Pipeline Through FastAPI**: Provides a clean REST API (`POST /api/v1/score` and `GET /health`) with interactive Swagger documentation.

---

## Key Features

- **PDF/TXT Resume Extraction**: High-fidelity text extraction from `.pdf` and `.txt` files with character validation and whitespace normalization.
- **Job Description Processing**: Automated extraction of structured hiring criteria from raw, unstructured job description text.
- **Gemini-Powered Requirement Extraction**: Extracts explicit hiring criteria using Google Gemini without delegating scoring or decision-making to the LLM.
- **Structured Requirement Categories**: Categorizes criteria into five distinct domains: Technical Skills, Experience, Education, Responsibilities, and Tools.
- **Semantic Matching with all-MiniLM-L6-v2**: Chunk-level dense vector semantic similarity scoring normalized to [0.0, 1.0].
- **Deterministic Keyword Matching**: Case-insensitive, whitespace-tolerant keyword matching using regex word boundaries.
- **Evidence Extraction**: Automatically extracts the strongest matching resume text chunk for each criterion.
- **Deterministic Scoring**: Completely reproducible Python-based scoring engine operating on explicit mathematical formulas.
- **Category-Weighted Scoring**: Configurable category weighting with dynamic missing-category renormalization.
- **Calibration Testing**: Verified empirical calibration demonstrating tight score clustering for similarly qualified resumes and substantial score divergence for mismatched candidates.
- **FastAPI API**: Production-ready asynchronous endpoints with comprehensive HTTP error status code translation.
- **Automated Testing**: Complete test suite of 101 automated tests covering document extraction, Gemini extraction, retries, semantic matching, scoring formulas, calibration, and API endpoints.

---

## Architecture

The complete system operates as a unified, deterministic pipeline:

```text
Resume + Job Description
        ↓
Document/Text Extraction
        ↓
Structured Job Requirement Extraction
        ↓
Semantic + Keyword Matching
        ↓
Criterion-Level Scoring
        ↓
Category Aggregation
        ↓
Overall Resume Fit Score
        ↓
FastAPI JSON Response
```

### Component Responsibilities

1. **Document/Text Extraction (`app/services/extractor.py`)**: Validates file types, decodes plain text, extracts PDF pages via `pypdf`, cleans whitespace, and enforces minimum character thresholds.
2. **Job Requirement Extraction (`app/services/criterion_extractor.py`)**: Prompts Gemini for structured JSON criteria, validates with Pydantic, deduplicates criteria, and handles transient API retries with exponential backoff.
3. **Dual-Signal Matching (`app/services/matcher.py`)**: Splits resume text into overlapping chunks, generates dense vector embeddings via `sentence-transformers`, computes cosine similarity, performs word-boundary keyword matching, and isolates the strongest evidence excerpt.
4. **Deterministic Scoring Engine (`app/services/scorer.py`)**: Applies linear weighting (60% semantic + 40% keyword) to each criterion, aggregates scores by category, renormalizes weights for omitted categories, and generates deterministic reasoning.
5. **REST API Interface (`app/main.py`)**: Orchestrates the pipeline, validates multipart form inputs, and maps domain exceptions to appropriate HTTP status codes.

> **Core Architectural Principle: Zero Scoring by LLMs**  
> Gemini is used **ONLY** for extracting structured requirements from the job description. Gemini is **NOT** used for candidate scoring, candidate ranking, final evaluation, score calculation, or criterion reasoning. Matching, scoring, and reasoning are 100% deterministic Python operations.

---

## Technology Stack

| Technology | Category / Purpose | Details |
|---|---|---|
| **Python** | Core Language | Python 3.10+ (tested on Python 3.12) |
| **FastAPI** | Web Framework | Asynchronous REST API, dependency injection, and OpenAPI schemas |
| **Uvicorn** | ASGI Web Server | High-performance ASGI server for hosting FastAPI |
| **Google GenAI SDK** | Requirement Extraction | `google-genai` SDK interfacing with Gemini for JSON criterion extraction |
| **Sentence Transformers** | Semantic Embeddings | Dense text embeddings using `all-MiniLM-L6-v2` |
| **Pydantic** | Schema Validation | Data validation and serialization for criteria, matches, and API responses |
| **PyPDF** | Document Parsing | Multi-page text extraction from PDF documents |
| **NumPy** | Numerical Operations | Vector dot products, norms, and cosine similarity calculations |
| **PyYAML** | Configuration | Parsing system configuration from `config.yaml` |
| **python-dotenv** | Environment Configuration | Loading environment variables and credentials from `.env` |
| **pytest** | Automated Testing | Comprehensive test framework for unit, integration, and calibration tests |

*(Note: Docker is not currently configured or used in this repository.)*

---

## Document Processing

The document ingestion service (`app/services/extractor.py`) provides robust text extraction and validation:

- **Supported Formats**: Strictly `.pdf` and `.txt` documents. Unsupported formats (such as `.docx`, `.png`, or `.exe`) are rejected with `UnsupportedFileTypeError`.
- **Text Extraction**:
  - **PDF Documents**: Extracted page by page using `pypdf.PdfReader` with stream-based and file-based support.
  - **TXT Documents**: Decoded using UTF-8 with fallback decoding to prevent encoding crashes.
- **Text Cleaning**: The `clean_text` utility strips non-printable control characters, normalizes Unicode whitespace, and collapses repeated whitespace while maintaining readable paragraph structures.
- **Minimum Resume Text Validation**: Enforces a minimum length threshold (`minimum_resume_chars: 100` in `config.yaml`). Uploads containing fewer than 100 characters raise `DocumentTooShortError`.
- **Extraction Error Handling**:
  - Empty files (0 bytes) raise `ResumeExtractionError`.
  - Corrupted, encrypted, or image-only scanned PDFs without an extractable text layer raise `UnreadableDocumentError` (mapped to HTTP 422).

---

## Job Requirement Extraction

The requirement extraction service (`app/services/criterion_extractor.py`) uses Google Gemini to translate raw job descriptions into structured hiring criteria.

### Supported Requirement Categories

| Category | Description |
|---|---|
| Technical Skills | Programming languages, software architectures, frameworks, engineering principles |
| Experience | Required years of experience, seniority levels, leadership background |
| Education | Degrees, academic majors, required licenses, or academic credentials |
| Responsibilities | Key day-to-day duties, project management, operational deliverables |
| Tools | Specific software applications, cloud platforms, databases, developer tooling |

### Validation & Reliability

- **JSON and Pydantic Validation**: The model is instructed to output JSON conforming strictly to the schema. Output is parsed, stripped of markdown wrappers, validated against `CriterionExtractionResponse`, deduplicated, and keyword-normalized.
- **Transient Error Retry with Exponential Backoff**:
  - Transient HTTP errors (`429`, `500`, `502`, `503`, `504`), connection drops, and read/connect timeouts are automatically retried.
  - Retries follow exponential backoff: Attempt 1 (~1s), Attempt 2 (~2s), Attempt 3 (~4s), capped at `max_retry_delay_seconds: 8`.
  - Permanent errors (`400 Bad Request`, `401/403 Auth Errors`, `404 Not Found`, schema validation failures) fail immediately without retrying.
- **No Tools or Automatic Function Calling (AFC)**: Requests explicitly set `tools=None` and `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`. This ensures clean, direct text generation without AFC execution loops or warning messages.

---

## Resume Matching

The matching engine (`app/services/matcher.py`) evaluates candidate resumes against extracted criteria using two independent signals:

```text
Extracted Criterion
  ├── Semantic Embeddings (all-MiniLM-L6-v2) ──► Semantic Score [0.0, 1.0]
  └── Deterministic Regex Keywords ────────────► Keyword Score [0.0, 1.0]
```

### Semantic Similarity
- **Paragraph-Aware Chunking**: Resumes are segmented into overlapping text blocks (`chunk_size: 500`, `chunk_overlap: 50` characters) preserving paragraph boundaries.
- **Embedding Generation**: Uses `all-MiniLM-L6-v2` via `sentence-transformers`, loaded once in memory by `EmbeddingModelManager`.
- **Cosine Similarity**: Vector cosine similarity is calculated between the criterion embedding (description + keywords) and each resume chunk embedding.
- **Score Normalization**: Clamps raw cosine similarity to [0.0, 1.0] to eliminate negative noise without distorting positive alignments:

$$
\mathrm{normalized\_score} = \max(0.0, \min(1.0, \mathrm{cosine\_similarity}))
$$

### Keyword Evidence
- **Deterministic Regex Matching**: Evaluates criterion keywords against resume text using case-insensitive, whitespace-tolerant regular expressions with `\b` word boundaries.
- **Keyword Score**: Calculated as the exact fraction of keywords found:

$$
\text{Keyword Score} =
\frac{\text{Matched Keywords}}
{\text{Total Criterion Keywords}}
$$

### Evidence Extraction
For every criterion, the engine records:
- `matched_keywords`: List of keywords identified in the resume.
- `unmatched_keywords`: List of keywords missing from the resume.
- `evidence`: The specific resume text chunk that produced the highest semantic similarity score.

---

## Scoring

All scoring is strictly deterministic and calculated in Python.

### 1. Criterion Scoring Formula

For each individual criterion, the score combines semantic similarity and keyword evidence according to configured weights (`config.yaml`):

$$
\mathrm{criterion\_score} = \left( 0.60 \times \mathrm{semantic\_score} + 0.40 \times \mathrm{keyword\_score} \right) \times 100.0
$$

The resulting criterion score is bounded in `[0.0, 100.0]`.

### 2. Category Weights

Criteria are grouped by category and weighted according to configured values:

| Category | Weight |
|---|---:|
| Technical Skills | 0.35 |
| Experience | 0.25 |
| Education | 0.10 |
| Responsibilities | 0.20 |
| Tools | 0.10 |

Within each category present in the job description, the category score is the arithmetic mean of its criteria:

$$
\mathrm{category\_score}(c) = \frac{1}{|K_c|} \sum_{i \in K_c} \mathrm{criterion\_score}(i)
$$

### 3. Missing-Category Renormalization

If a job description lacks criteria for one or more categories (e.g., no explicit education or tool requirements), the weights of only the **present** categories are renormalized proportionally:

$$
w'_c = \frac{w_c}{\sum_{k \in \mathrm{present}} w_k}
$$

The overall score is computed as:

$$
\mathrm{overall\_score} = \sum_{c \in \mathrm{present}} w'_c \times \mathrm{category\_score}(c)
$$

This ensures missing categories never penalize candidates or distort the `0.0` to `100.0` scale.

### 4. Deterministic Reasoning

For each criterion, the engine produces an explainable, fact-based summary referencing:
- **Semantic Tier**: Strong (>= 0.75), Moderate (>= 0.50), or Weak (< 0.50).
- **Keyword Metrics**: Count and list of matched vs. missing keywords.
- **Resume Excerpt**: Direct quote of the strongest matching evidence block.

---

## Calibration

The purpose of calibration testing is to verify scoring stability: two candidates with genuinely similar qualifications must not produce an implausibly large score divergence (such as a 40-point gap), while a candidate with mismatched qualifications must score substantially lower.

### Calibration Design
- **Fixed Job Description**: Senior Backend AI Engineer (`samples/sample_job_description.txt`).
- **Fixed Extracted Criteria**: 6 representative criteria across all categories to eliminate LLM extraction variability.
- **Three Realistic Resumes** (`samples/calibration/`):
  - **Resume A (Alex Morgan)**: Senior Backend AI Engineer (6 years experience, Python, FastAPI, PostgreSQL, Redis, Docker, AWS, BS CS).
  - **Resume B (Jordan Lee)**: Lead Backend & AI Platform Engineer (6.5 years experience, Python, FastAPI, PostgreSQL, Redis, Docker, GCP, BS Computer Engineering). Similar qualifications with different phrasing and structure.
  - **Resume C (Taylor Brooks)**: Junior Frontend & Graphic Designer (1.5 years experience, React, HTML/CSS, Figma, BA Graphic Design). Materially different profile.

### Actual Measured Results

These scores were calculated directly by the deterministic scoring engine (no hardcoded values or artificial adjustments):

| Candidate | Score |
|---|---:|
| **Resume A — Alex Morgan** | 66.6 |
| **Resume B — Jordan Lee** | 59.9 |
| **Resume C — Taylor Brooks** | 14.2 |

- **Observed Similar Candidate Gap (|A - B|)**: **6.7 points**
- **Configured Maximum Similar-Resume Gap**: **15.0 points** (Passed: 6.7 <= 15.0)
- **Score Difference vs. Mismatched Profile (A vs. C)**: **+52.4 points** (66.6 - 14.2)
- **Score Difference vs. Mismatched Profile (B vs. C)**: **+45.7 points** (59.9 - 14.2)

To run the calibration verification test:

```powershell
python -m pytest tests/test_calibration.py -v -s
```

---

## API

### Start the Server

```powershell
python -m uvicorn app.main:app --reload
```

The service will start on `http://127.0.0.1:8000`.

### Swagger UI Documentation

Interactive OpenAPI documentation is available at:
```text
http://127.0.0.1:8000/docs
```

### Health Check

```http
GET /health
```

**Response (200 OK)**:
```json
{
  "status": "healthy"
}
```

### Resume Scoring

```http
POST /api/v1/score
```

**Request Parameters** (`multipart/form-data`):
- `job_description` (string, required): Plain text job description.
- `resume_file` (file, required): Candidate resume file (`.pdf` or `.txt`).

**Example Request (`curl`)**:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/score" \
  -F "job_description=Senior Python Engineer needed with 5+ years experience in FastAPI, Docker, and PostgreSQL." \
  -F "resume_file=@samples/sample_resume.pdf"
```

**Response Schema (`ScoringResponse`)**:
- `overall_score` (float, 0.0–100.0): Overall candidate match score.
- `criteria` (list): Detailed evaluation for each extracted criterion:
  - `name` (string): Title of the hiring requirement.
  - `category` (string): Requirement category (`technical_skills`, `experience`, `education`, `responsibilities`, or `tools`).
  - `score` (float, 0.0–100.0): Combined criterion score.
  - `semantic_score` (float, 0.0–1.0): Normalized semantic similarity score.
  - `keyword_score` (float, 0.0–1.0): Fraction of criterion keywords found.
  - `matched_keywords` (list of strings): Keywords identified in the resume text.
  - `evidence` (string): Excerpt from the resume serving as strongest evidence.
  - `reason` (string): Deterministic explanation of the score.
- `category_scores` (dict): Aggregated score for each present category (e.g. `{"technical_skills": 85.0, "experience": 78.0, ...}`).

**Example Response**:
```json
{
  "overall_score": 78.4,
  "criteria": [
    {
      "name": "Python & FastAPI Microservices",
      "category": "technical_skills",
      "score": 92.8,
      "semantic_score": 0.88,
      "keyword_score": 1.0,
      "matched_keywords": ["Python", "FastAPI", "Microservices"],
      "evidence": "Senior Software Engineer with 6 years experience building Python and FastAPI microservices...",
      "reason": "Strong semantic alignment (0.88); all 3 of 3 required keywords found (Python, FastAPI, Microservices). Resume evidence: \"Senior Software Engineer with 6 years experience...\"."
    }
  ],
  "category_scores": {
    "technical_skills": 92.8,
    "experience": 80.0,
    "education": 75.0,
    "responsibilities": 70.0,
    "tools": 74.2
  }
}
```

---

## Example Workflow

A complete evaluation follows these steps:

1. **Start the FastAPI server**:
   ```powershell
   python -m uvicorn app.main:app --reload
   ```
2. **Open Swagger UI** in your browser at `http://127.0.0.1:8000/docs`.
3. **Open the `POST /api/v1/score` endpoint** and click **Try it out**.
4. **Paste a job description** into the `job_description` form field.
5. **Upload a PDF or TXT resume** using the `resume_file` file picker.
6. **Execute the request** to trigger the end-to-end evaluation.
7. **Review the results**: Inspect `overall_score`, individual criterion scores, matched/unmatched keywords, resume evidence excerpts, and aggregated category scores.

---

## Testing

The project maintains a comprehensive automated test suite with mocked external services (no live Gemini calls during unit tests).

Run all tests:

```powershell
python -m pytest -v
```

Run the calibration test suite specifically:

```powershell
python -m pytest tests/test_calibration.py -v -s
```

### Verified Test Results

```text
====================== 101 passed, 4 warnings in 39.21s =======================
```

**Test Suite Breakdown**:
- `tests/test_parser.py`: **25 passed** (PDF extraction, TXT decoding, unreadable PDFs, short resumes, clean text utilities)
- `tests/test_criterion_extractor.py`: **29 passed** (Gemini extraction, JSON parsing, Pydantic validation, exponential backoff retries, AFC disabled)
- `tests/test_matcher.py`: **20 passed** (all-MiniLM-L6-v2 embeddings, cosine normalization, keyword regex, paragraph chunking)
- `tests/test_scorer.py`: **14 passed** (scoring formulas, category weights, missing-category renormalization, reasoning generation)
- `tests/test_calibration.py`: **1 passed** (empirical calibration across three realistic resumes)
- `tests/test_api.py`: **12 passed** (FastAPI endpoint integration, validation errors, error code translations)

**Total**: **101 tests passed**.

---

## Error Handling

The service includes domain-specific exception hierarchies translated to explicit HTTP status codes:

| Error Scenario | Domain Exception | HTTP Status | Detail |
|---|---|---|---|
| Unsupported File Format | `UnsupportedFileTypeError` | `400 Bad Request` | File is not `.pdf` or `.txt` (e.g. `.docx`, `.png`) |
| Empty Upload | `ResumeExtractionError` | `400 Bad Request` | Uploaded resume contains 0 bytes |
| Empty Job Description | `InvalidJobDescriptionError` | `400 Bad Request` | Job description is empty or whitespace-only |
| Resume Too Short | `DocumentTooShortError` | `400 Bad Request` | Resume contains fewer characters than `minimum_resume_chars` (100) |
| Unreadable / Blank Scanned PDF | `UnreadableDocumentError` | `422 Unprocessable Entity` | PDF contains no extractable text layer (e.g. scanned image) |
| No Criteria Extracted | `CriterionExtractionError` | `422 Unprocessable Entity` | Gemini was unable to identify any criteria in the job description |
| Gemini API Timeout | `GeminiTimeoutError` | `504 Gateway Timeout` | Gemini API request timed out after all retries |
| Gemini Transient / Network Error | `GeminiAPIError` | `502 Bad Gateway` | Communication failure with Gemini API after retry exhaustion |
| Malformed Gemini Response | `InvalidExtractionJSONError` | `502 Bad Gateway` | Gemini output could not be parsed as valid JSON |
| Schema Validation Error | `ExtractionSchemaValidationError` | `502 Bad Gateway` | Gemini JSON output violates the Pydantic criterion schema |
| Missing API Key / Config Error | `ConfigurationError` | `500 Internal Server Error` | `GEMINI_API_KEY` is not set or configuration is missing |
| Embedding Inference Failure | `EmbeddingModelError` | `500 Internal Server Error` | Sentence-transformers model failed during encoding |
| Invalid Weight Configuration | `InvalidWeightConfigurationError` | `500 Internal Server Error` | Scoring or category weights in `config.yaml` do not sum to 1.0 |

---

## Configuration

### `config.yaml`
Central system parameters controlling scoring weights, thresholds, model settings, and retry policies:

```yaml
scoring:
  semantic_weight: 0.60
  keyword_weight: 0.40

category_weights:
  technical_skills: 0.35
  experience: 0.25
  education: 0.10
  responsibilities: 0.20
  tools: 0.10

thresholds:
  strong_match: 0.75
  partial_match: 0.50
  minimum_resume_chars: 100

matching:
  chunk_size: 500
  chunk_overlap: 50
  embedding_model: "all-MiniLM-L6-v2"

calibration:
  max_similar_resume_gap: 15.0

gemini:
  max_retries: 3
  initial_retry_delay_seconds: 1
  max_retry_delay_seconds: 8
```

### `.env`
Configures credentials and model selection:

```bash
# Gemini API Key (Required for requirement extraction)
GEMINI_API_KEY=your_gemini_api_key_here

# Configured Gemini model (default: gemini-3.5-flash)
GEMINI_MODEL=gemini-3.5-flash
```

> **Security Note**: Never commit your `.env` file or API keys to version control. An example template is provided in `.env.example`.

---

## Project Structure

```text
resume-fit-scorer/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Configuration loader and validator (config.yaml & .env)
│   ├── main.py                   # FastAPI application & REST endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic schemas (Criterion, ScoringResponse, etc.)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extractor.py          # Document extraction & text validation (.pdf, .txt)
│   │   ├── criterion_extractor.py # Gemini requirement extraction with retry & backoff
│   │   ├── matcher.py            # Sentence-transformers semantic & keyword matching
│   │   └── scorer.py             # Deterministic scoring, category weights, reasoning
│   └── utils/
│       ├── __init__.py
│       └── text.py               # Text cleaning and normalization utilities
├── samples/
│   ├── sample_job_description.txt # Sample job description
│   ├── sample_resume.pdf         # Sample candidate resume (PDF)
│   ├── sample_resume.txt         # Sample candidate resume (TXT)
│   └── calibration/
│       ├── calibration_resume_a.txt # Senior Backend Engineer A (Alex Morgan)
│       ├── calibration_resume_b.txt # Senior Backend Engineer B (Jordan Lee)
│       └── calibration_resume_c.txt # Junior Frontend Designer C (Taylor Brooks)
├── tests/
│   ├── test_api.py               # FastAPI endpoint integration & error tests (12 tests)
│   ├── test_calibration.py       # Calibration test on realistic resumes (1 test)
│   ├── test_criterion_extractor.py # Gemini extraction, validation, & retry tests (29 tests)
│   ├── test_matcher.py           # Embedding & keyword matching tests (20 tests)
│   ├── test_parser.py            # Document parsing & validation tests (25 tests)
│   └── test_scorer.py            # Scoring formulas, weights, & reasoning tests (14 tests)
├── .env.example                  # Environment variable template
├── config.yaml                   # Central configuration (weights, thresholds, retries)
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```
