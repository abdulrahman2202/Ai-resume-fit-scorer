# Resume Job Fit Scorer

An applied-AI evaluation tool designed to analyze resumes against job descriptions, extract structured evaluation criteria, compute semantic and keyword match scores, and provide actionable, calibrated candidate feedback.

---

## Current Status: Step 4 – Deterministic Scoring, Calibration, and FastAPI API

All four core development steps are fully implemented and verified:
- **Step 1**: Multi-format document ingestion and validation (.pdf, .txt).
- **Step 2**: Gemini-powered extraction of structured hiring criteria with Pydantic validation.
- **Step 3**: Semantic similarity (`all-MiniLM-L6-v2`) and deterministic keyword matching.
- **Step 4**: Deterministic scoring engine, missing-category renormalization, calibration suite, and production-ready FastAPI endpoints.

---

## Pipeline Architecture

```
Job Description Text / File
          ↓
[Step 2] Gemini Requirement Extraction (gemini-2.5-flash)
          ↓ (Strictly extracts criteria, never assigns scores)
Structured Hiring Criteria (Pydantic validated)
          │
          ├─────────────────────────────────────────────────┐
          │                                                 │
Candidate Resume (.pdf / .txt)                              │
          ↓                                                 │
[Step 1] Document Ingestion & Validation                    │
          ↓                                                 │
[Step 3] Dual-Signal Matching Engine                        │
          ├─► Paragraph Chunking (500-char blocks)         │
          ├─► Semantic Similarity (all-MiniLM-L6-v2)       │
          └─► Keyword Evidence (Deterministic Regex)        │
          │                                                 │
          ▼                                                 ▼
[Step 4] Deterministic Scoring Engine
          ├─► Criterion Scoring: 0.60 Semantic + 0.40 Keyword (0–100 scale)
          ├─► Category Aggregation & Renormalization
          ├─► Factual, Deterministic Reasoning Generation
          └─► Overall Candidate Score (0.0 to 100.0)
          │
          ▼
FastAPI POST /api/v1/score
```

> **Zero Scoring by LLMs Mandate**: Gemini's sole responsibility is extracting requirements from the job description. All scoring, weighting, and reasoning calculations are 100% deterministic, reproducible, and executed locally in Python.

---

## Scoring Methodology & Formulas

### 1. Criterion-Level Score
For each individual criterion, the score combines semantic similarity and keyword evidence according to configured weights (`config.yaml`):

$$\text{criterion\_score} = \left( w_{\text{semantic}} \times s_{\text{semantic}} + w_{\text{keyword}} \times s_{\text{keyword}} \right) \times 100.0$$

- **Default Weights**: $w_{\text{semantic}} = 0.60$, $w_{\text{keyword}} = 0.40$ (sum to 1.0).
- **Semantic Score ($s_{\text{semantic}}$)**: Bounded in $[0.0, 1.0]$. Clamps negative cosine noise to 0.0 without affine distortion.
- **Keyword Score ($s_{\text{keyword}}$)**: Exact fraction of criterion keywords found in the resume $[0.0, 1.0]$.
- **Scale**: Scaled to $[0.0, 100.0]$.

### 2. Category Weighting & Missing-Category Renormalization
Each criterion belongs to one of five categories with configured base weights:
- `technical_skills`: 0.35
- `experience`: 0.25
- `education`: 0.10
- `responsibilities`: 0.20
- `tools`: 0.10

Within each category present in the job description, the category score is the arithmetic mean of its criteria:
$$\text{category\_score}(c) = \frac{1}{|K_c|} \sum_{i \in K_c} \text{criterion\_score}(i)$$

**Renormalization Rule**:
If a job description omits certain categories (e.g., no explicit education or tool requirements), the weights of only the *present* categories are renormalized proportionally:

$$w'_c = \frac{w_c}{\sum_{k \in \text{present}} w_k}$$

The overall score is then computed as:
$$\text{overall\_score} = \sum_{c \in \text{present}} w'_c \times \text{category\_score}(c) \in [0.0, 100.0]$$

This ensures that missing categories do not artificially depress a candidate's score.

### 3. Deterministic Reasoning Generation
For every criterion, the engine produces an explainable, fact-based explanation derived strictly from matching data:
- **Semantic Tier**: Strong ($\ge 0.75$), Moderate ($\ge 0.50$), Partial ($\ge 0.25$), or Minimal ($< 0.25$).
- **Keyword Evidence**: Specific count of matched vs. total keywords and explicit list of missing keywords.
- **Evidence Excerpt**: The exact text block from the candidate's resume that yielded the highest semantic match.

---

## Calibration Suite & Empirical Results

### Assessment Requirement
> *"Two similar resumes should not score 40 points apart — demonstrate calibration on three samples."*

### Calibration Design
To eliminate Gemini extraction variance from the calibration measurement:
1. **ONE Fixed Job Description**: Senior Backend AI Engineer (`samples/sample_job_description.txt`).
2. **ONE Fixed Extracted Criteria Set**: 6 representative criteria across all categories.
3. **Three Realistic Resumes** (`samples/calibration/`):
   - **Resume A (`calibration_resume_a.txt`)**: Alex Morgan — Senior Backend AI Engineer (6 years, Python, FastAPI, PostgreSQL, Redis, Docker, AWS, BS CS).
   - **Resume B (`calibration_resume_b.txt`)**: Jordan Lee — Lead Backend & AI Platform Engineer (6.5 years, Python, FastAPI, PostgreSQL, Redis, Docker, GCP, BS Computer Engineering). Genuinely similar qualifications, differently structured and phrased.
   - **Resume C (`calibration_resume_c.txt`)**: Taylor Brooks — Junior Frontend & Graphic Designer (1.5 years, React, HTML/CSS, Figma, BA Graphic Design). Materially different profile.

### Calibration Metric & Acceptance Threshold
- **Metric**: $\text{similar\_resume\_score\_gap} = | \text{score}_A - \text{score}_B |$
- **Configured Maximum Acceptable A/B Gap**: **15.0 points** (configured in `config.yaml` as an acceptance threshold, well below the 40-point anomaly limit).

### Actual Observed Results
Measured strictly by the deterministic pipeline (no post-processing or artificial score adjustments):

| Candidate | Profile Summary | Actual Overall Score |
|---|---|---|
| **Resume A** (Alex Morgan) | Senior Backend AI Engineer | **66.6 / 100** |
| **Resume B** (Jordan Lee) | Lead Backend & AI Platform Engineer | **59.9 / 100** |
| **Resume C** (Taylor Brooks) | Junior Frontend Designer | **14.2 / 100** |

- **Configured Maximum Acceptable A/B Gap**: `15.0 points`
- **Observed A/B Gap**: **6.7 points** ($|66.6 - 59.9| = 6.7 \le 15.0$)
- **Material Difference vs. Junior Frontend (A vs. C)**: **+52.4 points** ($66.6 - 14.2$)
- **Material Difference vs. Junior Frontend (B vs. C)**: **+45.7 points** ($59.9 - 14.2$)

Both strong candidates score within 6.7 points of each other, while the unrelated profile scores over 45 points lower, confirming robust model calibration.

---

## FastAPI REST API

### Endpoints

#### 1. `GET /health`
Returns service operational status.

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy"
}
```

#### 2. `POST /api/v1/score`
Uploads a job description and resume document (.pdf or .txt) to perform end-to-end evaluation.

**Parameters** (`multipart/form-data`):
- `job_description` (string, required): Plain text job description.
- `resume_file` (file, required): Candidate resume (.pdf or .txt).

**Sample Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/score" \
  -F "job_description=$(cat samples/sample_job_description.txt)" \
  -F "resume_file=@samples/sample_resume.pdf"
```

**Sample Response**:
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
    "technical_skills": 88.5,
    "experience": 82.0,
    "education": 75.0,
    "responsibilities": 70.0,
    "tools": 80.0
  }
}
```

### Error Handling & HTTP Status Codes

- `400 Bad Request`: Empty job description, unsupported file format (e.g., `.docx`), 0-byte file, or text below character threshold (`DocumentTooShortError`).
- `422 Unprocessable Entity`: Corrupted or image-only/scanned PDF (`UnreadableDocumentError`), or JD yielding zero criteria.
- `502 Bad Gateway`: Gemini API communication failure, empty Gemini response, or malformed/non-JSON model output.
- `504 Gateway Timeout`: Gemini request timeout.
- `500 Internal Server Error`: Sentence-transformers embedding failure or server weight misconfiguration.

---

## Project Structure

```
app/
├── __init__.py
├── config.py                 # Configuration loader and validator (config.yaml & .env)
├── main.py                   # FastAPI application & /api/v1/score endpoint
├── models/
│   ├── __init__.py
│   └── schemas.py            # Pydantic schemas (Criterion, ScoredCriterion, ScoringResponse)
├── services/
│   ├── __init__.py
│   ├── extractor.py          # Document ingestion & PDF/TXT text extraction (Step 1)
│   ├── criterion_extractor.py # Gemini JD criteria extraction & validation (Step 2)
│   ├── matcher.py            # Sentence embeddings & keyword matching (Step 3)
│   └── scorer.py             # Deterministic scoring, category weights, reasoning (Step 4)
└── utils/
    ├── __init__.py
    └── text.py               # Text cleaning and normalization utilities

samples/
├── sample_resume.txt         # Sample software engineer resume
├── sample_resume.pdf         # Sample ML engineer resume
├── sample_job_description.txt # Sample job description
└── calibration/
    ├── calibration_resume_a.txt # Senior Backend Engineer A
    ├── calibration_resume_b.txt # Senior Backend Engineer B (similar profile)
    └── calibration_resume_c.txt # Junior Frontend Designer C (materially different)

tests/
├── test_parser.py            # Step 1 unit tests (25 tests)
├── test_criterion_extractor.py # Step 2 unit tests (23 tests)
├── test_matcher.py           # Step 3 unit & integration tests (20 tests)
├── test_scorer.py            # Step 4 scoring formula & weight tests (14 tests)
├── test_calibration.py       # Step 4 empirical calibration suite (1 test)
└── test_api.py               # Step 4 FastAPI integration & validation tests (12 tests)

config.yaml                   # Central configuration (weights, thresholds, chunking)
requirements.txt              # Python dependencies
.env                          # Gemini API credentials
```

---

## Running the Tests

Run the complete test suite:

```bash
pytest -v
```

Or using the virtual environment:

```powershell
.\venv\Scripts\python -m pytest -v
```

### Complete Test Results
All 95 tests across all four steps execute and pass:
- `tests/test_parser.py`: 25 passed
- `tests/test_criterion_extractor.py`: 23 passed
- `tests/test_matcher.py`: 20 passed
- `tests/test_scorer.py`: 14 passed
- `tests/test_calibration.py`: 1 passed
- `tests/test_api.py`: 12 passed

**Total: 95 passed in ~35 seconds (100% pass rate).**

---

## System Limitations
1. **OCR Not Implemented**: Scanned image-only PDFs without an embedded text layer will raise `UnreadableDocumentError` (HTTP 422).
2. **Supported Formats**: Strictly `.pdf` and `.txt`. Microsoft Word `.docx` documents are rejected at validation.
3. **Local Embedding Latency**: The initial cold-start load of `all-MiniLM-L6-v2` into memory takes 1–3 seconds on CPU, after which inference runs in milliseconds per resume chunk.
