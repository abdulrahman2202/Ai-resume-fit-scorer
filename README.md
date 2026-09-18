# Resume Job Fit Scorer

An applied-AI evaluation tool designed to analyze resumes against job descriptions, extract structured evaluation criteria, compute semantic and keyword match scores, and provide actionable feedback.

---

## Current Status: Step 3 – Resume-to-Criterion Matching Engine

The first three phases of the pipeline are complete:
- **Step 1**: Multi-format document ingestion and validation for resumes and job descriptions.
- **Step 2**: Gemini-powered extraction of structured hiring criteria with strict Pydantic schema validation.
- **Step 3**: Deterministic resume-to-criterion matching engine computing dual signals (semantic similarity via `all-MiniLM-L6-v2` and keyword evidence).

---

## Architecture & Design Decisions

```
Job Description
      ↓
Gemini API (gemini-2.5-flash) [Step 2]
      ↓
Structured Hiring Criteria (Pydantic validated)
      ↓
┌────────────────────────────────────────────────────────┐
│ Matching Engine [Step 3]                               │
│                                                        │
│  Resume Text ──► Paragraph Chunking (500 char blocks) │
│                        │                               │
│                        ▼                               │
│  1. Semantic Match: all-MiniLM-L6-v2 Embeddings        │
│     (Cosine similarity against resume chunks)          │
│     ──► semantic_score & best evidence chunk           │
│                                                        │
│  2. Keyword Evidence: Deterministic Regex Matching     │
│     (Case-insensitive, whitespace-tolerant)            │
│     ──► keyword_score, matched & unmatched lists       │
└────────────────────────────────────────────────────────┘
      ↓
CriterionMatchResult (No final scoring in Step 3)
```

### Why Embeddings Are Used
Embedding models (`all-MiniLM-L6-v2` via `sentence-transformers`) transform text into dense vector representations where geometric proximity corresponds to semantic similarity. Candidates frequently express qualifications through synonyms, paraphrasing, or conceptual descriptions (e.g. "built asynchronous microservices" matches "REST API backend engineering") that keyword searches miss entirely.

### Why Keyword Matching Is Also Used
While embeddings capture semantic context, they can be overly forgiving with strict technical specifics. For example, in vector space, "AWS", "GCP", and "Azure" are proximate because all represent cloud providers. Keyword evidence enforces precision by verifying that exact required tools, languages, frameworks, or certifications are literally present.

### Why Gemini Is NOT Used for Matching
Employing LLMs for candidate matching introduces:
1. **Non-determinism**: Varying scores across runs for the exact same resume.
2. **Hallucination risk**: Imagining candidate qualifications or overlooking stated experience.
3. **Prompt injection vulnerabilities**: Susceptibility to resume-based prompt injections.
4. **Latency & Cost**: Significant network latency and token expenses per candidate.

A hybrid of local embeddings and deterministic regex keyword matching is completely reproducible, explainable, private, and runs in milliseconds.

### How Resume Chunking Works
Resumes are naturally structured into sections, bullet points, and paragraphs. Our simple, maintainable chunker:
- Respects natural paragraph boundaries (`\n\n`) and line breaks.
- Sequentially bundles paragraphs up to a configurable `max_chunk_size` (default: 500 characters).
- Supports a configurable `chunk_overlap` (default: 50 characters) to preserve context across boundaries.
- Slices oversized blocks cleanly without external NLP dependencies.
- Evaluates each criterion against all chunks to isolate the single strongest evidence snippet.

### What the Semantic Score Means
The semantic score represents the peak cosine similarity between a criterion's query (`name: description`) and the candidate's resume chunks:
- **Raw Cosine Similarity**: Mathematical cosine in $[-1.0, 1.0]$.
- **Normalization Transformation**: Clamped to $[0.0, 1.0]$ via `max(0.0, min(1.0, raw_sim))`. Positive cosine values are preserved without distortion, while negative noise (divergent/opposite concepts) is clamped to zero (avoiding affine distortions that falsely turn orthogonal text into 50% matches).
- **Evidence**: The specific resume chunk that generated the highest similarity is captured as supporting evidence.

### What the Keyword Score Means
The keyword score is the exact proportion of a criterion's predefined keywords present in the resume:
$$\text{keyword\_score} = \frac{\text{matched\_keywords}}{\text{total\_keywords}} \in [0.0, 1.0]$$
- **Matching Rules**: Case-insensitive, whitespace-tolerant, and word-boundary aware (`(?<!\w)...\s+...(?!\w)`). Technical symbols (e.g., `C++`, `C#`, `.NET`) are safely matched.
- If a criterion defines no keywords, `keyword_score` defaults safely to `0.0`.

---

## Step 3 Features Implemented

1. **`ResumeMatcher` Service (`app/services/matcher.py`)**:
   - Paragraph-aware resume chunking with configurable size and overlap.
   - Singleton `EmbeddingModelManager` ensuring `all-MiniLM-L6-v2` is loaded once and reused across evaluations.
   - Pure NumPy cosine similarity and explicit $[0.0, 1.0]$ normalization.
   - Deterministic keyword evidence extractor returning matched and unmatched keyword lists.
   - Safe edge-case handling for empty resumes, blank criteria, and inference exceptions.

2. **Pydantic Match Schemas (`app/models/schemas.py`)**:
   - `CriterionMatchResult`: Holds `criterion_name`, `criterion_category`, `semantic_score`, `raw_semantic_score`, `keyword_score`, `matched_keywords`, `unmatched_keywords`, and `evidence`.
   - `ResumeMatchResult`: Holds the collection of individual `CriterionMatchResult` objects.

3. **Configurable Matching Parameters (`config.yaml`)**:
   - `matching.chunk_size`: 500 characters
   - `matching.chunk_overlap`: 50 characters
   - `matching.embedding_model`: `all-MiniLM-L6-v2`

---

## Project Structure

```
app/
├── __init__.py
├── config.py                 # Configuration loader for config.yaml & .env
├── main.py
├── models/
│   ├── __init__.py
│   └── schemas.py            # Pydantic schemas (Criterion, CriterionMatchResult, etc.)
├── services/
│   ├── __init__.py
│   ├── extractor.py          # Document parsing & text extraction (Step 1)
│   ├── criterion_extractor.py # Gemini JD criteria extraction & validation (Step 2)
│   ├── matcher.py            # Dual-signal resume-to-criterion matching engine (Step 3)
│   └── scorer.py
└── utils/
    ├── __init__.py
    └── text.py               # Text normalization and cleaning utilities

tests/
├── test_parser.py            # Step 1 unit tests (25 tests)
├── test_criterion_extractor.py # Step 2 unit tests (23 tests)
└── test_matcher.py           # Step 3 unit & integration tests (20 tests)

samples/
├── sample_resume.txt         # Sample software engineer resume in TXT
├── sample_resume.pdf         # Sample ML engineer resume in PDF
└── sample_job_description.txt # Sample job description

config.yaml                   # Application scoring, threshold, and matching configuration
requirements.txt              # Python dependencies
.env                          # Gemini API credentials & model settings
```

---

## Running the Unit Tests

Run the complete test suite using `pytest`:

```bash
pytest -v
```

Or using the project's virtual environment:

```powershell
.\venv\Scripts\python -m pytest -v
```

All 68 test cases across all three steps will execute and pass:
- **Step 1 (Parser)**: 25 tests
- **Step 2 (Gemini Criterion Extractor)**: 23 tests
- **Step 3 (Resume Matcher)**: 20 tests
