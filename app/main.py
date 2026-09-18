"""
FastAPI Application for Resume Job Fit Scorer.

Provides:
- GET /health : Service health check
- POST /api/v1/score : End-to-end resume evaluation against a job description
"""

import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import (
    ConfigurationError,
    validate_configuration,
)
from app.models.schemas import HealthResponse, ScoringResponse
from app.services.criterion_extractor import (
    CriterionExtractionError,
    CriterionExtractor,
    EmptyGeminiResponseError,
    ExtractionSchemaValidationError,
    GeminiAPIError,
    GeminiTimeoutError,
    InvalidExtractionJSONError,
    InvalidJobDescriptionError,
)
from app.services.extractor import (
    DocumentTooShortError,
    ResumeExtractionError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    extract_resume,
)
from app.services.matcher import (
    EmbeddingModelError,
    InvalidCriterionError,
    InvalidInputError,
    MatchingError,
    ResumeMatcher,
)
from app.services.scorer import (
    InvalidWeightConfigurationError,
    ScoringEngine,
    ScoringError,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Job Fit Scorer",
    description="Applied-AI engine evaluating resumes against job criteria using semantic embeddings and keyword evidence.",
    version="1.0.0",
)


# ==============================================================================
# Dependency Providers
# ==============================================================================

def get_criterion_extractor() -> CriterionExtractor:
    return CriterionExtractor()


def get_resume_matcher() -> ResumeMatcher:
    return ResumeMatcher()


def get_scoring_engine() -> ScoringEngine:
    return ScoringEngine()


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Service health check endpoint."""
    return HealthResponse(status="healthy")


@app.post(
    "/api/v1/score",
    response_model=ScoringResponse,
    status_code=status.HTTP_200_OK,
    tags=["Scoring"],
)
async def score_resume(
    job_description: str = Form(..., description="Plain text job description"),
    resume_file: UploadFile = File(..., description="Resume document in PDF or TXT format"),
    criterion_extractor: CriterionExtractor = Depends(get_criterion_extractor),
    resume_matcher: ResumeMatcher = Depends(get_resume_matcher),
    scoring_engine: ScoringEngine = Depends(get_scoring_engine),
) -> ScoringResponse:
    """
    Score a candidate resume against a job description.

    Pipeline:
    1. Validate inputs and file extension (.pdf, .txt).
    2. Extract and clean resume text.
    3. Extract structured criteria from job description via Gemini.
    4. Match resume text against criteria using semantic embeddings and keyword matching.
    5. Compute deterministic criterion scores and category-weighted overall score.
    """
    # 1. Validate Job Description
    if not job_description or not job_description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description cannot be empty or blank.",
        )

    # 2. Validate Resume File
    filename = resume_file.filename or "resume.txt"
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Only .pdf and .txt are supported.",
        )

    try:
        content_bytes = await resume_file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded resume file: {exc}",
        )

    if not content_bytes or len(content_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded resume file is empty (0 bytes).",
        )

    # 3. Extract Resume Text
    try:
        resume_stream = io.BytesIO(content_bytes)
        resume_text = extract_resume(resume_stream, filename=filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DocumentTooShortError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except UnreadableDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to extract readable text from resume document: {exc}",
        )
    except ResumeExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # 4. Extract Job Criteria via Gemini
    try:
        extraction_response = criterion_extractor.extract_criteria(job_description.strip())
    except InvalidJobDescriptionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except GeminiTimeoutError as exc:
        logger.error("Gemini API timed out during criterion extraction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Gemini API timed out while analyzing job description.",
        )
    except (EmptyGeminiResponseError, InvalidExtractionJSONError, ExtractionSchemaValidationError) as exc:
        logger.error("Gemini criterion extraction failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini returned an invalid or malformed response: {exc}",
        )
    except GeminiAPIError as exc:
        logger.error("Gemini API communication error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API communication failure: {exc}",
        )
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server configuration error: {exc}",
        )
    except CriterionExtractionError as exc:
        logger.error("Criterion extraction error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    if not extraction_response.criteria:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No hiring criteria could be extracted from the provided job description.",
        )

    # 5. Match Resume against Criteria
    try:
        match_result = resume_matcher.match_criteria(resume_text, extraction_response.criteria)
    except (InvalidCriterionError, InvalidInputError) as exc:
        logger.error("Matching input error: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except EmbeddingModelError as exc:
        logger.error("Embedding model failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding model inference failure: {exc}",
        )
    except MatchingError as exc:
        logger.error("Matching error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # 6. Score Criteria & Calculate Overall Score
    try:
        scoring_response = scoring_engine.evaluate(
            matches=match_result.matches,
            criteria=extraction_response.criteria,
        )
    except InvalidWeightConfigurationError as exc:
        logger.error("Invalid weight configuration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring configuration error: {exc}",
        )
    except ScoringError as exc:
        logger.error("Scoring error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return scoring_response
