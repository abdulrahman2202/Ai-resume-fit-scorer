"""
Integration and validation tests for the FastAPI application endpoints.

Tests:
- GET /health
- POST /api/v1/score with valid TXT and PDF uploads
- Input validations (empty JD, unsupported file formats, empty files, short files)
- Error handling for Gemini failures, timeouts, malformed JSON, and embedding errors
"""

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app, get_criterion_extractor, get_resume_matcher
from app.models.schemas import (
    Criterion,
    CriterionCategory,
    CriterionExtractionResponse,
)
from app.services.criterion_extractor import (
    EmptyGeminiResponseError,
    GeminiAPIError,
    GeminiTimeoutError,
    InvalidExtractionJSONError,
)
from app.services.matcher import EmbeddingModelError

client = TestClient(app)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


# ==============================================================================
# Mock Fixtures
# ==============================================================================

@pytest.fixture
def mock_criterion_extractor():
    mock = MagicMock()
    mock.extract_criteria.return_value = CriterionExtractionResponse(
        criteria=[
            Criterion(
                name="Python Skills",
                category=CriterionCategory.TECHNICAL_SKILLS,
                description="Experience in Python programming.",
                keywords=["Python"],
            ),
            Criterion(
                name="Docker Tools",
                category=CriterionCategory.TOOLS,
                description="Containerization with Docker.",
                keywords=["Docker"],
            ),
        ]
    )
    return mock


# ==============================================================================
# 1. Health Check Endpoint
# ==============================================================================

def test_health_check():
    """Verify GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}


# ==============================================================================
# 2. POST /api/v1/score Valid Flows
# ==============================================================================

def test_score_endpoint_valid_txt_upload(mock_criterion_extractor):
    """Test successful scoring request using a TXT resume upload."""
    app.dependency_overrides[get_criterion_extractor] = lambda: mock_criterion_extractor

    jd_text = "Senior Python Developer required with Docker experience."
    resume_content = (
        "Professional Summary:\n"
        "Senior Python Software Engineer with 6 years of experience.\n\n"
        "Technical Skills:\n"
        "Expertise in Python, FastAPI, Docker containers, and PostgreSQL.\n\n"
        "Experience:\n"
        "Engineered scalable backend microservices and automated build pipelines."
    )

    response = client.post(
        "/api/v1/score",
        data={"job_description": jd_text},
        files={"resume_file": ("resume.txt", io.BytesIO(resume_content.encode("utf-8")), "text/plain")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "overall_score" in data
    assert 0.0 <= data["overall_score"] <= 100.0
    assert "criteria" in data
    assert len(data["criteria"]) == 2
    assert data["criteria"][0]["name"] == "Python Skills"
    assert "reason" in data["criteria"][0]
    assert data["criteria"][0]["score"] >= 0.0


def test_score_endpoint_valid_pdf_upload(mock_criterion_extractor):
    """Test successful scoring request using the sample PDF resume."""
    app.dependency_overrides[get_criterion_extractor] = lambda: mock_criterion_extractor

    pdf_path = SAMPLES_DIR / "sample_resume.pdf"
    assert pdf_path.is_file()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    response = client.post(
        "/api/v1/score",
        data={"job_description": "Machine Learning Engineer with Python background."},
        files={"resume_file": ("sample_resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "overall_score" in data
    assert len(data["criteria"]) == 2


# ==============================================================================
# 3. Validation Failures (Bad Requests)
# ==============================================================================

def test_score_endpoint_empty_job_description():
    """Verify HTTP 400 when job description is empty or whitespace."""
    response = client.post(
        "/api/v1/score",
        data={"job_description": "   \n  "},
        files={"resume_file": ("resume.txt", io.BytesIO(b"Valid resume text here..."), "text/plain")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Job description cannot be empty" in response.json()["detail"]


def test_score_endpoint_unsupported_file_extension():
    """Verify HTTP 400 when uploaded file has an unsupported extension (e.g. .docx)."""
    response = client.post(
        "/api/v1/score",
        data={"job_description": "Job description here..."},
        files={"resume_file": ("resume.docx", io.BytesIO(b"Some docx bytes"), "application/octet-stream")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unsupported file format" in response.json()["detail"]


def test_score_endpoint_empty_file_upload():
    """Verify HTTP 400 when uploaded file is empty (0 bytes)."""
    response = client.post(
        "/api/v1/score",
        data={"job_description": "Job description here..."},
        files={"resume_file": ("resume.txt", io.BytesIO(b""), "text/plain")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "empty (0 bytes)" in response.json()["detail"]


def test_score_endpoint_document_too_short():
    """Verify HTTP 400 when resume text is under the character threshold."""
    short_content = "Too short"
    response = client.post(
        "/api/v1/score",
        data={"job_description": "Job description text..."},
        files={"resume_file": ("resume.txt", io.BytesIO(short_content.encode("utf-8")), "text/plain")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "too short" in response.json()["detail"]


def test_score_endpoint_unreadable_pdf():
    """Verify HTTP 422 when PDF is corrupt."""
    corrupt_pdf = b"%PDF-1.4 corrupt data not real pdf content %%EOF"
    response = client.post(
        "/api/v1/score",
        data={"job_description": "Valid job description..."},
        files={"resume_file": ("corrupted.pdf", io.BytesIO(corrupt_pdf), "application/pdf")},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Unable to extract readable text" in response.json()["detail"]


# ==============================================================================
# 4. Service Failure & Error Translation
# ==============================================================================

def test_score_endpoint_gemini_api_failure():
    """Verify HTTP 502 when Gemini API encounters a communication failure."""
    failing_extractor = MagicMock()
    failing_extractor.extract_criteria.side_effect = GeminiAPIError("Endpoint unreachable")

    app.dependency_overrides[get_criterion_extractor] = lambda: failing_extractor

    valid_resume = (
        "Professional Summary:\n"
        "Senior Software Engineer with 6 years of experience building Python and FastAPI microservices.\n\n"
        "Technical Skills:\n"
        "Python, Docker, AWS, PostgreSQL, Redis.\n\n"
        "Experience:\n"
        "Designed and maintained scalable cloud backend systems and data pipelines."
    )
    response = client.post(
        "/api/v1/score",
        data={"job_description": "Job description text..."},
        files={"resume_file": ("resume.txt", io.BytesIO(valid_resume.encode("utf-8")), "text/plain")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert "Gemini API communication failure" in response.json()["detail"]


def test_score_endpoint_gemini_timeout():
    """Verify HTTP 504 when Gemini API times out."""
    timing_out_extractor = MagicMock()
    timing_out_extractor.extract_criteria.side_effect = GeminiTimeoutError("Request timed out")

    app.dependency_overrides[get_criterion_extractor] = lambda: timing_out_extractor

    valid_resume = (
        "Professional Summary:\n"
        "Senior Software Engineer with 6 years of experience building Python and FastAPI microservices.\n\n"
        "Technical Skills:\n"
        "Python, Docker, AWS, PostgreSQL, Redis.\n\n"
        "Experience:\n"
        "Designed and maintained scalable cloud backend systems and data pipelines."
    )
    response = client.post(
        "/api/v1/score",
        data={"job_description": "Job description text..."},
        files={"resume_file": ("resume.txt", io.BytesIO(valid_resume.encode("utf-8")), "text/plain")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
    assert "timed out" in response.json()["detail"]


def test_score_endpoint_gemini_malformed_json():
    """Verify HTTP 502 when Gemini returns invalid JSON."""
    malformed_extractor = MagicMock()
    malformed_extractor.extract_criteria.side_effect = InvalidExtractionJSONError("JSON syntax error")

    app.dependency_overrides[get_criterion_extractor] = lambda: malformed_extractor

    valid_resume = (
        "Professional Summary:\n"
        "Senior Software Engineer with 6 years of experience building Python and FastAPI microservices.\n\n"
        "Technical Skills:\n"
        "Python, Docker, AWS, PostgreSQL, Redis.\n\n"
        "Experience:\n"
        "Designed and maintained scalable cloud backend systems and data pipelines."
    )
    response = client.post(
        "/api/v1/score",
        data={"job_description": "Job description text..."},
        files={"resume_file": ("resume.txt", io.BytesIO(valid_resume.encode("utf-8")), "text/plain")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert "invalid or malformed response" in response.json()["detail"]


def test_score_endpoint_embedding_inference_failure(mock_criterion_extractor):
    """Verify HTTP 500 when embedding model fails during matching."""
    app.dependency_overrides[get_criterion_extractor] = lambda: mock_criterion_extractor

    failing_matcher = MagicMock()
    failing_matcher.match_criteria.side_effect = EmbeddingModelError("CUDA out of memory")
    app.dependency_overrides[get_resume_matcher] = lambda: failing_matcher

    valid_resume = (
        "Professional Summary:\n"
        "Senior Software Engineer with 6 years of experience building Python and FastAPI microservices.\n\n"
        "Technical Skills:\n"
        "Python, Docker, AWS, PostgreSQL, Redis.\n\n"
        "Experience:\n"
        "Designed and maintained scalable cloud backend systems and data pipelines."
    )
    response = client.post(
        "/api/v1/score",
        data={"job_description": "Job description text..."},
        files={"resume_file": ("resume.txt", io.BytesIO(valid_resume.encode("utf-8")), "text/plain")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Embedding model inference failure" in response.json()["detail"]
