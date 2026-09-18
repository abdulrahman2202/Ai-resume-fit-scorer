"""
Unit tests for the Gemini Job Requirement Extraction service.

All tests are isolated and use mocked responses; no real network calls or
Gemini API quota are consumed.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai import errors

from app.models.schemas import (
    Criterion,
    CriterionCategory,
    CriterionExtractionResponse,
)
from app.services.criterion_extractor import (
    ConfigurationError,
    CriterionExtractor,
    EmptyGeminiResponseError,
    ExtractionSchemaValidationError,
    GeminiAPIError,
    GeminiTimeoutError,
    InvalidExtractionJSONError,
    InvalidJobDescriptionError,
    extract_job_criteria,
)


# ==============================================================================
# Helpers and Fixtures
# ==============================================================================

def make_mock_client(response_text: str | None = None, side_effect: Exception | None = None) -> MagicMock:
    """Helper to create a mock Gemini Client."""
    mock_client = MagicMock()
    if side_effect:
        mock_client.models.generate_content.side_effect = side_effect
    else:
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_client.models.generate_content.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_jd() -> str:
    return """
    Senior Python Developer
    We are looking for an engineer with 5+ years of experience with Python and FastAPI.
    Must have a BS in Computer Science.
    Responsibilities include building distributed systems and deploying to AWS.
    Experience with Docker and PostgreSQL is required.
    """


# ==============================================================================
# 1. Valid Gemini Responses
# ==============================================================================

def test_valid_single_criterion(sample_jd: str):
    """Test extracting a single valid criterion."""
    payload = {
        "criteria": [
            {
                "name": "Python Programming",
                "category": "technical_skills",
                "description": "Strong proficiency in modern Python.",
                "keywords": ["Python", "AsyncIO"],
            }
        ]
    }
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)

    result = extractor.extract_criteria(sample_jd)

    assert isinstance(result, CriterionExtractionResponse)
    assert len(result.criteria) == 1
    c = result.criteria[0]
    assert c.name == "Python Programming"
    assert c.category == CriterionCategory.TECHNICAL_SKILLS
    assert c.description == "Strong proficiency in modern Python."
    assert c.keywords == ["Python", "AsyncIO"]


def test_valid_multiple_criteria_all_categories(sample_jd: str):
    """Test extracting multiple criteria covering all 5 allowed categories."""
    payload = {
        "criteria": [
            {
                "name": "Backend Development",
                "category": "technical_skills",
                "description": "Proficiency with Python and FastAPI.",
                "keywords": ["Python", "FastAPI"],
            },
            {
                "name": "Senior Experience",
                "category": "experience",
                "description": "At least 5 years in software engineering.",
                "keywords": ["5+ years", "Senior"],
            },
            {
                "name": "Degree Qualification",
                "category": "education",
                "description": "BS in Computer Science or equivalent.",
                "keywords": ["BS CS", "Bachelor"],
            },
            {
                "name": "System Architecture",
                "category": "responsibilities",
                "description": "Design distributed scalable services.",
                "keywords": ["Distributed Systems", "Microservices"],
            },
            {
                "name": "Cloud Infrastructure",
                "category": "tools",
                "description": "Experience with Docker, PostgreSQL, and AWS.",
                "keywords": ["Docker", "PostgreSQL", "AWS"],
            },
        ]
    }
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)

    result = extractor.extract_criteria(sample_jd)

    assert len(result.criteria) == 5
    categories = {c.category for c in result.criteria}
    assert categories == {
        CriterionCategory.TECHNICAL_SKILLS,
        CriterionCategory.EXPERIENCE,
        CriterionCategory.EDUCATION,
        CriterionCategory.RESPONSIBILITIES,
        CriterionCategory.TOOLS,
    }


def test_markdown_wrapped_json(sample_jd: str):
    """Test parsing when Gemini wraps JSON in markdown code fences."""
    raw = """```json
    {
        "criteria": [
            {
                "name": "Docker Tooling",
                "category": "tools",
                "description": "Containerization with Docker.",
                "keywords": ["Docker", "Containers"]
            }
        ]
    }
    ```"""
    client = make_mock_client(raw)
    extractor = CriterionExtractor(client=client)

    result = extractor.extract_criteria(sample_jd)
    assert len(result.criteria) == 1
    assert result.criteria[0].name == "Docker Tooling"
    assert result.criteria[0].category == CriterionCategory.TOOLS


def test_top_level_list_handling(sample_jd: str):
    """Test handling when Gemini returns a JSON list directly instead of an object."""
    raw = json.dumps([
        {
            "name": "FastAPI",
            "category": "technical_skills",
            "description": "Experience building APIs with FastAPI.",
            "keywords": ["FastAPI", "REST"],
        }
    ])
    client = make_mock_client(raw)
    extractor = CriterionExtractor(client=client)

    result = extractor.extract_criteria(sample_jd)
    assert len(result.criteria) == 1
    assert result.criteria[0].name == "FastAPI"


# ==============================================================================
# 2. Normalization & Deduplication
# ==============================================================================

def test_duplicate_criteria_deduplication(sample_jd: str):
    """Test that duplicate criteria with same category and name are merged."""
    payload = {
        "criteria": [
            {
                "name": "Python Proficiency",
                "category": "technical_skills",
                "description": "Expert in Python.",
                "keywords": ["Python", "PEP8"],
            },
            {
                "name": "  python proficiency  ",  # duplicate with whitespace and different casing
                "category": "technical_skills",
                "description": "Expert in Python.",
                "keywords": ["Python", "PyTest", "asyncio"],  # extra keywords
            },
        ]
    }
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)

    result = extractor.extract_criteria(sample_jd)
    assert len(result.criteria) == 1
    c = result.criteria[0]
    assert c.name == "Python Proficiency"
    # Keywords should be merged and deduplicated case-insensitively
    assert len(c.keywords) == 4
    assert [k.lower() for k in c.keywords] == ["python", "pep8", "pytest", "asyncio"]


def test_keyword_whitespace_and_duplicate_stripping(sample_jd: str):
    """Test that individual keywords are stripped and deduplicated."""
    payload = {
        "criteria": [
            {
                "name": "AWS Cloud",
                "category": "tools",
                "description": "Cloud hosting.",
                "keywords": ["  AWS  ", "EC2", "aws", "  EC2  ", "", "S3"],
            }
        ]
    }
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)

    result = extractor.extract_criteria(sample_jd)
    assert len(result.criteria) == 1
    # Should strip spaces, drop empty strings, deduplicate case-insensitively
    assert result.criteria[0].keywords == ["AWS", "EC2", "S3"]


# ==============================================================================
# 3. Malformed JSON & JSON Structure Failures
# ==============================================================================

def test_malformed_json_syntax_error(sample_jd: str):
    """Test that malformed JSON syntax raises InvalidExtractionJSONError."""
    client = make_mock_client('{"criteria": [{"name": "Python", "category": "technical_skills"')
    extractor = CriterionExtractor(client=client)

    with pytest.raises(InvalidExtractionJSONError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "Failed to parse Gemini response as JSON" in str(exc_info.value)


def test_unexpected_json_primitive_type(sample_jd: str):
    """Test that JSON returning a primitive or non-dict/non-list raises InvalidExtractionJSONError."""
    client = make_mock_client('"This is just a string, not criteria"')
    extractor = CriterionExtractor(client=client)

    with pytest.raises(InvalidExtractionJSONError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "Expected JSON object or list from Gemini" in str(exc_info.value)


# ==============================================================================
# 4. Schema Validation Failures (Missing fields, Invalid categories)
# ==============================================================================

def test_missing_required_criterion_fields(sample_jd: str):
    """Test that missing required fields in a criterion raises ExtractionSchemaValidationError."""
    payload = {
        "criteria": [
            {
                # Missing 'name' and 'description'
                "category": "technical_skills",
                "keywords": ["Python"],
            }
        ]
    }
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)

    with pytest.raises(ExtractionSchemaValidationError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "Pydantic schema validation" in str(exc_info.value)


def test_missing_criteria_field_in_object(sample_jd: str):
    """Test that an object missing the 'criteria' key raises ExtractionSchemaValidationError."""
    payload = {"skills": ["Python", "FastAPI"]}  # Missing 'criteria'
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)

    with pytest.raises(ExtractionSchemaValidationError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "Pydantic schema validation" in str(exc_info.value)


def test_invalid_criterion_category(sample_jd: str):
    """Test that a criterion with an unallowed category raises ExtractionSchemaValidationError."""
    payload = {
        "criteria": [
            {
                "name": "Teamwork",
                "category": "soft_skills",  # Invalid category!
                "description": "Works well with others.",
                "keywords": ["communication"],
            }
        ]
    }
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)

    with pytest.raises(ExtractionSchemaValidationError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "Pydantic schema validation" in str(exc_info.value)


# ==============================================================================
# 5. Empty Responses & Candidate Formatting Failures
# ==============================================================================

def test_empty_string_gemini_response(sample_jd: str):
    """Test that an empty string response raises EmptyGeminiResponseError."""
    client = make_mock_client("")
    extractor = CriterionExtractor(client=client)

    with pytest.raises(EmptyGeminiResponseError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "empty or whitespace-only response" in str(exc_info.value)


def test_whitespace_only_gemini_response(sample_jd: str):
    """Test that a whitespace-only response raises EmptyGeminiResponseError."""
    client = make_mock_client("   \n   \t  ")
    extractor = CriterionExtractor(client=client)

    with pytest.raises(EmptyGeminiResponseError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "empty or whitespace-only response" in str(exc_info.value)


def test_none_gemini_response(sample_jd: str):
    """Test that a None return from client.models.generate_content raises EmptyGeminiResponseError."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = None
    extractor = CriterionExtractor(client=mock_client)

    with pytest.raises(EmptyGeminiResponseError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "returned None" in str(exc_info.value)


def test_response_with_no_candidates_or_text(sample_jd: str):
    """Test that a response object with text=None and no candidates raises EmptyGeminiResponseError."""
    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = []
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    extractor = CriterionExtractor(client=mock_client)

    with pytest.raises(EmptyGeminiResponseError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "contained no candidates or text" in str(exc_info.value)


# ==============================================================================
# 6. Gemini API, Network & Timeout Failures
# ==============================================================================

def test_gemini_api_error(sample_jd: str):
    """Test that an APIError from the Gemini SDK raises GeminiAPIError."""
    api_err = errors.APIError(500, {"error": "Internal Gemini Server Error"})
    client = make_mock_client(side_effect=api_err)
    extractor = CriterionExtractor(client=client, sleep_fn=MagicMock())

    with pytest.raises(GeminiAPIError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "Gemini API error" in str(exc_info.value)


def test_gemini_network_error(sample_jd: str):
    """Test that network connection errors raise GeminiAPIError."""
    net_err = httpx.ConnectError("Failed to establish connection to Gemini endpoint")
    client = make_mock_client(side_effect=net_err)
    extractor = CriterionExtractor(client=client, sleep_fn=MagicMock())

    with pytest.raises(GeminiAPIError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "Gemini network communication error" in str(exc_info.value)


def test_gemini_timeout_error(sample_jd: str):
    """Test that timeout exceptions raise GeminiTimeoutError."""
    timeout_err = httpx.TimeoutException("Read timed out after 30.0 seconds")
    client = make_mock_client(side_effect=timeout_err)
    extractor = CriterionExtractor(client=client, sleep_fn=MagicMock())

    with pytest.raises(GeminiTimeoutError) as exc_info:
        extractor.extract_criteria(sample_jd)
    assert "request timed out" in str(exc_info.value)


# ==============================================================================
# 7. Input Validation & Configuration Errors
# ==============================================================================

def test_empty_job_description_raises_error():
    """Test that passing an empty or whitespace JD raises InvalidJobDescriptionError."""
    client = make_mock_client("{}")
    extractor = CriterionExtractor(client=client)

    with pytest.raises(InvalidJobDescriptionError):
        extractor.extract_criteria("")

    with pytest.raises(InvalidJobDescriptionError):
        extractor.extract_criteria("   \n\t  ")


def test_missing_api_key_configuration():
    """Test that initializing without client and without API key raises ConfigurationError."""
    with patch("app.services.criterion_extractor.get_gemini_api_key", return_value=None):
        with pytest.raises(ConfigurationError) as exc_info:
            CriterionExtractor(api_key=None, client=None)
        assert "Gemini API key is not configured" in str(exc_info.value)


def test_placeholder_api_key_raises_configuration_error():
    """Test that default placeholder API key raises ConfigurationError."""
    with patch("app.services.criterion_extractor.get_gemini_api_key", return_value="your_gemini_api_key_here"):
        with pytest.raises(ConfigurationError) as exc_info:
            CriterionExtractor(api_key=None, client=None)
        assert "Gemini API key is not configured" in str(exc_info.value)


# ==============================================================================
# 8. Convenience Wrapper Function
# ==============================================================================

def test_extract_job_criteria_wrapper(sample_jd: str):
    """Test the extract_job_criteria convenience function."""
    payload = {
        "criteria": [
            {
                "name": "PostgreSQL",
                "category": "tools",
                "description": "Relational database skills.",
                "keywords": ["PostgreSQL", "SQL"],
            }
        ]
    }
    client = make_mock_client(json.dumps(payload))

    result = extract_job_criteria(sample_jd, client=client)
    assert len(result.criteria) == 1
    assert result.criteria[0].name == "PostgreSQL"
    assert result.criteria[0].category == CriterionCategory.TOOLS


def test_sample_job_description_file_integration():
    """Test extracting criteria from the actual sample_job_description.txt file with a mock client."""
    from pathlib import Path
    sample_path = Path(__file__).resolve().parent.parent / "samples" / "sample_job_description.txt"
    jd_content = sample_path.read_text(encoding="utf-8")

    payload = {
        "criteria": [
            {
                "name": "Python & FastAPI Microservices",
                "category": "technical_skills",
                "description": "Design and build production-grade microservices with Python and FastAPI.",
                "keywords": ["Python", "FastAPI", "Microservices"],
            },
            {
                "name": "Software Engineering Experience",
                "category": "experience",
                "description": "5+ years of software engineering experience.",
                "keywords": ["5+ years", "Senior Engineer"],
            },
            {
                "name": "Computer Science Degree",
                "category": "education",
                "description": "Bachelor's degree in Computer Science or equivalent practical experience.",
                "keywords": ["Bachelor's", "Computer Science"],
            },
            {
                "name": "Clean Code and Testing",
                "category": "responsibilities",
                "description": "Write clean, modular code with high test coverage using pytest.",
                "keywords": ["Clean Code", "pytest", "Test Coverage"],
            },
            {
                "name": "Database & Container Tools",
                "category": "tools",
                "description": "Experience with PostgreSQL, Redis, Docker, and AWS/GCP.",
                "keywords": ["PostgreSQL", "Redis", "Docker", "AWS", "GCP"],
            },
        ]
    }
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)
    result = extractor.extract_criteria(jd_content)

    assert len(result.criteria) == 5
    assert all(isinstance(c, Criterion) for c in result.criteria)
    # Check that client received prompt containing sample JD text
    called_args = client.models.generate_content.call_args
    prompt_used = called_args.kwargs["contents"]
    assert "Senior Backend AI Engineer" in prompt_used
    assert "CloudScale AI" in prompt_used


# ==============================================================================
# 9. Transient Failure Retry & Exponential Backoff Tests
# ==============================================================================

def test_retry_503_followed_by_success(sample_jd: str):
    """Verify that a 503 UNAVAILABLE failure retries and succeeds on the next attempt."""
    err_503 = errors.APIError(503, {"error": "503 UNAVAILABLE: High demand"})
    success_resp = MagicMock()
    success_resp.text = json.dumps({
        "criteria": [
            {
                "name": "Python Skills",
                "category": "technical_skills",
                "description": "Python engineering.",
                "keywords": ["Python"],
            }
        ]
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [err_503, success_resp]
    mock_sleep = MagicMock()

    extractor = CriterionExtractor(
        client=mock_client,
        max_retries=3,
        initial_retry_delay=1.0,
        sleep_fn=mock_sleep,
    )

    result = extractor.extract_criteria(sample_jd)

    assert len(result.criteria) == 1
    assert result.criteria[0].name == "Python Skills"
    # generate_content should be called twice (1 failure + 1 success)
    assert mock_client.models.generate_content.call_count == 2
    # sleep_fn should be called once with ~1.0s delay
    assert mock_sleep.call_count == 1
    assert mock_sleep.call_args[0][0] == 1.0


def test_retry_503_followed_by_503_followed_by_success(sample_jd: str):
    """Verify that multiple consecutive 503 errors retry with exponential backoff and succeed."""
    err_503_1 = errors.APIError(503, {"error": "503 UNAVAILABLE: High demand"})
    err_503_2 = errors.APIError(503, {"error": "503 UNAVAILABLE: High demand"})
    success_resp = MagicMock()
    success_resp.text = json.dumps({
        "criteria": [
            {
                "name": "FastAPI",
                "category": "technical_skills",
                "description": "FastAPI APIs.",
                "keywords": ["FastAPI"],
            }
        ]
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [err_503_1, err_503_2, success_resp]
    mock_sleep = MagicMock()

    extractor = CriterionExtractor(
        client=mock_client,
        max_retries=3,
        initial_retry_delay=1.0,
        sleep_fn=mock_sleep,
    )

    result = extractor.extract_criteria(sample_jd)

    assert len(result.criteria) == 1
    assert mock_client.models.generate_content.call_count == 3
    # Delays: 1.0s for attempt 1, 2.0s for attempt 2
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0][0][0] == 1.0
    assert mock_sleep.call_args_list[1][0][0] == 2.0


def test_retry_503_exhausts_all_retries(sample_jd: str):
    """Verify that persistent 503 errors exhaust all retries and raise GeminiAPIError."""
    err_503 = errors.APIError(503, {"error": "503 UNAVAILABLE: High demand"})
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = err_503
    mock_sleep = MagicMock()

    extractor = CriterionExtractor(
        client=mock_client,
        max_retries=3,
        initial_retry_delay=1.0,
        sleep_fn=mock_sleep,
    )

    with pytest.raises(GeminiAPIError) as exc_info:
        extractor.extract_criteria(sample_jd)

    assert "Gemini API error (503)" in str(exc_info.value)
    # 1 initial try + 3 retries = 4 calls total
    assert mock_client.models.generate_content.call_count == 4
    # 3 retry delays: 1.0, 2.0, 4.0
    assert mock_sleep.call_count == 3
    delays = [call[0][0] for call in mock_sleep.call_args_list]
    assert delays == [1.0, 2.0, 4.0]


def test_permanent_400_and_404_no_retry(sample_jd: str):
    """Verify that permanent 400 (Bad Request) and 404 (Not Found) errors do NOT trigger retries."""
    err_400 = errors.APIError(400, {"error": "Invalid argument provided to Gemini"})
    mock_client_400 = MagicMock()
    mock_client_400.models.generate_content.side_effect = err_400
    mock_sleep_400 = MagicMock()

    extractor_400 = CriterionExtractor(
        client=mock_client_400,
        max_retries=3,
        sleep_fn=mock_sleep_400,
    )

    with pytest.raises(GeminiAPIError):
        extractor_400.extract_criteria(sample_jd)

    # Exactly 1 call, 0 sleep retries
    assert mock_client_400.models.generate_content.call_count == 1
    assert mock_sleep_400.call_count == 0

    err_404 = errors.APIError(404, {"error": "Model not found"})
    mock_client_404 = MagicMock()
    mock_client_404.models.generate_content.side_effect = err_404
    mock_sleep_404 = MagicMock()

    extractor_404 = CriterionExtractor(
        client=mock_client_404,
        max_retries=3,
        sleep_fn=mock_sleep_404,
    )

    with pytest.raises(GeminiAPIError):
        extractor_404.extract_criteria(sample_jd)

    assert mock_client_404.models.generate_content.call_count == 1
    assert mock_sleep_404.call_count == 0


def test_successful_request_makes_one_call_only(sample_jd: str):
    """Verify that a successful request executes exactly one call with zero retries."""
    payload = {
        "criteria": [
            {
                "name": "Testing",
                "category": "responsibilities",
                "description": "Write pytest suites.",
                "keywords": ["pytest"],
            }
        ]
    }
    client = make_mock_client(json.dumps(payload))
    mock_sleep = MagicMock()

    extractor = CriterionExtractor(
        client=client,
        max_retries=3,
        sleep_fn=mock_sleep,
    )

    result = extractor.extract_criteria(sample_jd)

    assert len(result.criteria) == 1
    assert client.models.generate_content.call_count == 1
    assert mock_sleep.call_count == 0


def test_gemini_call_configuration_disables_afc_and_tools(sample_jd: str):
    """Verify that generate_content is called with tools=None and AFC explicitly disabled."""
    payload = {
        "criteria": [
            {
                "name": "Python",
                "category": "technical_skills",
                "description": "Python engineering skills",
                "keywords": ["Python"],
            }
        ]
    }
    client = make_mock_client(json.dumps(payload))
    extractor = CriterionExtractor(client=client)

    extractor.extract_criteria(sample_jd)

    assert client.models.generate_content.call_count == 1
    call_kwargs = client.models.generate_content.call_args.kwargs
    config = call_kwargs.get("config")
    assert config is not None
    # Verify no tools or agents are passed
    assert getattr(config, "tools", None) is None
    # Verify AFC is explicitly disabled
    afc = getattr(config, "automatic_function_calling", None)
    assert afc is not None
    assert afc.disable is True
    # Verify simple JSON generation config
    assert getattr(config, "response_mime_type", None) == "application/json"
    assert getattr(config, "temperature", None) == 0.0

