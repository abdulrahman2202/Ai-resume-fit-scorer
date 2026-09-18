"""
Job Requirement Extraction Service using Google Gemini API.

Extracts structured hiring criteria (technical skills, experience, education,
responsibilities, tools) from raw job descriptions and validates them with Pydantic.
This module strictly performs requirement extraction without candidate scoring,
ranking, or evaluation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

import httpx
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from app.config import get_gemini_api_key, get_gemini_model
from app.models.schemas import (
    Criterion,
    CriterionCategory,
    CriterionExtractionResponse,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exception Hierarchy
# ==============================================================================

class CriterionExtractionError(Exception):
    """Base exception for all criterion extraction errors."""
    pass


class ConfigurationError(CriterionExtractionError):
    """Raised when Gemini configuration (e.g. API key) is missing or invalid."""
    pass


class InvalidJobDescriptionError(CriterionExtractionError):
    """Raised when the input job description is empty or invalid."""
    pass


class GeminiAPIError(CriterionExtractionError):
    """Raised when Gemini API, network, or communication fails."""
    pass


class GeminiTimeoutError(GeminiAPIError):
    """Raised when a Gemini API call times out."""
    pass


class EmptyGeminiResponseError(CriterionExtractionError):
    """Raised when Gemini returns an empty, null, or blocked response."""
    pass


class InvalidExtractionJSONError(CriterionExtractionError):
    """Raised when Gemini output cannot be parsed as valid JSON."""
    pass


class ExtractionSchemaValidationError(CriterionExtractionError):
    """Raised when Gemini output fails Pydantic schema validation."""
    pass


# ==============================================================================
# Prompt Template
# ==============================================================================

CRITERION_EXTRACTION_PROMPT_TEMPLATE = """You are an expert talent evaluator and technical recruiter.
Your sole task is to analyze the following Job Description and extract structured hiring requirements.

### STRICT OPERATIONAL RULES:
1. DO NOT score or evaluate candidates.
2. DO NOT calculate overall scores, weights, percentages, or rank candidates.
3. DO NOT decide whether any candidate is good or bad.
4. Extract ONLY requirements that are explicitly stated or strongly implied by the job description.
5. DO NOT invent, hallucinate, or assume requirements not grounded in the text.
6. Categorize each requirement into EXACTLY ONE of the following 5 allowed categories:
   - "technical_skills": Programming languages, software architectures, algorithms, core engineering principles, domain expertise.
   - "experience": Years of experience, seniority level, industry background, leadership/mentorship track record.
   - "education": Degrees, academic majors, required licenses, or academic credentials.
   - "responsibilities": Key day-to-day duties, operational deliverables, cross-functional execution, project management.
   - "tools": Specific software applications, cloud platforms (AWS, GCP, Azure), databases, libraries, developer tooling, CI/CD systems.
7. For each criterion, generate 2 to 6 concise, highly relevant `keywords` (synonyms, technical terms, acronyms) useful for resume matching.
8. Consolidate and avoid duplicate or redundant criteria.
9. Output MUST be valid JSON conforming strictly to the schema below. Do NOT output any markdown commentary outside the JSON.

### REQUIRED JSON SCHEMA:
{
  "criteria": [
    {
      "name": "Short, clear requirement title (e.g., 'Python Backend Engineering')",
      "category": "technical_skills | experience | education | responsibilities | tools",
      "description": "Comprehensive description of what is required according to the job description.",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }
  ]
}

### JOB DESCRIPTION:
\"\"\"
{job_description}
\"\"\"
"""


# ==============================================================================
# Extractor Implementation
# ==============================================================================

class CriterionExtractor:
    """
    Extracts structured hiring criteria from job descriptions using the Google GenAI SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional[genai.Client] = None,
    ) -> None:
        """
        Initialize the CriterionExtractor.

        Args:
            api_key: Gemini API key. Defaults to environment variable GEMINI_API_KEY.
            model_name: Gemini model name. Defaults to environment variable GEMINI_MODEL or 'gemini-2.5-flash'.
            client: Pre-configured genai.Client instance (useful for mocking in tests).
        """
        self.model_name = model_name or get_gemini_model()

        if client is not None:
            self.client = client
        else:
            resolved_key = api_key or get_gemini_api_key()
            if not resolved_key or resolved_key.strip() in ("", "your_gemini_api_key_here"):
                raise ConfigurationError(
                    "Gemini API key is not configured. Please set GEMINI_API_KEY in .env or pass api_key."
                )
            self.client = genai.Client(api_key=resolved_key)

    def _build_prompt(self, job_description: str) -> str:
        """Format the job description into the extraction prompt."""
        return CRITERION_EXTRACTION_PROMPT_TEMPLATE.replace("{job_description}", job_description)

    def _clean_json_text(self, text: str) -> str:
        """
        Strip markdown code block markers (e.g. ```json ... ```) from model output.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    def _call_gemini(self, prompt: str) -> str:
        """
        Execute the Gemini API request. Isolated for mocking and clean error translation.

        Raises:
            GeminiTimeoutError: If the request times out.
            GeminiAPIError: If an API, network, or communication error occurs.
            EmptyGeminiResponseError: If the response is empty, None, or contains no candidates.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise GeminiTimeoutError(f"Gemini API request timed out: {exc}") from exc
        except errors.APIError as exc:
            raise GeminiAPIError(f"Gemini API error: {exc}") from exc
        except (httpx.HTTPError, httpx.RequestError) as exc:
            raise GeminiAPIError(f"Gemini network communication error: {exc}") from exc
        except Exception as exc:
            err_msg = str(exc).lower()
            if "timeout" in err_msg or "timed out" in err_msg:
                raise GeminiTimeoutError(f"Gemini API request timed out: {exc}") from exc
            raise GeminiAPIError(f"Unexpected error calling Gemini API: {exc}") from exc

        if response is None:
            raise EmptyGeminiResponseError("Gemini API returned None.")

        # Attempt to retrieve text from response
        text = getattr(response, "text", None)
        if text is None:
            # Check for candidate contents
            candidates = getattr(response, "candidates", None)
            if not candidates:
                raise EmptyGeminiResponseError("Gemini API response contained no candidates or text.")
            try:
                parts = candidates[0].content.parts
                text = "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
            except Exception:
                pass

        if not text or not text.strip():
            raise EmptyGeminiResponseError("Gemini API returned an empty or whitespace-only response.")

        return text.strip()

    def _parse_and_validate(self, response_text: str) -> CriterionExtractionResponse:
        """
        Parse raw response string as JSON and validate against CriterionExtractionResponse.

        Raises:
            InvalidExtractionJSONError: If the string cannot be parsed as JSON.
            ExtractionSchemaValidationError: If the JSON structure does not match the Pydantic schema.
        """
        cleaned = self._clean_json_text(response_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise InvalidExtractionJSONError(
                f"Failed to parse Gemini response as JSON: {exc}"
            ) from exc

        # Handle case where Gemini returns a list directly instead of {"criteria": [...]}
        if isinstance(data, list):
            data = {"criteria": data}
        elif not isinstance(data, dict):
            raise InvalidExtractionJSONError(
                f"Expected JSON object or list from Gemini, received {type(data).__name__}."
            )

        try:
            validated = CriterionExtractionResponse.model_validate(data)
        except ValidationError as exc:
            raise ExtractionSchemaValidationError(
                f"Gemini response failed Pydantic schema validation: {exc}"
            ) from exc

        return validated.deduplicate_criteria()

    def extract_criteria(self, job_description: str) -> CriterionExtractionResponse:
        """
        Extract structured hiring criteria from a job description text.

        Args:
            job_description: Plain text job description.

        Returns:
            CriterionExtractionResponse containing validated and deduplicated criteria.

        Raises:
            InvalidJobDescriptionError: If the job description is empty or blank.
            GeminiTimeoutError: If the Gemini call times out.
            GeminiAPIError: If the Gemini call fails.
            EmptyGeminiResponseError: If Gemini returns an empty response.
            InvalidExtractionJSONError: If the response is not valid JSON.
            ExtractionSchemaValidationError: If the response fails schema validation.
        """
        if not job_description or not job_description.strip():
            raise InvalidJobDescriptionError("Job description cannot be empty or whitespace-only.")

        prompt = self._build_prompt(job_description.strip())
        raw_response = self._call_gemini(prompt)
        return self._parse_and_validate(raw_response)


def extract_job_criteria(
    job_description: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> CriterionExtractionResponse:
    """
    Convenience wrapper to extract structured criteria from a job description.
    """
    extractor = CriterionExtractor(api_key=api_key, model_name=model_name, client=client)
    return extractor.extract_criteria(job_description)
