from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CriterionCategory(str, Enum):
    """Allowed categories for extracted job criteria."""
    TECHNICAL_SKILLS = "technical_skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    RESPONSIBILITIES = "responsibilities"
    TOOLS = "tools"


class Criterion(BaseModel):
    """A single structured hiring criterion extracted from a job description."""
    name: str = Field(
        ...,
        description="Concise name of the requirement (e.g., 'Python Backend Development', 'Cloud Infrastructure')",
        min_length=1,
    )
    category: CriterionCategory = Field(
        ...,
        description="Category of the criterion: technical_skills, experience, education, responsibilities, or tools",
    )
    description: str = Field(
        ...,
        description="Detailed requirement description as specified or implied in the job description",
        min_length=1,
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Useful matching keywords, technologies, skills, or phrases associated with this criterion",
    )

    @field_validator("name", "description", mode="before")
    @classmethod
    def strip_text(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
        return v

    @field_validator("keywords", mode="before")
    @classmethod
    def normalize_keywords(cls, v: list) -> List[str]:
        if not isinstance(v, list):
            return []
        seen = set()
        normalized = []
        for item in v:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned.lower() not in seen:
                    seen.add(cleaned.lower())
                    normalized.append(cleaned)
        return normalized


class CriterionExtractionResponse(BaseModel):
    """Container for the structured criteria extracted by Gemini."""
    criteria: List[Criterion] = Field(
        ...,
        description="List of extracted hiring criteria",
    )

    def deduplicate_criteria(self) -> "CriterionExtractionResponse":
        """
        Deduplicate criteria by (category, normalized name).
        Merges keywords if duplicate criteria are encountered.
        """
        unique_criteria: dict[tuple[str, str], Criterion] = {}
        for c in self.criteria:
            key = (c.category.value, c.name.strip().lower())
            if key not in unique_criteria:
                unique_criteria[key] = c
            else:
                # Merge keywords
                existing = unique_criteria[key]
                combined_kw = existing.keywords + [
                    k for k in c.keywords if k.lower() not in {ek.lower() for ek in existing.keywords}
                ]
                unique_criteria[key] = Criterion(
                    name=existing.name,
                    category=existing.category,
                    description=existing.description,
                    keywords=combined_kw,
                )
        return CriterionExtractionResponse(criteria=list(unique_criteria.values()))


class CriterionMatchResult(BaseModel):
    """Result of matching a single job criterion against a resume."""
    criterion_name: str = Field(..., description="Name of the criterion being evaluated")
    criterion_category: Optional[CriterionCategory] = Field(
        None, description="Category of the criterion"
    )
    semantic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score normalized to [0.0, 1.0]",
    )
    raw_semantic_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Raw cosine similarity in [-1.0, 1.0] before clamping",
    )
    keyword_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of criterion keywords found in the resume [0.0, 1.0]",
    )
    matched_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords found in the resume",
    )
    unmatched_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords not found in the resume",
    )
    evidence: str = Field(
        default="",
        description="Resume text chunk that produced the strongest semantic match",
    )


class ResumeMatchResult(BaseModel):
    """Aggregate result of matching all criteria against a resume."""
    matches: List[CriterionMatchResult] = Field(
        default_factory=list,
        description="Individual criterion match results",
    )


class ScoredCriterion(BaseModel):
    """Criterion evaluated with deterministic score and reasoning."""
    name: str = Field(..., description="Name of the criterion")
    category: CriterionCategory = Field(..., description="Category of the criterion")
    score: float = Field(..., ge=0.0, le=100.0, description="Criterion score scaled from 0.0 to 100.0")
    semantic_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score [0.0, 1.0]")
    keyword_score: float = Field(..., ge=0.0, le=1.0, description="Keyword evidence score [0.0, 1.0]")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords found in the resume")
    evidence: str = Field(default="", description="Resume chunk serving as evidence")
    reason: str = Field(..., description="Deterministic explanation of the score")


class ScoringResponse(BaseModel):
    """API response containing overall fit score and per-criterion evaluations."""
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall candidate match score [0.0, 100.0]")
    criteria: List[ScoredCriterion] = Field(..., description="List of scored criteria evaluations")
    category_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Aggregated score per category",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Application health status")
