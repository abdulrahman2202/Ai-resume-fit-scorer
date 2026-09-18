from enum import Enum
from typing import List
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
