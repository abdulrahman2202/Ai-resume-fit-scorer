"""
Deterministic Scoring Engine for Resume Job Fit Scorer.

Computes criterion-level and overall fit scores using strictly deterministic formulas
derived from semantic similarity and keyword evidence. Category-weighted scores are
renormalized when certain categories are absent from the job description.
Gemini is never used for scoring or reasoning generation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from app.config import (
    ConfigurationError,
    get_category_weights,
    get_scoring_weights,
)
from app.models.schemas import (
    Criterion,
    CriterionCategory,
    CriterionMatchResult,
    ResumeMatchResult,
    ScoredCriterion,
    ScoringResponse,
)


# ==============================================================================
# Custom Exceptions
# ==============================================================================

class ScoringError(Exception):
    """Base exception for scoring errors."""
    pass


class InvalidWeightConfigurationError(ScoringError):
    """Raised when scoring or category weights are invalid or do not sum to 1.0."""
    pass


# ==============================================================================
# Deterministic Reasoning Generator
# ==============================================================================

def generate_deterministic_reason(
    semantic_score: float,
    keyword_score: float,
    matched_keywords: List[str],
    total_keywords: List[str],
    evidence: str,
) -> str:
    """
    Produce factual, deterministic reasoning explaining a criterion score.
    Relies purely on match data; no LLM is consulted.
    """
    # 1. Semantic evaluation
    if semantic_score >= 0.75:
        sem_desc = f"Strong semantic alignment ({semantic_score:.2f})"
    elif semantic_score >= 0.50:
        sem_desc = f"Moderate semantic alignment ({semantic_score:.2f})"
    elif semantic_score >= 0.25:
        sem_desc = f"Partial semantic alignment ({semantic_score:.2f})"
    else:
        sem_desc = f"Minimal semantic alignment ({semantic_score:.2f})"

    # 2. Keyword evidence evaluation
    total_count = len(total_keywords)
    matched_count = len(matched_keywords)
    if total_count > 0:
        if matched_count == total_count:
            kw_desc = f"all {matched_count} of {total_count} required keywords found ({', '.join(matched_keywords)})"
        elif matched_count > 0:
            matched_lower = {k.lower() for k in matched_keywords}
            unmatched = [k for k in total_keywords if k.lower() not in matched_lower]
            kw_desc = f"{matched_count} of {total_count} keywords found ({', '.join(matched_keywords)}). Missing: {', '.join(unmatched)}"
        else:
            kw_desc = f"none of {total_count} required keywords found (missing: {', '.join(total_keywords)})"
    else:
        kw_desc = "no specific keywords required"

    # 3. Evidence excerpt
    if evidence and evidence.strip():
        clean_ev = " ".join(evidence.split())
        if len(clean_ev) > 130:
            clean_ev = clean_ev[:127] + "..."
        ev_desc = f'Resume evidence: "{clean_ev}"'
    else:
        ev_desc = "No relevant resume evidence found"

    return f"{sem_desc}; {kw_desc}. {ev_desc}."


# ==============================================================================
# Scoring Calculation Functions
# ==============================================================================

def calculate_criterion_score(
    semantic_score: float,
    keyword_score: float,
    semantic_weight: float = 0.60,
    keyword_weight: float = 0.40,
) -> float:
    """
    Calculate a single criterion score on a 0.0 to 100.0 scale.

    Formula:
        criterion_score = (semantic_weight * semantic_score + keyword_weight * keyword_score) * 100.0
    """
    raw_score = (semantic_weight * semantic_score + keyword_weight * keyword_score) * 100.0
    clamped = max(0.0, min(100.0, raw_score))
    return round(clamped, 1)


def calculate_overall_score(
    scored_criteria: List[ScoredCriterion],
    category_weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the overall score using category weights with missing-category renormalization.

    Missing Category Renormalization:
    - If a job description only defines criteria for a subset of categories (e.g. technical_skills and tools),
      the weights of the present categories are scaled proportionally so their sum equals exactly 1.0.
    - Example: If present categories have weights 0.35 and 0.10 (sum = 0.45), their renormalized
      weights become 0.35/0.45 (~0.778) and 0.10/0.45 (~0.222).
    - This guarantees that the final overall score accurately reflects 100% of the available criteria
      without penalizing the candidate for categories the employer omitted from the JD.

    Returns:
        Tuple of (overall_score: float, category_scores: Dict[str, float])
    """
    if not scored_criteria:
        return 0.0, {}

    # Group scores by category
    cat_scores_map: Dict[str, List[float]] = {}
    for c in scored_criteria:
        cat_key = c.category.value if hasattr(c.category, "value") else str(c.category)
        cat_scores_map.setdefault(cat_key, []).append(c.score)

    category_scores: Dict[str, float] = {
        cat: round(sum(scores) / len(scores), 1)
        for cat, scores in cat_scores_map.items()
    }

    # Identify weights for categories present in the criteria
    present_weights = {
        cat: category_weights.get(cat, 0.0)
        for cat in category_scores
    }
    total_present_weight = sum(present_weights.values())

    if total_present_weight <= 0.0:
        # Fallback to unweighted mean if weights are zero or unspecified
        simple_avg = sum(category_scores.values()) / len(category_scores)
        return round(simple_avg, 1), category_scores

    # Renormalize weights so their sum equals 1.0
    overall = sum(
        category_scores[cat] * (present_weights[cat] / total_present_weight)
        for cat in category_scores
    )

    return round(max(0.0, min(100.0, overall)), 1), category_scores


# ==============================================================================
# Scoring Engine
# ==============================================================================

class ScoringEngine:
    """
    Deterministic scoring engine applying configurable weights and missing-category
    renormalization.
    """

    def __init__(
        self,
        scoring_weights: Optional[Dict[str, float]] = None,
        category_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.scoring_weights = scoring_weights if scoring_weights is not None else get_scoring_weights()
        self.category_weights = category_weights if category_weights is not None else get_category_weights()
        self.validate_weights()

    def validate_weights(self) -> None:
        """
        Ensure scoring weights and category weights are non-negative and sum to 1.0.
        """
        sem = self.scoring_weights.get("semantic_weight", 0.60)
        kw = self.scoring_weights.get("keyword_weight", 0.40)

        if sem < 0.0 or kw < 0.0:
            raise InvalidWeightConfigurationError("Scoring weights must be non-negative.")

        total_scoring = sem + kw
        if not math.isclose(total_scoring, 1.0, abs_tol=1e-4):
            raise InvalidWeightConfigurationError(
                f"Scoring weights must sum to 1.0, got semantic={sem} + keyword={kw} = {total_scoring:.4f}"
            )

        if not self.category_weights:
            raise InvalidWeightConfigurationError("Category weights dictionary cannot be empty.")

        if any(w < 0.0 for w in self.category_weights.values()):
            raise InvalidWeightConfigurationError("Category weights must be non-negative.")

        total_cat = sum(self.category_weights.values())
        if not math.isclose(total_cat, 1.0, abs_tol=1e-4):
            raise InvalidWeightConfigurationError(
                f"Category weights must sum to 1.0, got {total_cat:.4f}: {self.category_weights}"
            )

    def score_criterion(
        self,
        match: CriterionMatchResult,
        criterion: Criterion,
    ) -> ScoredCriterion:
        """
        Evaluate a single criterion match result into a ScoredCriterion with deterministic reasoning.
        """
        sem_w = self.scoring_weights.get("semantic_weight", 0.60)
        kw_w = self.scoring_weights.get("keyword_weight", 0.40)

        score = calculate_criterion_score(
            semantic_score=match.semantic_score,
            keyword_score=match.keyword_score,
            semantic_weight=sem_w,
            keyword_weight=kw_w,
        )

        reason = generate_deterministic_reason(
            semantic_score=match.semantic_score,
            keyword_score=match.keyword_score,
            matched_keywords=match.matched_keywords,
            total_keywords=criterion.keywords,
            evidence=match.evidence,
        )

        category = match.criterion_category or criterion.category

        return ScoredCriterion(
            name=criterion.name,
            category=category,
            score=score,
            semantic_score=match.semantic_score,
            keyword_score=match.keyword_score,
            matched_keywords=match.matched_keywords,
            evidence=match.evidence,
            reason=reason,
        )

    def evaluate(
        self,
        matches: List[CriterionMatchResult],
        criteria: List[Criterion],
    ) -> ScoringResponse:
        """
        Score all criteria matches and compute overall fit score with category weights.
        """
        if len(matches) != len(criteria):
            raise ScoringError(
                f"Mismatch in count: {len(matches)} matches provided for {len(criteria)} criteria."
            )

        scored_criteria: List[ScoredCriterion] = []
        for match, criterion in zip(matches, criteria):
            scored = self.score_criterion(match, criterion)
            scored_criteria.append(scored)

        overall_score, category_scores = calculate_overall_score(
            scored_criteria,
            self.category_weights,
        )

        return ScoringResponse(
            overall_score=overall_score,
            criteria=scored_criteria,
            category_scores=category_scores,
        )
