"""
Unit tests for the Deterministic Scoring Engine.

Tests formula calculations, category weighting, missing-category renormalization,
weight validation, 0-100 bounding, and deterministic reasoning generation.
"""

import pytest

from app.config import ConfigurationError
from app.models.schemas import (
    Criterion,
    CriterionCategory,
    CriterionMatchResult,
    ScoredCriterion,
    ScoringResponse,
)
from app.services.scorer import (
    InvalidWeightConfigurationError,
    ScoringEngine,
    calculate_criterion_score,
    calculate_overall_score,
    generate_deterministic_reason,
)


# ==============================================================================
# 1. Criterion Score Calculation Formula
# ==============================================================================

def test_criterion_score_formula():
    """
    Test criterion_score = (semantic_weight * semantic_score + keyword_weight * keyword_score) * 100.
    With default weights 0.60 semantic and 0.40 keyword:
    sem=0.80, kw=1.00 -> (0.60*0.80 + 0.40*1.00) * 100 = (0.48 + 0.40) * 100 = 88.0
    """
    score = calculate_criterion_score(
        semantic_score=0.80,
        keyword_score=1.00,
        semantic_weight=0.60,
        keyword_weight=0.40,
    )
    assert score == 88.0


def test_criterion_score_boundary_zero_and_hundred():
    """Verify scores are strictly bounded in [0.0, 100.0]."""
    score_zero = calculate_criterion_score(0.0, 0.0)
    assert score_zero == 0.0

    score_max = calculate_criterion_score(1.0, 1.0)
    assert score_max == 100.0


def test_criterion_score_custom_weights():
    """Verify criterion scoring respects custom weight distributions."""
    score = calculate_criterion_score(
        semantic_score=0.50,
        keyword_score=0.50,
        semantic_weight=0.70,
        keyword_weight=0.30,
    )
    assert score == 50.0


# ==============================================================================
# 2. Weight Configuration Validation
# ==============================================================================

def test_invalid_scoring_weights_sum():
    """Raise InvalidWeightConfigurationError when scoring weights sum != 1.0."""
    with pytest.raises(InvalidWeightConfigurationError):
        ScoringEngine(
            scoring_weights={"semantic_weight": 0.50, "keyword_weight": 0.30}  # Sum = 0.80
        )


def test_negative_scoring_weights():
    """Raise InvalidWeightConfigurationError when weights are negative."""
    with pytest.raises(InvalidWeightConfigurationError):
        ScoringEngine(
            scoring_weights={"semantic_weight": 1.20, "keyword_weight": -0.20}
        )


def test_invalid_category_weights_sum():
    """Raise InvalidWeightConfigurationError when category weights sum != 1.0."""
    with pytest.raises(InvalidWeightConfigurationError):
        ScoringEngine(
            category_weights={"technical_skills": 0.50, "tools": 0.20}  # Sum = 0.70
        )


# ==============================================================================
# 3. Category Weighting & Missing-Category Renormalization
# ==============================================================================

def test_category_weighting_all_categories_present():
    """
    Test overall score when all 5 categories are present.
    Configured weights:
    - technical_skills: 0.35, score: 80
    - experience: 0.25, score: 90
    - education: 0.10, score: 100
    - responsibilities: 0.20, score: 70
    - tools: 0.10, score: 60
    Expected: (0.35*80) + (0.25*90) + (0.10*100) + (0.20*70) + (0.10*60)
            = 28.0 + 22.5 + 10.0 + 14.0 + 6.0 = 80.5
    """
    weights = {
        "technical_skills": 0.35,
        "experience": 0.25,
        "education": 0.10,
        "responsibilities": 0.20,
        "tools": 0.10,
    }

    scored_criteria = [
        ScoredCriterion(
            name="Python",
            category=CriterionCategory.TECHNICAL_SKILLS,
            score=80.0,
            semantic_score=0.8,
            keyword_score=0.8,
            matched_keywords=["Python"],
            evidence="...",
            reason="...",
        ),
        ScoredCriterion(
            name="Years of Experience",
            category=CriterionCategory.EXPERIENCE,
            score=90.0,
            semantic_score=0.9,
            keyword_score=0.9,
            matched_keywords=[],
            evidence="...",
            reason="...",
        ),
        ScoredCriterion(
            name="BS in CS",
            category=CriterionCategory.EDUCATION,
            score=100.0,
            semantic_score=1.0,
            keyword_score=1.0,
            matched_keywords=[],
            evidence="...",
            reason="...",
        ),
        ScoredCriterion(
            name="Architecture",
            category=CriterionCategory.RESPONSIBILITIES,
            score=70.0,
            semantic_score=0.7,
            keyword_score=0.7,
            matched_keywords=[],
            evidence="...",
            reason="...",
        ),
        ScoredCriterion(
            name="Docker",
            category=CriterionCategory.TOOLS,
            score=60.0,
            semantic_score=0.6,
            keyword_score=0.6,
            matched_keywords=["Docker"],
            evidence="...",
            reason="...",
        ),
    ]

    overall, cat_scores = calculate_overall_score(scored_criteria, weights)
    assert overall == 80.5
    assert cat_scores["technical_skills"] == 80.0
    assert cat_scores["education"] == 100.0


def test_missing_category_renormalization():
    """
    Verify renormalization when only 2 of 5 categories are present in the JD.
    Base weights:
    - technical_skills: 0.35, score: 80
    - tools: 0.10, score: 100
    Absent categories: experience (0.25), education (0.10), responsibilities (0.20).
    Sum of present base weights = 0.35 + 0.10 = 0.45.
    Renormalized weights:
    - technical_skills: 0.35 / 0.45 = 7/9 (~0.7778)
    - tools: 0.10 / 0.45 = 2/9 (~0.2222)
    Expected overall = (80 * 7/9) + (100 * 2/9) = (560 + 200) / 9 = 760 / 9 = 84.444... -> 84.4
    """
    weights = {
        "technical_skills": 0.35,
        "experience": 0.25,
        "education": 0.10,
        "responsibilities": 0.20,
        "tools": 0.10,
    }

    scored_criteria = [
        ScoredCriterion(
            name="Python",
            category=CriterionCategory.TECHNICAL_SKILLS,
            score=80.0,
            semantic_score=0.8,
            keyword_score=0.8,
            matched_keywords=["Python"],
            evidence="...",
            reason="...",
        ),
        ScoredCriterion(
            name="Docker",
            category=CriterionCategory.TOOLS,
            score=100.0,
            semantic_score=1.0,
            keyword_score=1.0,
            matched_keywords=["Docker"],
            evidence="...",
            reason="...",
        ),
    ]

    overall, cat_scores = calculate_overall_score(scored_criteria, weights)
    assert overall == 84.4
    assert cat_scores == {"technical_skills": 80.0, "tools": 100.0}


def test_multiple_criteria_in_same_category_averaging():
    """Verify criteria within the same category are averaged before weighting."""
    weights = {
        "technical_skills": 0.50,
        "tools": 0.50,
    }

    scored_criteria = [
        ScoredCriterion(
            name="Python",
            category=CriterionCategory.TECHNICAL_SKILLS,
            score=80.0,
            semantic_score=0.8,
            keyword_score=0.8,
            matched_keywords=[],
            evidence="...",
            reason="...",
        ),
        ScoredCriterion(
            name="FastAPI",
            category=CriterionCategory.TECHNICAL_SKILLS,
            score=100.0,
            semantic_score=1.0,
            keyword_score=1.0,
            matched_keywords=[],
            evidence="...",
            reason="...",
        ),
        ScoredCriterion(
            name="Docker",
            category=CriterionCategory.TOOLS,
            score=60.0,
            semantic_score=0.6,
            keyword_score=0.6,
            matched_keywords=[],
            evidence="...",
            reason="...",
        ),
    ]

    overall, cat_scores = calculate_overall_score(scored_criteria, weights)
    # technical_skills avg = (80 + 100) / 2 = 90.0
    # tools avg = 60.0
    # overall = 0.50*90 + 0.50*60 = 45 + 30 = 75.0
    assert cat_scores["technical_skills"] == 90.0
    assert cat_scores["tools"] == 60.0
    assert overall == 75.0


def test_empty_criteria_returns_zero():
    """Empty criteria list produces 0.0 overall score."""
    overall, cat_scores = calculate_overall_score([], {"technical_skills": 1.0})
    assert overall == 0.0
    assert cat_scores == {}


# ==============================================================================
# 4. Deterministic Reasoning Generation
# ==============================================================================

def test_deterministic_reason_full_keywords():
    """Verify reason when all keywords are matched."""
    reason = generate_deterministic_reason(
        semantic_score=0.88,
        keyword_score=1.0,
        matched_keywords=["Python", "FastAPI"],
        total_keywords=["Python", "FastAPI"],
        evidence="Senior engineer with Python and FastAPI background.",
    )
    assert "Strong semantic alignment (0.88)" in reason
    assert "all 2 of 2 required keywords found (Python, FastAPI)" in reason
    assert 'Resume evidence: "Senior engineer with Python and FastAPI background."' in reason


def test_deterministic_reason_partial_keywords():
    """Verify reason when only partial keywords are matched."""
    reason = generate_deterministic_reason(
        semantic_score=0.62,
        keyword_score=0.5,
        matched_keywords=["Python"],
        total_keywords=["Python", "FastAPI"],
        evidence="Experienced in Python.",
    )
    assert "Moderate semantic alignment (0.62)" in reason
    assert "1 of 2 keywords found (Python). Missing: FastAPI" in reason


def test_deterministic_reason_no_keywords():
    """Verify reason when no keywords are required or found."""
    reason = generate_deterministic_reason(
        semantic_score=0.15,
        keyword_score=0.0,
        matched_keywords=[],
        total_keywords=["Docker"],
        evidence="",
    )
    assert "Minimal semantic alignment (0.15)" in reason
    assert "none of 1 required keywords found (missing: Docker)" in reason
    assert "No relevant resume evidence found" in reason


# ==============================================================================
# 5. ScoringEngine End-to-End Evaluation
# ==============================================================================

def test_scoring_engine_evaluate():
    """Test full ScoringEngine evaluation flow."""
    engine = ScoringEngine()

    c1 = Criterion(
        name="Python Backend",
        category=CriterionCategory.TECHNICAL_SKILLS,
        description="Python services",
        keywords=["Python", "FastAPI"],
    )
    m1 = CriterionMatchResult(
        criterion_name="Python Backend",
        criterion_category=CriterionCategory.TECHNICAL_SKILLS,
        semantic_score=0.85,
        raw_semantic_score=0.85,
        keyword_score=1.0,
        matched_keywords=["Python", "FastAPI"],
        unmatched_keywords=[],
        evidence="Python and FastAPI services.",
    )

    response = engine.evaluate(matches=[m1], criteria=[c1])

    assert isinstance(response, ScoringResponse)
    assert 0.0 <= response.overall_score <= 100.0
    assert len(response.criteria) == 1
    assert response.criteria[0].score == 91.0  # 0.60*0.85 + 0.40*1.0 = 0.51 + 0.40 = 0.91 -> 91.0
    assert "Python Backend" in response.criteria[0].name
    assert response.category_scores is not None
    assert "technical_skills" in response.category_scores
