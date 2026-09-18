"""
Calibration Test Suite.

Demonstrates that two similarly qualified candidates (Resume A and Resume B)
produce close scores without implausible divergence (gap <= max_similar_resume_gap),
while a materially weaker candidate (Resume C) scores substantially lower.

Uses ONE fixed job description and ONE fixed extracted criteria set so that
Gemini extraction variance does not affect the deterministic measurement.
"""

from pathlib import Path
import pytest

from app.config import get_max_similar_resume_gap
from app.models.schemas import Criterion, CriterionCategory
from app.services.extractor import extract_resume
from app.services.matcher import EmbeddingModelManager, ResumeMatcher
from app.services.scorer import ScoringEngine

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "samples" / "calibration"

# Fixed, realistic criteria set extracted from samples/sample_job_description.txt
FIXED_CALIBRATION_CRITERIA = [
    Criterion(
        name="Python & FastAPI Microservices",
        category=CriterionCategory.TECHNICAL_SKILLS,
        description="Design and build production-grade backend microservices using Python and FastAPI.",
        keywords=["Python", "FastAPI", "Microservices"],
    ),
    Criterion(
        name="NLP & Embedding Pipelines",
        category=CriterionCategory.TECHNICAL_SKILLS,
        description="Build document extraction and semantic matching systems utilizing modern NLP and embedding models.",
        keywords=["NLP", "Embeddings", "Semantic Search"],
    ),
    Criterion(
        name="Senior Engineering Experience",
        category=CriterionCategory.EXPERIENCE,
        description="5+ years of software engineering experience with strong proficiency in Python.",
        keywords=["5+ years", "Senior"],
    ),
    Criterion(
        name="Computer Science Degree",
        category=CriterionCategory.EDUCATION,
        description="Bachelor's degree in Computer Science or equivalent practical experience.",
        keywords=["Bachelor's", "Computer Science"],
    ),
    Criterion(
        name="Clean Code & Test Coverage",
        category=CriterionCategory.RESPONSIBILITIES,
        description="Write clean, maintainable, modular code with high test coverage using pytest.",
        keywords=["pytest", "Test Coverage"],
    ),
    Criterion(
        name="Database & Container Tooling",
        category=CriterionCategory.TOOLS,
        description="Hands-on experience with Docker, containerization, PostgreSQL, Redis, and cloud infrastructure.",
        keywords=["Docker", "PostgreSQL", "Redis", "AWS"],
    ),
]


def score_candidate_resume(file_path: Path, matcher: ResumeMatcher, engine: ScoringEngine):
    """Run full matching and scoring pipeline on a resume file."""
    text = extract_resume(file_path)
    match_result = matcher.match_criteria(text, FIXED_CALIBRATION_CRITERIA)
    evaluation = engine.evaluate(match_result.matches, FIXED_CALIBRATION_CRITERIA)
    return evaluation


def test_calibration_three_samples():
    """
    Measure calibration across three sample resumes:
    - Resume A: Senior Backend AI Engineer (Alex Morgan)
    - Resume B: Senior Backend AI Engineer (Jordan Lee, similarly qualified)
    - Resume C: Junior Frontend/Design (Taylor Brooks, materially different)
    """
    resume_a_path = CALIBRATION_DIR / "calibration_resume_a.txt"
    resume_b_path = CALIBRATION_DIR / "calibration_resume_b.txt"
    resume_c_path = CALIBRATION_DIR / "calibration_resume_c.txt"

    assert resume_a_path.is_file(), f"Missing {resume_a_path}"
    assert resume_b_path.is_file(), f"Missing {resume_b_path}"
    assert resume_c_path.is_file(), f"Missing {resume_c_path}"

    # Use the real sentence-transformers model (loaded once from local cache)
    real_model = EmbeddingModelManager.get_model("all-MiniLM-L6-v2")
    matcher = ResumeMatcher(model=real_model)
    engine = ScoringEngine()

    eval_a = score_candidate_resume(resume_a_path, matcher, engine)
    eval_b = score_candidate_resume(resume_b_path, matcher, engine)
    eval_c = score_candidate_resume(resume_c_path, matcher, engine)

    score_a = eval_a.overall_score
    score_b = eval_b.overall_score
    score_c = eval_c.overall_score

    similar_resume_score_gap = round(abs(score_a - score_b), 2)
    max_allowed_gap = get_max_similar_resume_gap()

    # Log actual scores for documentation and verification
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS:")
    print(f"  Resume A (Alex Morgan) score   : {score_a:.1f}")
    print(f"  Resume B (Jordan Lee) score    : {score_b:.1f}")
    print(f"  Resume C (Taylor Brooks) score : {score_c:.1f}")
    print(f"  Observed Similar Gap (|A - B|) : {similar_resume_score_gap:.1f}")
    print(f"  Configured Acceptance Threshold: {max_allowed_gap:.1f}")
    print("=" * 60)

    # 1. Acceptance threshold: Similar resumes must not diverge beyond threshold
    assert similar_resume_score_gap <= max_allowed_gap, (
        f"Calibration failed: Similar resumes A and B scored {score_a} and {score_b} "
        f"(gap {similar_resume_score_gap:.1f} > threshold {max_allowed_gap:.1f})"
    )

    # 2. Resumes A and B must both score well above the partial match threshold (>= 50.0)
    assert score_a >= 50.0, f"Resume A score ({score_a}) unexpectedly low."
    assert score_b >= 50.0, f"Resume B score ({score_b}) unexpectedly low."

    # 3. Material difference: Resume C must score significantly lower than A and B
    assert score_a - score_c >= 25.0, (
        f"Resume C ({score_c}) did not score materially lower than Resume A ({score_a})"
    )
    assert score_b - score_c >= 20.0, (
        f"Resume C ({score_c}) did not score materially lower than Resume B ({score_b})"
    )

    # 4. Criterion-level verification: verify Resume C is specifically weaker on technical skills & tools
    c_scores_by_name = {c.name: c.score for c in eval_c.criteria}
    a_scores_by_name = {c.name: c.score for c in eval_a.criteria}

    # Python & FastAPI Microservices: C should score far lower
    assert a_scores_by_name["Python & FastAPI Microservices"] - c_scores_by_name["Python & FastAPI Microservices"] > 40.0
    # Database & Container Tooling: C should score far lower
    assert a_scores_by_name["Database & Container Tooling"] - c_scores_by_name["Database & Container Tooling"] > 30.0
