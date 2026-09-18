"""
Unit and integration tests for the Resume-to-Criterion Matching Engine.

Deterministic unit tests utilize mock embedding models to evaluate cosine similarity,
chunking, keyword matching, and edge cases with speed and precision.
One integration test validates the real sentence-transformers (all-MiniLM-L6-v2) model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.models.schemas import (
    Criterion,
    CriterionCategory,
    CriterionMatchResult,
    ResumeMatchResult,
)
from app.services.matcher import (
    EmbeddingModelError,
    EmbeddingModelManager,
    InvalidCriterionError,
    InvalidInputError,
    ResumeMatcher,
    chunk_resume_text,
    compute_cosine_similarity,
    match_keywords,
    match_resume_to_criteria,
    normalize_cosine_similarity,
)


# ==============================================================================
# Mock Embedding Model
# ==============================================================================

class DeterministicMockEmbeddingModel:
    """
    Mock embedding model that maps exact text queries to specified numpy vectors.
    Calculates actual cosine similarities without neural network inference.
    """
    def __init__(
        self,
        query_map: dict[str, np.ndarray] | None = None,
        default_dim: int = 4,
    ) -> None:
        self.query_map = query_map or {}
        self.default_dim = default_dim

    def encode(self, texts, convert_to_numpy: bool = True):
        if isinstance(texts, str):
            if texts in self.query_map:
                return self.query_map[texts]
            # Fallback zero vector
            return np.zeros(self.default_dim, dtype=np.float32)

        results = []
        for text in texts:
            if text in self.query_map:
                results.append(self.query_map[text])
            else:
                results.append(np.zeros(self.default_dim, dtype=np.float32))
        return np.array(results, dtype=np.float32)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def python_criterion() -> Criterion:
    return Criterion(
        name="Python Backend Development",
        category=CriterionCategory.TECHNICAL_SKILLS,
        description="Experience building backend APIs with Python and FastAPI.",
        keywords=["Python", "FastAPI", "API"],
    )


# ==============================================================================
# 1. Semantic Similarity & Normalization Tests
# ==============================================================================

def test_cosine_similarity_calculation():
    """Verify raw cosine similarity calculation for known geometric vectors."""
    # Orthogonal vectors: cosine = 0.0
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert compute_cosine_similarity(v1, v2) == 0.0

    # Parallel vectors: cosine = 1.0
    v3 = np.array([2.0, 0.0, 0.0])
    assert compute_cosine_similarity(v1, v3) == 1.0

    # Opposite vectors: cosine = -1.0
    v4 = np.array([-1.0, 0.0, 0.0])
    assert compute_cosine_similarity(v1, v4) == -1.0

    # Known 45 degree angle: cos(45 deg) = sqrt(2)/2 ~= 0.7071
    v5 = np.array([1.0, 1.0, 0.0])
    expected = 1.0 / np.sqrt(2.0)
    assert pytest.approx(compute_cosine_similarity(v1, v5), rel=1e-4) == expected

    # Zero vector handling
    v_zero = np.array([0.0, 0.0, 0.0])
    assert compute_cosine_similarity(v1, v_zero) == 0.0


def test_normalize_cosine_similarity_explicit_behavior():
    """
    Test explicit transformation of raw cosine similarity into [0.0, 1.0].
    Positive values must NOT be distorted, and negative values must clamp to 0.0.
    """
    # Positive values preserved directly
    assert normalize_cosine_similarity(1.0) == 1.0
    assert normalize_cosine_similarity(0.85) == 0.85
    assert normalize_cosine_similarity(0.50) == 0.50
    assert normalize_cosine_similarity(0.0) == 0.0

    # Negative values clamp to 0.0 (not inflated to 0.50)
    assert normalize_cosine_similarity(-0.25) == 0.0
    assert normalize_cosine_similarity(-1.0) == 0.0

    # Boundary safety
    assert normalize_cosine_similarity(1.05) == 1.0
    assert normalize_cosine_similarity(-1.5) == 0.0


def test_strong_semantic_match(python_criterion: Criterion):
    """Test strong semantic similarity when a resume chunk aligns closely with the criterion."""
    query = f"{python_criterion.name}: {python_criterion.description}".strip()
    resume_chunk = "Senior Python engineer building microservices with FastAPI and async libraries."

    # Vector alignment: cos(theta) = 0.95
    vec_crit = np.array([1.0, 0.0, 0.0, 0.0])
    vec_chunk = np.array([0.95, np.sqrt(1 - 0.95**2), 0.0, 0.0])

    mock_model = DeterministicMockEmbeddingModel({
        query: vec_crit,
        resume_chunk: vec_chunk,
    })

    matcher = ResumeMatcher(model=mock_model)
    result = matcher.match_criterion(resume_chunk, python_criterion)

    assert result.criterion_name == "Python Backend Development"
    assert pytest.approx(result.semantic_score, rel=1e-3) == 0.95
    assert pytest.approx(result.raw_semantic_score, rel=1e-3) == 0.95
    assert result.evidence == resume_chunk


def test_weak_semantic_match(python_criterion: Criterion):
    """Test weak semantic similarity when resume content is unrelated to criterion."""
    query = f"{python_criterion.name}: {python_criterion.description}".strip()
    resume_chunk = "Lead marketing coordinator organizing public trade shows and press events."

    # Orthogonal vectors: cos(theta) = 0.05
    vec_crit = np.array([1.0, 0.0, 0.0, 0.0])
    vec_chunk = np.array([0.05, np.sqrt(1 - 0.05**2), 0.0, 0.0])

    mock_model = DeterministicMockEmbeddingModel({
        query: vec_crit,
        resume_chunk: vec_chunk,
    })

    matcher = ResumeMatcher(model=mock_model)
    result = matcher.match_criterion(resume_chunk, python_criterion)

    assert pytest.approx(result.semantic_score, rel=1e-3) == 0.05
    assert result.semantic_score < 0.20


# ==============================================================================
# 2. Keyword Evidence Tests
# ==============================================================================

def test_exact_keyword_match(python_criterion: Criterion):
    """Test 100% keyword match when all criterion keywords are present."""
    resume_text = "Proficient in Python and FastAPI for building RESTful API systems."
    score, matched, unmatched = match_keywords(resume_text, python_criterion.keywords)

    assert score == 1.0
    assert set(matched) == {"Python", "FastAPI", "API"}
    assert unmatched == []


def test_partial_keyword_match(python_criterion: Criterion):
    """Test partial keyword match (e.g. 2 out of 3 keywords matched -> 0.6667)."""
    resume_text = "Experienced software engineer with deep Python knowledge designing backend systems."
    score, matched, unmatched = match_keywords(resume_text, python_criterion.keywords)

    assert pytest.approx(score, rel=1e-3) == 0.3333
    assert matched == ["Python"]
    assert set(unmatched) == {"FastAPI", "API"}


def test_no_keyword_match(python_criterion: Criterion):
    """Test zero keyword match when no keywords appear in the resume text."""
    resume_text = "Database administrator with deep Oracle SQL and Linux administration experience."
    score, matched, unmatched = match_keywords(resume_text, python_criterion.keywords)

    assert score == 0.0
    assert matched == []
    assert set(unmatched) == {"Python", "FastAPI", "API"}


def test_case_insensitive_and_whitespace_tolerant_matching():
    """Test that keyword matching is strictly case-insensitive and handles spacing."""
    resume_text = "Built services with python, FASTAPI, and restful   api architecture."
    keywords = ["PYTHON", "FastAPI", "restful api"]

    score, matched, unmatched = match_keywords(resume_text, keywords)
    assert score == 1.0
    assert len(matched) == 3
    assert unmatched == []


def test_word_boundary_keyword_matching():
    """Test that keywords match whole words or technical symbols, not substrings."""
    resume_text = "Developed pythonic algorithms in C# and TypeScript."
    keywords = ["Python", "C#", "Java"]

    score, matched, unmatched = match_keywords(resume_text, keywords)
    # 'pythonic' should NOT match 'Python'
    # 'C#' should match 'C#'
    # 'Java' should not match
    assert "C#" in matched
    assert "Python" in unmatched
    assert "Java" in unmatched
    assert pytest.approx(score, rel=1e-3) == 1 / 3


# ==============================================================================
# 3. Resume Chunking & Multi-Chunk Selection Tests
# ==============================================================================

def test_chunk_resume_text_paragraph_aware():
    """Test that chunking preserves paragraph structure up to max_chunk_size."""
    para1 = "Paragraph 1: Background in software development."
    para2 = "Paragraph 2: Detailed experience with distributed systems."
    para3 = "Paragraph 3: Education and degrees."
    full_text = f"{para1}\n\n{para2}\n\n{para3}"

    # Generous chunk size: all paragraphs fit in one chunk
    chunks = chunk_resume_text(full_text, max_chunk_size=500)
    assert len(chunks) == 1
    assert para1 in chunks[0]
    assert para3 in chunks[0]

    # Small chunk size: forces paragraph separation
    chunks_small = chunk_resume_text(full_text, max_chunk_size=70, overlap=0)
    assert len(chunks_small) >= 3
    assert para1 in chunks_small[0]
    assert para2 in chunks_small[1]


def test_multiple_resume_chunks_selects_best_evidence(python_criterion: Criterion):
    """Test that the matching engine evaluates all chunks and returns the best matching chunk."""
    chunk_weak = "Early Career: Worked in sales and business development for 2 years."
    chunk_strong = "Backend Role: Built high-concurrency Python microservices using FastAPI."
    resume_text = f"{chunk_weak}\n\n{chunk_strong}"

    query = f"{python_criterion.name}: {python_criterion.description}".strip()

    vec_crit = np.array([1.0, 0.0, 0.0])
    vec_weak = np.array([0.1, 0.99, 0.0])
    vec_strong = np.array([0.88, 0.20, 0.0])

    mock_model = DeterministicMockEmbeddingModel({
        query: vec_crit,
        chunk_weak: vec_weak,
        chunk_strong: vec_strong,
    }, default_dim=3)

    matcher = ResumeMatcher(model=mock_model, chunk_size=100, chunk_overlap=0)
    result = matcher.match_criterion(resume_text, python_criterion)

    # Strongest chunk should be selected
    assert result.evidence == chunk_strong
    # Cosine similarity for vec_strong
    expected_sim = compute_cosine_similarity(vec_crit, vec_strong)
    assert pytest.approx(result.semantic_score, rel=1e-3) == normalize_cosine_similarity(expected_sim)
    assert pytest.approx(result.raw_semantic_score, rel=1e-3) == expected_sim


# ==============================================================================
# 4. Edge Cases & Input Validation Tests
# ==============================================================================

def test_empty_resume(python_criterion: Criterion):
    """Test matching against an empty or whitespace-only resume."""
    mock_model = DeterministicMockEmbeddingModel()
    matcher = ResumeMatcher(model=mock_model)

    result = matcher.match_criterion("", python_criterion)
    assert result.semantic_score == 0.0
    assert result.raw_semantic_score == 0.0
    assert result.keyword_score == 0.0
    assert result.matched_keywords == []
    assert result.evidence == ""

    result_spaces = matcher.match_criterion("   \n\t  ", python_criterion)
    assert result_spaces.semantic_score == 0.0
    assert result_spaces.keyword_score == 0.0


def test_empty_keyword_list():
    """Test matching when a criterion has an empty keyword list."""
    criterion_no_kw = Criterion(
        name="Communication",
        category=CriterionCategory.RESPONSIBILITIES,
        description="Collaborate cross-functionally across departments.",
        keywords=[],
    )
    query = f"{criterion_no_kw.name}: {criterion_no_kw.description}".strip()
    resume_text = "Collaborated across multiple engineering teams to ship features."

    vec = np.array([1.0, 0.0])
    mock_model = DeterministicMockEmbeddingModel({query: vec, resume_text: vec}, default_dim=2)
    matcher = ResumeMatcher(model=mock_model)

    result = matcher.match_criterion(resume_text, criterion_no_kw)
    assert result.keyword_score == 0.0
    assert result.matched_keywords == []
    assert result.unmatched_keywords == []
    assert result.semantic_score == 1.0


def test_invalid_criterion_input():
    """Test that passing an invalid criterion raises InvalidCriterionError."""
    matcher = ResumeMatcher(model=DeterministicMockEmbeddingModel())

    with pytest.raises(InvalidCriterionError):
        matcher.match_criterion("Resume text", "Not a criterion object")  # type: ignore

    with pytest.raises(InvalidCriterionError):
        matcher.match_criterion("Resume text", None)  # type: ignore

    with pytest.raises(InvalidCriterionError):
        # Criterion object with empty name bypassing validation
        invalid_c = Criterion.model_construct(
            name="   ",
            category=CriterionCategory.TOOLS,
            description="Docker containers",
            keywords=[],
        )
        matcher.match_criterion("Resume text", invalid_c)


def test_invalid_criteria_list():
    """Test that passing None for criteria raises InvalidInputError."""
    matcher = ResumeMatcher(model=DeterministicMockEmbeddingModel())

    with pytest.raises(InvalidInputError):
        matcher.match_criteria("Resume text", None)  # type: ignore


def test_embedding_model_inference_failure(python_criterion: Criterion):
    """Test that embedding model exceptions are translated into EmbeddingModelError."""
    failing_model = MagicMock()
    failing_model.encode.side_effect = RuntimeError("CUDA out of memory or inference crash")

    matcher = ResumeMatcher(model=failing_model)
    with pytest.raises(EmbeddingModelError) as exc_info:
        matcher.match_criterion("Valid resume text", python_criterion)
    assert "Embedding inference failed" in str(exc_info.value)


def test_embedding_model_manager_load_failure():
    """Test that model load failures raise EmbeddingModelError."""
    EmbeddingModelManager.clear()
    with patch("sentence_transformers.SentenceTransformer", side_effect=Exception("Model path not found")):
        with pytest.raises(EmbeddingModelError) as exc_info:
            EmbeddingModelManager.get_model("nonexistent-model-xyz")
        assert "Failed to load sentence-transformers model" in str(exc_info.value)
    EmbeddingModelManager.clear()


# ==============================================================================
# 5. Batch Matching & Convenience Wrapper Tests
# ==============================================================================

def test_match_criteria_batch():
    """Test matching multiple criteria against a resume in a single batch."""
    crit1 = Criterion(
        name="Python Skills",
        category=CriterionCategory.TECHNICAL_SKILLS,
        description="Python backend programming.",
        keywords=["Python"],
    )
    crit2 = Criterion(
        name="AWS Cloud",
        category=CriterionCategory.TOOLS,
        description="Deploying applications to AWS cloud infrastructure.",
        keywords=["AWS"],
    )

    resume_text = "Proficient in Python. Extensive experience deploying containerized services on AWS."

    q1 = f"{crit1.name}: {crit1.description}".strip()
    q2 = f"{crit2.name}: {crit2.description}".strip()

    vec_crit1 = np.array([1.0, 0.0])
    vec_crit2 = np.array([0.0, 1.0])
    vec_resume = np.array([0.8, 0.6])

    mock_model = DeterministicMockEmbeddingModel({
        q1: vec_crit1,
        q2: vec_crit2,
        resume_text: vec_resume,
    }, default_dim=2)

    matcher = ResumeMatcher(model=mock_model)
    batch_result = matcher.match_criteria(resume_text, [crit1, crit2])

    assert isinstance(batch_result, ResumeMatchResult)
    assert len(batch_result.matches) == 2
    assert batch_result.matches[0].criterion_name == "Python Skills"
    assert batch_result.matches[1].criterion_name == "AWS Cloud"
    assert batch_result.matches[0].keyword_score == 1.0
    assert batch_result.matches[1].keyword_score == 1.0


def test_convenience_wrapper_function(python_criterion: Criterion):
    """Test match_resume_to_criteria top-level convenience wrapper."""
    mock_model = DeterministicMockEmbeddingModel(default_dim=2)
    result = match_resume_to_criteria(
        "Python developer with FastAPI experience.",
        [python_criterion],
        model=mock_model,
    )
    assert isinstance(result, ResumeMatchResult)
    assert len(result.matches) == 1


# ==============================================================================
# 6. Real Model Integration Test (all-MiniLM-L6-v2)
# ==============================================================================

def test_real_all_minilm_integration(python_criterion: Criterion):
    """
    Integration test using the locally cached all-MiniLM-L6-v2 model.
    Verifies real embedding generation, cosine similarity, and chunk matching without network access.
    """
    real_model = EmbeddingModelManager.get_model("all-MiniLM-L6-v2")
    matcher = ResumeMatcher(model=real_model)

    resume_text = (
        "Professional Summary:\n"
        "Senior Software Engineer specializing in backend systems using Python, FastAPI, and PostgreSQL.\n\n"
        "Experience:\n"
        "Designed and maintained asynchronous microservices and REST API systems for enterprise clients.\n\n"
        "Education:\n"
        "Bachelor of Science in Computer Science."
    )

    result = matcher.match_criterion(resume_text, python_criterion)

    # Validate output schema types and ranges
    assert isinstance(result, CriterionMatchResult)
    assert result.criterion_name == "Python Backend Development"
    assert 0.0 <= result.semantic_score <= 1.0
    assert -1.0 <= result.raw_semantic_score <= 1.0
    # Relevant chunk should have substantial semantic similarity (> 0.40)
    assert result.semantic_score > 0.40
    # Keywords 'Python', 'FastAPI', 'API' should all be matched
    assert "Python" in result.matched_keywords
    assert "FastAPI" in result.matched_keywords
    assert "API" in result.matched_keywords
    assert result.keyword_score == 1.0
    assert len(result.evidence) > 0
