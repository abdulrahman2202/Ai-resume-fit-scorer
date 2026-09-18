"""
Resume-to-Criterion Matching Engine.

Evaluates resumes against extracted job criteria using two independent signals:
1. Semantic similarity: Sentence-Transformers (all-MiniLM-L6-v2) with paragraph-aware chunking and cosine similarity.
2. Keyword evidence: Deterministic, case-insensitive, whitespace-tolerant keyword matching.
"""

import logging
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from app.config import (
    get_chunk_overlap,
    get_chunk_size,
    get_embedding_model_name,
)
from app.models.schemas import (
    Criterion,
    CriterionCategory,
    CriterionMatchResult,
    ResumeMatchResult,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exception Hierarchy
# ==============================================================================

class MatchingError(Exception):
    """Base exception for all matching engine errors."""
    pass


class EmbeddingModelError(MatchingError):
    """Raised when the sentence-transformers model fails to load or infer."""
    pass


class InvalidCriterionError(MatchingError):
    """Raised when an invalid criterion object or definition is provided."""
    pass


class InvalidInputError(MatchingError):
    """Raised when input parameters to the matching engine are invalid."""
    pass


# ==============================================================================
# Semantic Similarity & Normalization Functions
# ==============================================================================

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute raw cosine similarity between two 1D embedding vectors.

    Cosine similarity is defined as:
        cos(theta) = (vec_a . vec_b) / (||vec_a||_2 * ||vec_b||_2)

    Returns:
        float: Raw cosine similarity strictly bounded in [-1.0, 1.0].
               Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot_product = float(np.dot(vec_a, vec_b))
    raw_sim = dot_product / (norm_a * norm_b)
    # Clamp against floating point precision drift
    return max(-1.0, min(1.0, raw_sim))


def normalize_cosine_similarity(raw_score: float) -> float:
    """
    Convert raw cosine similarity (in [-1.0, 1.0]) to a normalized score in [0.0, 1.0].

    Transformation & Range Behavior:
    - In sentence embeddings (e.g. all-MiniLM-L6-v2), positive cosine similarity
      directly measures semantic alignment.
    - Zero represents orthogonality (unrelated concepts), and negative values indicate
      semantic opposition or noise.
    - Rather than applying an affine transform `(cos + 1) / 2`—which would artificially
      distort an unrelated match (cosine 0.0) into an undeserved 50% score (0.50)—we clamp
      negative values to 0.0:
          normalized = max(0.0, min(1.0, float(raw_score)))
    - This preserves positive raw cosine values without distortion, while ensuring
      the score remains strictly in the [0.0, 1.0] interval.
    """
    return max(0.0, min(1.0, float(raw_score)))


# ==============================================================================
# Simple, Maintainable Text Chunking
# ==============================================================================

def chunk_resume_text(
    resume_text: str,
    max_chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split resume text into paragraph-aware, coherent text chunks.

    Approach:
    1. Splits resume text by paragraph breaks (double newlines '\\n\\n') or single newlines.
    2. Sequentially bundles adjacent paragraphs until adding another would exceed max_chunk_size.
    3. Retains up to `overlap` characters from the previous chunk when starting a new chunk.
    4. Oversized paragraphs exceeding `max_chunk_size` are sliced at max_chunk_size boundaries.
    5. Strips whitespace and discards empty chunks.
    """
    if not resume_text or not resume_text.strip():
        return []

    if max_chunk_size <= 0:
        raise InvalidInputError(f"max_chunk_size must be positive, got {max_chunk_size}")

    normalized_text = resume_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split into logical paragraphs
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized_text) if b.strip()]
    if not raw_blocks:
        return []

    chunks: List[str] = []
    current_chunk_parts: List[str] = []
    current_length = 0

    for block in raw_blocks:
        # If a single paragraph is longer than max_chunk_size, slice it
        if len(block) > max_chunk_size:
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts).strip())
                current_chunk_parts = []
                current_length = 0

            start = 0
            step = max(1, max_chunk_size - overlap) if overlap < max_chunk_size else max_chunk_size
            while start < len(block):
                end = min(start + max_chunk_size, len(block))
                sub_block = block[start:end].strip()
                if sub_block:
                    chunks.append(sub_block)
                if end == len(block):
                    break
                start += step
            continue

        added_len = len(block) + (2 if current_chunk_parts else 0)
        if current_length + added_len <= max_chunk_size:
            current_chunk_parts.append(block)
            current_length += added_len
        else:
            chunks.append("\n\n".join(current_chunk_parts).strip())
            # Add trailing overlap if feasible
            if overlap > 0 and current_chunk_parts and len(current_chunk_parts[-1]) <= overlap:
                current_chunk_parts = [current_chunk_parts[-1], block]
                current_length = len(current_chunk_parts[0]) + 2 + len(block)
            else:
                current_chunk_parts = [block]
                current_length = len(block)

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts).strip())

    return [c for c in chunks if c.strip()]


# ==============================================================================
# Deterministic Keyword Matching
# ==============================================================================

def match_keywords(
    resume_text: str,
    keywords: List[str],
) -> Tuple[float, List[str], List[str]]:
    """
    Perform deterministic, case-insensitive, whitespace-tolerant keyword matching.

    Args:
        resume_text: Raw or normalized resume text.
        keywords: List of keyword strings from a Criterion.

    Returns:
        Tuple containing:
        - keyword_score: float in [0.0, 1.0] (matched_count / total_count)
        - matched_keywords: List of matched keyword strings
        - unmatched_keywords: List of unmatched keyword strings
    """
    if not resume_text or not resume_text.strip() or not keywords:
        # If no resume text or empty keywords, score is 0.0
        return 0.0, [], list(keywords)

    normalized_resume = " ".join(resume_text.lower().split())

    matched: List[str] = []
    unmatched: List[str] = []

    for kw in keywords:
        clean_kw = kw.strip()
        if not clean_kw:
            continue

        # Normalize internal whitespace within multi-word keywords
        words = clean_kw.lower().split()
        if not words:
            continue

        # Build boundary-aware regex: (?<!\w)word1\s+word2(?!\w)
        escaped_words = [re.escape(w) for w in words]
        kw_regex = r"(?<!\w)" + r"\s+".join(escaped_words) + r"(?!\w)"

        if re.search(kw_regex, normalized_resume, flags=re.IGNORECASE):
            matched.append(clean_kw)
        else:
            unmatched.append(clean_kw)

    total_valid = len(matched) + len(unmatched)
    score = float(len(matched) / total_valid) if total_valid > 0 else 0.0
    return round(score, 4), matched, unmatched


# ==============================================================================
# Embedding Model Manager (Singleton)
# ==============================================================================

class EmbeddingModelManager:
    """
    Singleton manager for the SentenceTransformer embedding model.
    Loads the model once and reuses it across matching requests.
    """
    _instance: Optional[Any] = None
    _loaded_model_name: Optional[str] = None

    @classmethod
    def get_model(cls, model_name: Optional[str] = None) -> Any:
        """
        Retrieve the cached SentenceTransformer model instance.
        Loads the model if not yet instantiated or if model_name changes.
        """
        target_model = model_name or get_embedding_model_name()

        if cls._instance is None or cls._loaded_model_name != target_model:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s", target_model)
                cls._instance = SentenceTransformer(target_model)
                cls._loaded_model_name = target_model
            except Exception as exc:
                raise EmbeddingModelError(
                    f"Failed to load sentence-transformers model '{target_model}': {exc}"
                ) from exc

        return cls._instance

    @classmethod
    def set_model(cls, model: Any, model_name: str = "mock-model") -> None:
        """Inject or mock the active model (primarily for unit testing)."""
        cls._instance = model
        cls._loaded_model_name = model_name

    @classmethod
    def clear(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None
        cls._loaded_model_name = None


# ==============================================================================
# Resume Matcher Engine
# ==============================================================================

class ResumeMatcher:
    """
    Resume-to-Criterion matching engine calculating semantic similarity
    and keyword evidence without candidate scoring.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """
        Initialize the ResumeMatcher.

        Args:
            model: Optional pre-loaded or mock embedding model.
            model_name: Sentence-Transformers model name. Defaults to config value.
            chunk_size: Max chunk character size. Defaults to config value (500).
            chunk_overlap: Character overlap. Defaults to config value (50).
        """
        self.chunk_size = chunk_size if chunk_size is not None else get_chunk_size()
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else get_chunk_overlap()
        self.model_name = model_name or get_embedding_model_name()
        self._model = model

    @property
    def model(self) -> Any:
        """Get the embedding model instance."""
        if self._model is not None:
            return self._model
        return EmbeddingModelManager.get_model(self.model_name)

    def match_criterion(
        self,
        resume_text: str,
        criterion: Union[Criterion, Any],
    ) -> CriterionMatchResult:
        """
        Match a single criterion against a resume text.

        Returns:
            CriterionMatchResult containing semantic_score, raw_semantic_score,
            keyword_score, matched_keywords, unmatched_keywords, and evidence chunk.

        Raises:
            InvalidCriterionError: If criterion is invalid or missing required attributes.
            EmbeddingModelError: If embedding inference fails.
        """
        if not isinstance(criterion, Criterion):
            raise InvalidCriterionError(
                f"Expected Criterion instance, received {type(criterion).__name__}."
            )

        if not criterion.name or not criterion.name.strip():
            raise InvalidCriterionError("Criterion name cannot be empty or blank.")

        # Safe handling for empty or whitespace-only resume text
        if not resume_text or not resume_text.strip():
            return CriterionMatchResult(
                criterion_name=criterion.name,
                criterion_category=criterion.category,
                semantic_score=0.0,
                raw_semantic_score=0.0,
                keyword_score=0.0,
                matched_keywords=[],
                unmatched_keywords=list(criterion.keywords),
                evidence="",
            )

        # 1. Keyword Evidence Matching
        keyword_score, matched_kw, unmatched_kw = match_keywords(
            resume_text, criterion.keywords
        )

        # 2. Text Chunking
        chunks = chunk_resume_text(
            resume_text,
            max_chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        if not chunks:
            return CriterionMatchResult(
                criterion_name=criterion.name,
                criterion_category=criterion.category,
                semantic_score=0.0,
                raw_semantic_score=0.0,
                keyword_score=keyword_score,
                matched_keywords=matched_kw,
                unmatched_keywords=unmatched_kw,
                evidence="",
            )

        # 3. Semantic Similarity Matching
        criterion_query = f"{criterion.name}: {criterion.description}".strip()

        try:
            crit_emb = self.model.encode(criterion_query, convert_to_numpy=True)
            chunk_embs = self.model.encode(chunks, convert_to_numpy=True)
        except Exception as exc:
            raise EmbeddingModelError(f"Embedding inference failed: {exc}") from exc

        crit_vec = np.asarray(crit_emb).flatten()
        chunk_arr = np.asarray(chunk_embs)
        if chunk_arr.ndim == 1:
            chunk_arr = chunk_arr.reshape(1, -1)

        best_raw_sim = -1.0
        best_chunk = chunks[0]

        for idx, chunk in enumerate(chunks):
            sim = compute_cosine_similarity(crit_vec, chunk_arr[idx])
            if sim > best_raw_sim:
                best_raw_sim = sim
                best_chunk = chunk

        normalized_semantic_score = normalize_cosine_similarity(best_raw_sim)

        return CriterionMatchResult(
            criterion_name=criterion.name,
            criterion_category=criterion.category,
            semantic_score=round(normalized_semantic_score, 4),
            raw_semantic_score=round(best_raw_sim, 4),
            keyword_score=keyword_score,
            matched_keywords=matched_kw,
            unmatched_keywords=unmatched_kw,
            evidence=best_chunk,
        )

    def match_criteria(
        self,
        resume_text: str,
        criteria: List[Criterion],
    ) -> ResumeMatchResult:
        """
        Match multiple criteria against a resume.

        Returns:
            ResumeMatchResult containing a list of CriterionMatchResult objects.
        """
        if criteria is None:
            raise InvalidInputError("Criteria list cannot be None.")

        results = [self.match_criterion(resume_text, c) for c in criteria]
        return ResumeMatchResult(matches=results)


def match_resume_to_criteria(
    resume_text: str,
    criteria: List[Criterion],
    model: Optional[Any] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> ResumeMatchResult:
    """
    Convenience function to match a resume against a list of criteria.
    """
    matcher = ResumeMatcher(
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return matcher.match_criteria(resume_text, criteria)
