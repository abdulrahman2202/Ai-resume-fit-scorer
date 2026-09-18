import math
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Path to the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
ENV_PATH = ROOT_DIR / ".env"

# Ensure .env is loaded
load_dotenv(dotenv_path=ENV_PATH)


class ConfigurationError(Exception):
    """Raised when configuration values are missing, corrupt, or invalid."""
    pass


DEFAULT_CONFIG: Dict[str, Any] = {
    "scoring": {
        "semantic_weight": 0.60,
        "keyword_weight": 0.40,
    },
    "category_weights": {
        "technical_skills": 0.35,
        "experience": 0.25,
        "education": 0.10,
        "responsibilities": 0.20,
        "tools": 0.10,
    },
    "thresholds": {
        "strong_match": 0.75,
        "partial_match": 0.50,
        "minimum_resume_chars": 100,
    },
    "matching": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embedding_model": "all-MiniLM-L6-v2",
    },
    "calibration": {
        "max_similar_resume_gap": 15.0,
    },
    "gemini": {
        "max_retries": 3,
        "initial_retry_delay_seconds": 1.0,
        "max_retry_delay_seconds": 8.0,
    },
}


def load_config(config_path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Load configuration from config.yaml file.
    Falls back to DEFAULT_CONFIG if file is not found or empty.
    """
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass

    return DEFAULT_CONFIG.copy()


def get_config() -> Dict[str, Any]:
    """Retrieve application configuration dictionary."""
    return load_config()


def get_minimum_resume_chars(config_path: Path | str = CONFIG_PATH) -> int:
    """
    Get the configured minimum character threshold for valid resumes.
    Defaults to 100 characters if not explicitly configured.
    """
    config = load_config(config_path)
    thresholds = config.get("thresholds", {})
    return int(thresholds.get("minimum_resume_chars", 100))


def get_gemini_api_key() -> Optional[str]:
    """Retrieve the Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")


def get_gemini_model() -> str:
    """Retrieve the Gemini model name from environment variables, defaulting to gemini-3.6-flash."""
    return os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


def get_gemini_config(config_path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """Retrieve the gemini configuration section."""
    config = load_config(config_path)
    return config.get("gemini", DEFAULT_CONFIG["gemini"])


def get_gemini_max_retries(config_path: Path | str = CONFIG_PATH) -> int:
    """Retrieve maximum retry attempts for Gemini API calls."""
    cfg = get_gemini_config(config_path)
    return int(cfg.get("max_retries", 3))


def get_gemini_initial_retry_delay(config_path: Path | str = CONFIG_PATH) -> float:
    """Retrieve initial retry delay in seconds for Gemini API calls."""
    cfg = get_gemini_config(config_path)
    return float(cfg.get("initial_retry_delay_seconds", 1.0))


def get_gemini_max_retry_delay(config_path: Path | str = CONFIG_PATH) -> float:
    """Retrieve maximum retry delay in seconds for Gemini API calls."""
    cfg = get_gemini_config(config_path)
    return float(cfg.get("max_retry_delay_seconds", 8.0))


def get_matching_config(config_path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """Retrieve the matching configuration section."""
    config = load_config(config_path)
    return config.get("matching", DEFAULT_CONFIG["matching"])


def get_chunk_size(config_path: Path | str = CONFIG_PATH) -> int:
    """Retrieve maximum character chunk size for resume splitting."""
    cfg = get_matching_config(config_path)
    return int(cfg.get("chunk_size", 500))


def get_chunk_overlap(config_path: Path | str = CONFIG_PATH) -> int:
    """Retrieve chunk overlap in characters."""
    cfg = get_matching_config(config_path)
    return int(cfg.get("chunk_overlap", 50))


def get_embedding_model_name(config_path: Path | str = CONFIG_PATH) -> str:
    """Retrieve the sentence-transformers model name."""
    cfg = get_matching_config(config_path)
    return str(cfg.get("embedding_model", "all-MiniLM-L6-v2"))


def get_scoring_weights(config_path: Path | str = CONFIG_PATH) -> Dict[str, float]:
    """
    Retrieve semantic and keyword scoring weights.
    Supports backward-compatible aliases (semantic_similarity_weight / semantic_weight).
    """
    config = load_config(config_path)
    scoring = config.get("scoring", {})
    sem = scoring.get("semantic_weight", scoring.get("semantic_similarity_weight", 0.60))
    kw = scoring.get("keyword_weight", 0.40)
    return {"semantic_weight": float(sem), "keyword_weight": float(kw)}


def get_category_weights(config_path: Path | str = CONFIG_PATH) -> Dict[str, float]:
    """
    Retrieve category weights.
    Supports backward-compatible aliases (category_weights / criteria_weights).
    """
    config = load_config(config_path)
    cat_weights = config.get("category_weights", config.get("criteria_weights", {}))
    if not cat_weights:
        cat_weights = DEFAULT_CONFIG["category_weights"]
    return {k: float(v) for k, v in cat_weights.items()}


def get_calibration_config(config_path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """Retrieve the calibration configuration section."""
    config = load_config(config_path)
    return config.get("calibration", DEFAULT_CONFIG.get("calibration", {"max_similar_resume_gap": 15.0}))


def get_max_similar_resume_gap(config_path: Path | str = CONFIG_PATH) -> float:
    """Retrieve the maximum acceptable score gap between similar resumes."""
    cal = get_calibration_config(config_path)
    return float(cal.get("max_similar_resume_gap", 15.0))


def validate_configuration(config_path: Path | str = CONFIG_PATH) -> None:
    """
    Validate that scoring weights and category weights are properly configured and sum to 1.0.

    Raises:
        ConfigurationError: If any weight is invalid or sums do not equal 1.0.
    """
    scoring = get_scoring_weights(config_path)
    sem = scoring["semantic_weight"]
    kw = scoring["keyword_weight"]

    if sem < 0.0 or kw < 0.0:
        raise ConfigurationError(f"Scoring weights must be non-negative, got semantic={sem}, keyword={kw}")

    total_scoring = sem + kw
    if not math.isclose(total_scoring, 1.0, abs_tol=1e-4):
        raise ConfigurationError(
            f"Scoring weights must sum to 1.0, got semantic_weight={sem} + keyword_weight={kw} = {total_scoring:.4f}"
        )

    cat_weights = get_category_weights(config_path)
    if not cat_weights:
        raise ConfigurationError("Category weights configuration is empty.")

    if any(w < 0.0 for w in cat_weights.values()):
        raise ConfigurationError(f"Category weights must be non-negative: {cat_weights}")

    total_cat = sum(cat_weights.values())
    if not math.isclose(total_cat, 1.0, abs_tol=1e-4):
        raise ConfigurationError(
            f"Category weights must sum to 1.0, got {total_cat:.4f}: {cat_weights}"
        )
