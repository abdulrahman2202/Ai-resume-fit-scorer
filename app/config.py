from pathlib import Path
from typing import Any, Dict
import yaml

# Path to the root config.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "scoring": {
        "semantic_similarity_weight": 0.60,
        "keyword_weight": 0.40,
    },
    "criteria_weights": {
        "technical_skills": 0.35,
        "experience": 0.25,
        "education": 0.15,
        "responsibilities": 0.15,
        "tools": 0.10,
    },
    "thresholds": {
        "strong_match": 0.75,
        "partial_match": 0.50,
        "minimum_resume_chars": 100,
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
