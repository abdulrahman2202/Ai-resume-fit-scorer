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


def get_gemini_api_key() -> Optional[str]:
    """Retrieve the Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")


def get_gemini_model() -> str:
    """Retrieve the Gemini model name from environment variables, defaulting to gemini-2.5-flash."""
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
