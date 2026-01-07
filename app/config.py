from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # app/
ML_DIR = BASE_DIR / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"


def initialize_directories():
    """Create required directories for the application."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)