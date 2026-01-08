from pathlib import Path

# app/
BASE_DIR = Path(__file__).resolve().parent

# performia/
PROJECT_ROOT = BASE_DIR.parent

ML_DIR = BASE_DIR / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"

 
DATA_DIR = PROJECT_ROOT / "data"


def initialize_directories():
    """Create required directories for the application."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)