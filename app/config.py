from pathlib import Path
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RiskThresholds:
    low: float = 0.15
    borderline: float = 0.25
    moderate: float = 0.50


RISK_THRESHOLDS = RiskThresholds()