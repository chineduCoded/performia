from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # app/
ML_DIR = BASE_DIR / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)