from pathlib import Path
from functools import lru_cache
from app.services.model_loader import ModelLoader
from app.config import ARTIFACTS_DIR


loader = ModelLoader(
    artifacts_dir=ARTIFACTS_DIR
)

def load_models():
    loader.load("risk", "risk_model.joblib")
    # loader.load("score", "score_model.joblib")
    # loader.load("trend", "trend_model.joblib")

@lru_cache()
def get_risk_artifact(model: str = "risk"): # allow POST /predict/risk?model=risk_v2
    return loader.load(model, f"{model}_model.joblib")