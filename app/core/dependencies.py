from pathlib import Path
import re
from functools import lru_cache
from app.services.model_loader import ModelLoader
from app.config import ARTIFACTS_DIR


loader = ModelLoader(
    artifacts_dir=ARTIFACTS_DIR
)

async def load_models():
    await loader.load("risk", "risk_model.joblib")
    # loader.load("score", "score_model.joblib")
    # loader.load("trend", "trend_model.joblib")

@lru_cache(maxsize=32)
def get_risk_artifact(model: str = "risk"): # allow POST /predict/risk?model=risk_v2
    if not re.match(r'^[a-zA-Z0-9_-]+$', model):
        raise ValueError(f"Invalid model name: {model}")
    return loader.load(model, f"{model}_model.joblib")