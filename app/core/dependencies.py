import re
from functools import lru_cache
from app.services.model_loader import ModelLoader
from app.config import ARTIFACTS_DIR

# Initialize loader
loader = ModelLoader(artifacts_dir=ARTIFACTS_DIR)


async def load_models():
    """
    Preload production models at app startup.
    """
    loader.load("risk", "current_risk_production.joblib")
    # loader.load("trend", "trend_model.joblib")


@lru_cache(maxsize=32)
def get_risk_artifact(model: str = "risk") -> object:
    """
    Return cached artifact for the requested risk model.

    Example query:
        GET /predict/risk?model=risk_v2
    """
    # Validate model name
    if not re.match(r'^[a-zA-Z0-9_-]+$', model):
        raise ValueError(f"Invalid model name: {model}")

    # Map logical model name to artifact file
    artifact_map = {
        "risk": "current_risk_production.joblib",
    }

    artifact_file = artifact_map.get(model)
    if artifact_file is None:
        raise ValueError(f"No artifact found for model '{model}'")

    return loader.load(model, artifact_file)
