import threading

from app.services.model_loader import ModelLoader
from app.services.base_predictor import BinaryClassifierPredictor
from app.services.model_loader import ModelLoader
from app.services.base_predictor import BinaryClassifierPredictor

_predictor: BinaryClassifierPredictor | None = None
_lock = threading.Lock()

def get_predictor() -> BinaryClassifierPredictor:
    global _predictor

    if _predictor is None:
        with _lock:
            # Double-check after acquiring lock
            if _predictor is None:
                model_artifact = ModelLoader.load_model("risk", "risk_model.joblib")
                _predictor = BinaryClassifierPredictor(model_artifact)

    return _predictor

def predict_risk(payload: dict) -> dict:
    predictor = get_predictor()
    return predictor.predict(payload)