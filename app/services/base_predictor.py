import pandas as pd
from app.services.model_artifact import ModelArtifact


class BinaryClassifierPredictor:
    def __init__(self, model_artifact: ModelArtifact):
        self.model = model_artifact.model
        self.threshold = model_artifact.threshold
        self.features = model_artifact.features

    def predict(self, payload: dict) -> dict:
        missing = set(self.features) - set(payload.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        df = pd.DataFrame([payload])[self.features]

        proba = float(self.model.predict_proba(df)[0, 1])
        is_at_risk = (proba >= self.threshold)

        return self._human_readable_response(proba, is_at_risk)
    
    def _human_readable_response(self, proba: float, is_at_risk: bool) -> dict:
        return {
            "decision": "Early Risk Detected" if is_at_risk else "No Immediate Risk",
            "risk_level": self._risk_level(proba),
            "confidence": round(proba, 3),
            "confidence_pct": f"{round(proba * 100)}%",
            "threshold": self.threshold,
            "explanation": self._explanation(proba, is_at_risk)
        }
    
    def _risk_level(self, proba: float) -> str:
        t = self.threshold

        if proba >= t * 1.5:
            return "Very High"
        if proba >= t * 1.2:
            return "High"
        if proba >= t:
            return "Moderate"
        if proba >= t * 0.7:
            return "Low"
        return "Very Low"
        
    def _explanation(self, proba: float, is_at_risk: bool) -> str:
        risk_level = self._risk_level(proba)
        pct = round(proba * 100)

        if is_at_risk:
            return (
                f"The student is classified as 'At Risk' based on a predicted risk "
                f"probability of {pct}%, which exceeds the early-warning threshold. "
                f"This corresponds to a {risk_level} risk level and suggests that "
                "early academic intervention may be beneficial."
            )
        else:
            return (
                f"The student is classified as 'Not At Risk' based on a predicted risk "
                f"probability of {pct}%, which is below the early-warning threshold. "
                f"This corresponds to a {risk_level} risk level. "
                "Current academic support strategies can be maintained."
            )
        