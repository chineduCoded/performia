from pydantic import BaseModel, Field

from app.utils.enums import RiskClassification, RiskSeverityLevel

class SeverityInfo(BaseModel):
    level: RiskSeverityLevel
    color: str


class RiskPredictionResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    probability_pct: float = Field(..., ge=0, le=100)
    predicted_risk: int        
    risk_class: RiskClassification
    risk_label: str
    advice: str
    severity: SeverityInfo
    model_version: str