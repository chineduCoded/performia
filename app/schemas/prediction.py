from pydantic import BaseModel, Field

from app.utils.enums import RiskLevel, RiskSeverityLevel

class SeverityInfo(BaseModel):
    level: RiskSeverityLevel
    color: str


class RiskPredictionResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    probability_pct: float = Field(..., ge=0, le=100)          
    risk: RiskLevel
    risk_label: str
    advice: str
    severity: SeverityInfo
    model_version: str