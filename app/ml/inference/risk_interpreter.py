from app.config import RISK_THRESHOLDS
from app.utils.enums import RiskSeverityLevel
from app.schemas.prediction import SeverityInfo
from app.utils.severity_map import SEVERITY_MAP


def interpret_risk(prob: float) -> dict:
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {prob}")

    if prob < RISK_THRESHOLDS.low:
        severity_dc = SEVERITY_MAP[RiskSeverityLevel.LOW]
        label = "Low risk"
        advice = (
            "You are currently performing well. "
            "Maintain consistent attendance and study habits."
        )

    elif prob < RISK_THRESHOLDS.borderline:
        severity_dc = SEVERITY_MAP[RiskSeverityLevel.BORDERLINE]
        label = "Borderline risk"
        advice = (
            "Your performance is slightly below ideal. "
            "Monitor attendance and review weak subjects early."
        )

    elif prob < RISK_THRESHOLDS.moderate:
        severity_dc = SEVERITY_MAP[RiskSeverityLevel.MODERATE]
        label = "Moderate risk"
        advice = (
            "There are clear signs of academic risk. "
            "Consider adjusting your study plan or seeking academic support."
        )

    else:
        severity_dc = SEVERITY_MAP[RiskSeverityLevel.HIGH]
        label = "High risk"
        advice = (
            "Immediate intervention recommended. "
            "Meet with your academic advisor or lecturer as soon as possible."
        )

    # Convert dataclass to Pydantic model
    severity = SeverityInfo(
        level=severity_dc.severity_level,
        color=severity_dc.color,
    )

    return {
        "risk_label": label,
        "advice": advice,
        "severity": severity,
    }