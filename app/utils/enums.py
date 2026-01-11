import enum
from dataclasses import dataclass

class PerformanceTrend(str, enum.Enum):
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"



class Level(int, enum.Enum):
    L100 = 100
    L200 = 200
    L300 = 300
    L400 = 400
    L500 = 500

class RiskLevel(str, enum.Enum):
    NOT_AT_RISK = "Not at risk"
    AT_RISK = "At risk"


class RiskSeverityLevel(str, enum.Enum):
    LOW = "Low"
    BORDERLINE = "Borderline"
    MODERATE = "Moderate"
    HIGH = "High"


@dataclass(frozen=True)
class Severity:
    severity_level: RiskSeverityLevel
    color: str