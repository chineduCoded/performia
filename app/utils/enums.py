import enum

class PerformanceTrend(str, enum.Enum):
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"