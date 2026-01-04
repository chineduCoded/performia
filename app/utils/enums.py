import enum

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