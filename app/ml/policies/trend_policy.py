from app.utils.enums import PerformanceTrend


class TrendPolicy:
    def classify(self, slope):
        if slope > 0.05:
            return PerformanceTrend.IMPROVING
        if slope < -0.05:
            return PerformanceTrend.DECLINING
        return PerformanceTrend.STABLE