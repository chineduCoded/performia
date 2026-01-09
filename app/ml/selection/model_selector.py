from typing import Dict


class ModelSelector:
    def __init__(
            self,
            min_precision: float,
            min_recall: float = 0.0,
            primary_metric: str = "f2",
        ):
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.primary_metric = primary_metric

    def select(self, model_metrics: Dict[str, dict]) -> str:
        """
        model_metrics = {
            "rf": {...},
            "xgb": {...},
            "lgbm": {...}
        }
        """

        # 1. Hard constraints
        candidates = {
            name: m for name, m in model_metrics.items()
            if m["precision"] >= self.min_precision
            and m["recall"] >= self.min_recall
        }

        if not candidates:
            raise ValueError("No model satisfies policy constraints")
        
        # 2. Rank by primary metric
        best_model = max(
            candidates.items(),
            key=lambda item: item[1][self.primary_metric],
        )

        return best_model[0]