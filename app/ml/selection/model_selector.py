from typing import Dict


class ModelSelector:
    def __init__(
        self,
        min_precision: float,
        min_recall: float = 0.0,
        primary_metric: str = "f2",
        allow_best_fallback: bool = False,
    ):
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.primary_metric = primary_metric
        self.allow_best_fallback = allow_best_fallback

    def select(self, model_metrics: Dict[str, dict]) -> str:
        """
        model_metrics = {
            "rf": {...},
            "xgb": {...},
            "lgbm": {...}
        }
        """

        # 1. Apply hard constraints
        candidates = {
            name: m
            for name, m in model_metrics.items()
            if m["precision"] >= self.min_precision
            and m["recall"] >= self.min_recall
        }

        # 2. If no model passes constraints
        if not candidates:
            if not self.allow_best_fallback:
                raise ValueError("No model satisfies policy constraints")

            # Governance-friendly fallback
            candidates = model_metrics

        # 3. Rank by primary metric (Fβ by default)
        best_model = max(
            candidates.items(),
            key=lambda item: item[1][self.primary_metric],
        )

        return best_model[0]
