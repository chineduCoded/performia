import numpy as np
from sklearn.metrics import precision_recall_curve

class ThresholdOptimizer:
    def __init__(self, min_precision: float):
        self.min_precision = min_precision

    def find_optimal_threshold(self, y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

        # Drop last element (sklearn quirk)
        precision, recall = precision[:-1], recall[:-1]

        valid = precision >= self.min_precision

        if not valid.any():
            return 1.0  # safest fallback: predict no one at risk

        best_recall = recall[valid].max()
        best_idxs = np.where(valid & (recall >= best_recall - 1e-6))[0]

        return thresholds[best_idxs].max()
