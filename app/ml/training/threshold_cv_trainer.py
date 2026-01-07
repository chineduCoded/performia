from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
import numpy as np

from ..split.student_splitter import StudentSplitter
from ..models.base import BaseModelSpec
from .threshold_optimizer import ThresholdOptimizer
from ..predictors.risk import RiskPredictor
from ..policies.threshold import ThresholdPolicy

class ThresholdCVTrainer:
    def __init__(
        self, 
        model: BaseModelSpec, 
        splitter: StudentSplitter,
        threshold_optimizer: ThresholdOptimizer
    ):
        self.pipeline: Pipeline = model.build_pipeline()
        self.splitter = splitter
        self.threshold_optimizer = threshold_optimizer

    def evaluate(self, X, y, student_ids):
        fold_metrics = []

        for train_idx, val_idx in self.splitter.split(X, y, student_ids):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train
            self.pipeline.fit(X_train, y_train)

            # Predict probabilities
            y_prob = self.pipeline.predict_proba(X_val)[:, 1]

            threshold = self.threshold_optimizer.find_optimal_threshold(
                y_val, y_prob
            )

            y_pred = (y_prob >= threshold).astype(int)

            fold_metrics.append({
                "threshold": threshold,
                "roc_auc": roc_auc_score(y_val, y_prob),
                "ap": average_precision_score(y_val, y_prob),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred),
                "accuracy": accuracy_score(y_val, y_pred),
            })

        return self._aggregate(fold_metrics)

    def _aggregate(self, metrics):
        return {
            k: np.mean([m[k] for m in metrics])
            for k in metrics[0]
        }
    
    def train(self, X, y):
        """
        Train final model on full data using threshold optimized CV
        """
        thresholds = []

        for train_idx, val_idx in self.splitter.split(X, y, student_ids=None):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.pipeline.fit(X_train, y_train)
            y_prob = self.pipeline.predict_proba(X_val)[:, 1]

            t = self.threshold_optimizer.find_optimal_threshold(y_val, y_prob)
            thresholds.append(t)

        final_threshold = float(np.mean(thresholds))

        # Train on full dataset
        self.pipeline.fit(X, y)

        return RiskPredictor(
            pipeline=self.pipeline,
            threshold_policy=ThresholdPolicy(final_threshold)
        )
