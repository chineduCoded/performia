from sklearn.pipeline import Pipeline
from .base import ProbabilityPredictor

class RiskPredictor(ProbabilityPredictor):
    def __init__(self, pipeline: Pipeline, threshold_policy):
        if not hasattr(threshold_policy, 'threshold'):
            raise ValueError("threshold_policy must have a 'threshold' attribute")
        
        self.pipeline = pipeline
        self.threshold_policy = threshold_policy

    def predict_proba(self, X):
        proba = self.pipeline.predict_proba(X)
        if proba.shape[1] != 2:
            raise ValueError(f"Expected binary classification (2 classes), got {proba.shape[1]} classes")
        return proba[:, 1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold_policy.threshold).astype(int)