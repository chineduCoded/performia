from sklearn.pipeline import Pipeline
from .base import ProbabilityPredictor

class RiskPredictor(ProbabilityPredictor):
    def __init__(self, pipeline: Pipeline, threshold_policy):
        self.pipeline = pipeline
        self.threshold_policy = threshold_policy

    def train(self, X, y):
        self.pipeline.fit(X, y)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold_policy.threshold).astype(int)