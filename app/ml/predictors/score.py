from sklearn.pipeline import Pipeline

from .base import BasePredictor

class ScorePredictor(BasePredictor):
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def train(self, X, y):
        self.pipelne.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)