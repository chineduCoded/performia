from sklearn.pipeline import Pipeline

from .base import RegressionPredictor

class ScorePredictor(RegressionPredictor):
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def train(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)