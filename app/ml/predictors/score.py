from sklearn.pipeline import Pipeline

from .base import BasePredictor

class ScorePredictor(BasePredictor):
    def __init__(self, pipelne: Pipeline):
        self.pipelne = pipelne

    def train(self, X, y):
        self.pipelne.fit(X, y)

    def predict(self, X):
        return self.pipelne.predict(X)