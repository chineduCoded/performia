from .base import SequencePredictor


class TrendPredictor(SequencePredictor):
    def __init__(self, model, trend_policy):
        self.model = model
        self.policy = trend_policy

    def train(self, X_seq, y_seq=None):
        if hasattr(self.model, "fit"):
            self.model.fit(X_seq, y_seq)

    def predict_sequence(self, X_seq):
        raw_trend = self.model.predict(X_seq)
        return self.policy.classify(raw_trend)