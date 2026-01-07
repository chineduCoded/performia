from .base import SequencePredictor


class TrendPredictor(SequencePredictor):
    def __init__(self, model, trend_policy):
        self.model = model
        self.policy = trend_policy

    def train(self, X_seq, y_seq=None):
        if not hasattr(self.model, "fit"):
            raise AttributeError(f"Model {type(self.model).__name__} does not have a 'fit' method")
        self.model.fit(X_seq, y_seq)

    def predict_sequence(self, X_seq):
        raw_trend = self.model.predict(X_seq)
        return self.policy.classify(raw_trend)