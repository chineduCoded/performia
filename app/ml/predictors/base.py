from abc import ABC, abstractmethod

# -------------------------
# Snapshot-based predictors
# -------------------------
class BasePredictor(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


# -------------------------
# Probability predictors
# -------------------------
class ProbabilityPredictor(BasePredictor):
    @abstractmethod
    def predict_proba(self, X):
        pass


# -------------------------
# Sequence-based predictors
# -------------------------
class SequencePredictor(ABC):
    @abstractmethod
    def train(self, X_seq, y_seq=None):
        pass

    @abstractmethod
    def predict_sequence(self, X_seq):
        pass
