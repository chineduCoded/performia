from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline

from app.ml.features.types import FeatureGroups

class BaseModelSpec(ABC):
    def __init__(
        self,
        features: FeatureGroups
    ):
        if not features["numeric"] and not features["categorical"]:
            raise ValueError("At least one of num_features or cat_features must be non-empty")
        
        overlap = set(features["numeric"]) & set(features["categorical"])
        if overlap:
            raise ValueError(f"Features cannot be both numerical and categorical: {overlap}")
        
        self.features = features
        self.num_features = list(features["numeric"])
        self.cat_features = list(features["categorical"])

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        pass

class BaseTemporalModelSpec(ABC):
    @abstractmethod
    def build_model(self):
        """Return a temporal model (not necessarily sklearn)"""
        pass