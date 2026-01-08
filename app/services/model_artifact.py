from dataclasses import dataclass
from typing import List

from app.ml.predictors.base import BasePredictor


@dataclass(frozen=True)
class ModelArtifact:
    predictor: BasePredictor
    features: List[str]
    version: str
    metrics: dict

    def validate_input(self, payload: dict):
        missing = set(self.features) - payload.keys()
        if missing:
            raise ValueError(f"Missing features: {missing}")
    
    def to_matrix(self, payload: dict):
        self.validate_input(payload)
        return [[payload[f] for f in self.features]]
