from dataclasses import dataclass
from typing import List

import pandas as pd

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
    
    def prepare_features(self, payload: dict) -> pd.DataFrame:
        """
        Convert payload dict to DataFrame with correct feature columns.
        """
        self.validate_input(payload)
        return pd.DataFrame([{f: payload[f] for f in self.features}])
