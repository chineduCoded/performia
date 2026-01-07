from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from ..base import BaseModelSpec

class RandomForestRiskModel(BaseModelSpec):
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

    def build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer([
            ("num", "passthrough", self.num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)
        ])

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: 3},
            random_state=42,
            n_jobs=-1
        )

        clf = CalibratedClassifierCV(rf, cv=3)

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", clf),
        ])