from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..base import BaseModelSpec

class LightGBMRiskModel(BaseModelSpec):
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

    def build_pipeline(self):
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
        ])

        lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            class_weight={0: 1, 1: 3},
            random_state=42
        )

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", lgbm),
        ])
