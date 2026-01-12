from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..base import BaseModelSpec

class XGBoostRiskModel(BaseModelSpec):
    # def __init__(self, num_features, cat_features):
    #     num_features = num_features or []
    #     cat_features = cat_features or []

        # if not num_features and not cat_features:
        #     raise ValueError("At least one of num_features or cat_features must be non-empty")
        
        # overlap = set(num_features) & set(cat_features)
        # if overlap:
        #     raise ValueError(f"Features cannot be both numerical and categorical: {overlap}")
        
        # self.num_features = num_features
        # self.cat_features = cat_features

    def build_pipeline(self):
        preprocessor = (
            ColumnTransformer(
                [
                    ("num", "passthrough", self.num_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_features),
                ],
                verbose_feature_names_out=False,
            )
            .set_output(transform="pandas")
        )

        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            eval_metric="logloss",
            random_state=42
        )

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", xgb),
        ])
