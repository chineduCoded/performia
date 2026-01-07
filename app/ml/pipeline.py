from pathlib import Path
import joblib

from app.ml.data.loader import StudentDataLoader
from app.ml.features.snapshot import SnapshotFeatureEngineer
from app.ml.split.student_splitter import StudentSplitter

from app.ml.models.risk_models.random_forest import RandomForestRiskModel
from app.ml.models.risk_models.xgboost import XGBoostRiskModel
from app.ml.models.risk_models.lightgbm import LightGBMRiskModel
from app.ml.models.base import BaseModelSpec

from app.ml.predictors.risk import RiskPredictor
from app.ml.policies.threshold import ThresholdPolicy
from app.services.model_artifact import ModelArtifact
from app.ml.training.threshold_cv_trainer import ThresholdCVTrainer, ThresholdOptimizer
from app.config import ARTIFACTS_DIR

def train_and_save_risk_models(csv_path: Path):
    df = StudentDataLoader(csv_path).load_latest_semester()
    engineer = SnapshotFeatureEngineer()
    df = engineer.transform(df)

    X = df[engineer.feature_columns]
    y = df["is_at_risk"]
    student_ids = df["student_id"]

    splitter = StudentSplitter(n_splits=5)
    threshold_optimizer = ThresholdOptimizer(min_precision=0.80)


    models = {
        "rf": RandomForestRiskModel(engineer.NUMERIC, engineer.CATEGORICAL),
        "xgb": XGBoostRiskModel(engineer.NUMERIC, engineer.CATEGORICAL),
        "lgbm": LightGBMRiskModel(engineer.NUMERIC, engineer.CATEGORICAL),
    }

    for name, model in models.items():
        trainer = ThresholdCVTrainer(
            model,
            splitter,
            threshold_optimizer
        )

        metrics = trainer.evaluate(X, y, student_ids)
        predictor = trainer.train(X, y)

        artifact = ModelArtifact(
            predictor=predictor,
            features=engineer.feature_columns,
            version="2025-01-07"
        )

        artifact_path = ARTIFACTS_DIR / f"risk_{name}.joblib"
        joblib.dump(artifact, artifact_path)

        print(f"[{name}] saved → {artifact_path}")
        print(f"[{name}] CV metrics → {metrics}")
