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

def train_and_save_risk_models(csv_path: Path, version=None):
    if not csv_path.exists():
       raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = StudentDataLoader(csv_path).load_latest_semester()
    if df.empty:
        raise ValueError("Loaded dataframe is empty")
    
    engineer = SnapshotFeatureEngineer()
    df = engineer.transform(df)

    required_cols = {"is_at_risk", "student_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df["student_id"].duplicated().any():
        raise ValueError("student_id contains duplicates")
    
    if df["is_at_risk"].isnull().any():
        raise ValueError("is_at_risk contains null values")

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

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        try:
            trainer = ThresholdCVTrainer(
                model,
                splitter,
                threshold_optimizer 
            )

            metrics = trainer.evaluate(X, y, student_ids)
            predictor = trainer.train(X, y)

            if version is None:
                from datetime import datetime
                version = datetime.now().strftime("%Y-%m-%d")

            artifact = ModelArtifact(
                predictor=predictor,
                features=engineer.feature_columns,
                version=version
            )

            artifact_path = ARTIFACTS_DIR / f"risk_{name}.joblib"
            joblib.dump(artifact, artifact_path)

            print(f"[{name}] saved → {artifact_path}")
            print(f"[{name}] CV metrics → {metrics}")
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            continue
