from datetime import datetime
from pathlib import Path
import shutil
import joblib

from app.ml.data.loader import StudentDataLoader
from app.ml.features.snapshot import SnapshotFeatureEngineer
from app.ml.split.student_splitter import StudentSplitter

from app.ml.models.risk_models.random_forest import RandomForestRiskModel
from app.ml.models.risk_models.xgboost import XGBoostRiskModel
from app.ml.models.risk_models.lightgbm import LightGBMRiskModel
from app.ml.selection.model_selector import ModelSelector

from app.services.model_artifact import ModelArtifact
from app.ml.training.threshold_cv_trainer import ThresholdCVTrainer, ThresholdOptimizer
from app.config import ARTIFACTS_DIR, DATA_DIR

MIN_PRECISION = 0.80
MIN_RECALL = 0.90

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
    threshold_optimizer = ThresholdOptimizer(min_precision=MIN_PRECISION)


    models = {
        "rf": RandomForestRiskModel(engineer.NUMERIC, engineer.CATEGORICAL),
        "xgb": XGBoostRiskModel(engineer.NUMERIC, engineer.CATEGORICAL),
        "lgbm": LightGBMRiskModel(engineer.NUMERIC, engineer.CATEGORICAL),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = datetime.now().strftime("%Y-%m-%d")

    model_metrics = {}
    artifact_paths = {}

    for name, model in models.items():
        try:
            trainer = ThresholdCVTrainer(
                model,
                splitter,
                threshold_optimizer 
            )

            metrics = trainer.evaluate(X, y, student_ids)
            predictor = trainer.train(X, y, student_ids)

            artifact = ModelArtifact(
                predictor=predictor,
                features=engineer.feature_columns,
                version=version,
                metrics=metrics
            )

            artifact_path = ARTIFACTS_DIR / f"risk_{name}.joblib"
            joblib.dump(artifact, artifact_path)

            model_metrics[name] = metrics
            artifact_paths[name] = artifact_path

            print(f"[{name}] saved: {artifact_path}")
            print(f"[{name}] CV metrics: {metrics}")
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            continue

    selector = ModelSelector(
        min_precision=MIN_PRECISION,
        min_recall=MIN_RECALL,
        primary_metric="f2"
    )

    best_model_name = selector.select(model_metrics)
    best_artifact_path = artifact_paths[best_model_name]

    print(f"\nSelected production model: {best_model_name}")

    production_path = ARTIFACTS_DIR / "risk_production.joblib"
    shutil.copy(best_artifact_path, production_path)


    print(f"Promoted {best_model_name}: {production_path}")


if __name__ == "__main__":
    train_and_save_risk_models(
        csv_path=Path(DATA_DIR / "nigerian_university_students.csv")
    )