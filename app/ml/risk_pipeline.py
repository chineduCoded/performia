from datetime import datetime
from pathlib import Path
import shutil
import joblib
from typing import List, Dict, Type

from app.ml.data.loader import StudentDataLoader
from app.ml.features.snapshot import SnapshotFeatureEngineer
from app.ml.split.student_splitter import StudentSplitter

from app.ml.models.base import BaseModelSpec
from app.ml.models.risk_models.random_forest import RandomForestRiskModel
from app.ml.models.risk_models.xgboost import XGBoostRiskModel
from app.ml.models.risk_models.lightgbm import LightGBMRiskModel
from app.ml.selection.model_selector import ModelSelector

from app.services.model_artifact import ModelArtifact
from app.ml.training.threshold_cv_trainer import ThresholdCVTrainer, ThresholdOptimizer
from app.ml.training.risk_task import RiskTask, RiskPolicy
from app.config import ARTIFACTS_DIR, DATA_DIR


# =========================
# Training Pipeline
# =========================
def train_and_save_risk_models(csv_path: Path, version: str | None = None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = datetime.now().strftime("%Y-%m-%d")

    # ----------- TASKS (PRODUCTION ONLY) -----------
    tasks: List[RiskTask] = [
        RiskTask(
            name="current_risk",
            loader=StudentDataLoader.load_latest,
            feature_engineer=SnapshotFeatureEngineer(),
            target="is_at_risk",
            policy=RiskPolicy(
                min_precision=0.80,
                min_recall=0.90,
                beta=2,
            ),
        )
    ]

    models: Dict[str, Type[BaseModelSpec]] = {
        "rf": RandomForestRiskModel,
        "xgb": XGBoostRiskModel,
        "lgbm": LightGBMRiskModel,
    }

    # ----------- TRAIN LOOP -----------
    for task in tasks:
        print(f"\n=== TRAINING TASK: {task.name} ===")

        loader = StudentDataLoader(csv_path)
        df = task.loader(loader)

        engineer = task.feature_engineer
        df = engineer.transform(df)

        required_cols = {"student_id", task.target}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for {task.name}: {missing}")

        X = df[engineer.feature_columns]
        y = df[task.target]
        student_ids = df["student_id"]

        splitter = StudentSplitter(n_splits=5)
        threshold_optimizer = ThresholdOptimizer(
            min_precision=task.policy.min_precision
        )

        model_metrics: Dict[str, dict] = {}
        artifact_paths: Dict[str, Path] = {}

        for name, ModelClass in models.items():
            try:
                model = ModelClass(engineer.feature_groups)

                trainer = ThresholdCVTrainer(
                    model=model,
                    splitter=splitter,
                    threshold_optimizer=threshold_optimizer,
                )

                metrics = trainer.evaluate(X, y, student_ids)
                predictor = trainer.train(X, y, student_ids)

                artifact = ModelArtifact(
                    predictor=predictor,
                    features=engineer.feature_columns,
                    version=version,
                    metrics=metrics,
                )

                artifact_path = ARTIFACTS_DIR / f"{task.name}_{name}.joblib"
                joblib.dump(artifact, artifact_path)

                model_metrics[name] = metrics
                artifact_paths[name] = artifact_path

                print(f"[{name}] saved → {artifact_path}")
                print(f"[{name}] CV metrics → {metrics}")

            except Exception as e:
                print(f"[{name}] FAILED: {e}")

        selector = ModelSelector(
            min_precision=task.policy.min_precision,
            min_recall=task.policy.min_recall,
            primary_metric=task.primary_metric,
            allow_best_fallback=True,
        )

        best_model = selector.select(model_metrics)
        production_path = ARTIFACTS_DIR / f"{task.name}_production.joblib"

        shutil.copy(artifact_paths[best_model], production_path)

        print(f"\nSelected model for {task.name}: {best_model}")
        print(f"Promoted to: {production_path}")


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    train_and_save_risk_models(
        csv_path=Path(DATA_DIR / "nigerian_university_students.csv")
    )
