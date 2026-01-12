import pandas as pd

from app.ml.features.types import FeatureGroups

class SnapshotFeatureEngineer:
    NUMERIC = (
        "attendance_pct",
        "study_hours_per_week",
        "prev_gpa",
        "ca_score",
        "exam_score",
        "level",
    )
    CATEGORICAL = ("department",)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["study_hours_per_week", "attendance_pct"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()

        df["study_effort_ratio"] = (
            df["study_hours_per_week"] / 35
        ).clip(0, 1.5)

        df["attendance_risk_flag"] = (df["attendance_pct"] < 50).astype(int)

        return df
    
    @property
    def feature_groups(self) -> FeatureGroups:
        return {
            "numeric": self.NUMERIC,
            "categorical": self.CATEGORICAL,
        }

    @property
    def feature_columns(self) -> list[str]:
        return [
            *self.NUMERIC,
            *self.CATEGORICAL,
        ]