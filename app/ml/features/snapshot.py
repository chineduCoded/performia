import pandas as pd

class SnapshotFeatureEngineer:
    NUMERIC = [
        "attendance_pct",
        "study_hours_per_week",
        "prev_gpa",
        "ca_score",
        "exam_score",
        "level",
    ]
    CATEGORICAL = ["department"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["study_effort_ratio"] = (
            df["study_hours_per_week"] / 35
        ).clip(0, 1.5)

        df["attendance_risk_flag"] = (df["attendance_pct"] < 50).astype(int)

        return df

    @property
    def feature_columns(self):
        return (
            self.NUMERIC
            + self.CATEGORICAL
            # + ["study_effort_ratio", "attendance_risk_flag"]
        )
