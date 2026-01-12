from pathlib import Path

import pandas as pd

from app.utils.enums import PerformanceTrend


class StudentDataLoader:
    def __init__(self, path: Path):
        self.path = path

    def load_all(self) -> pd.DataFrame:
        df = self._read()
        self._validate(df)
        return df

    def load_latest(self) -> pd.DataFrame:
        df = self._read()
        df_latest = (
            df.sort_values(["student_id", "semester"])
              .groupby("student_id", as_index=False)
              .tail(1)
              .reset_index(drop=True)
        )
        self._validate(df_latest)
        return df_latest
    
    def load_next_score(self) -> pd.DataFrame:
        """Predict next final_score using previous semesters."""
        df = self._read()
        df = df.sort_values(["student_id", "semester"])

        df["prev_final_score"] = df.groupby("student_id")["final_score"].shift(1)
        df["prev_gpa"] = df.groupby("student_id")["prev_gpa"].shift(1)

        # Drop first semester (no history)
        df = df.dropna(subset=["prev_final_score"])

        self._validate(df)

        return df

    def load_performance_trend(self,threshold: float = 3.0) -> pd.DataFrame:
        """Classify student performance"""
        df = self._read()
        df = df.sort_values(["student_id", "semester"])

        df["prev_final_score"] = df.groupby("student_id")["final_score"].shift(1)
        df["score_delta"] = df["final_score"] - df["prev_final_score"]

        def label_trend(delta):
            if pd.isna(delta):
                return None
            if delta > threshold:
                return PerformanceTrend.IMPROVING.value
            elif delta < -threshold:
                return PerformanceTrend.DECLINING.value
            return PerformanceTrend.STABLE.value

        df["performance_trend"] = df["score_delta"].apply(label_trend)

        df = df.dropna(subset=["prev_final_score"])
        self._validate(df)

        return df
    
    def load_early_warning(self) -> pd.DataFrame:
        df = self._read()
        df = df.sort_values(["student_id", "semester"])

        df["next_is_at_risk"] = (
            df.groupby("student_id")["is_at_risk"].shift(-1)
        )

        df = df.dropna(subset=["next_is_at_risk"])
        self._validate(df)

        return df

    def _read(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file is empty: {self.path}")
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {self.path}: {str(e)}")
    
    def _validate(self, df: pd.DataFrame):
        # ---- Level validation ----
        valid_levels = {100, 200, 300, 400, 500}
        invalid_levels = set(df["level"].dropna().unique()) - valid_levels
        if invalid_levels:
            raise ValueError(f"Invalid level values found: {invalid_levels}")

        # ---- Department validation ----
        if df["department"].isnull().any():
            raise ValueError("department column contains null values")
        if not df["department"].apply(lambda x: isinstance(x, str)).all():
            raise ValueError("department column must contain only string values")

        # ---- Attendance & scores ----
        if not df["attendance_pct"].between(0, 100).all():
            raise ValueError("attendance_pct must be between 0 and 100")

        if "prev_gpa" in df.columns:
            if not df["prev_gpa"].dropna().between(0, 5).all():
                raise ValueError("prev_gpa must be between 0 and 5")

        if not df["ca_score"].between(0, 40).all():
            raise ValueError("ca_score must be between 0 and 40")

        if not df["exam_score"].between(0, 60).all():
            raise ValueError("exam_score must be between 0 and 60")

        # ---- Optional target validation (training only) ----
        if "is_at_risk" in df.columns:
            invalid_risk = set(df["is_at_risk"].dropna().unique()) - {0, 1}
            if invalid_risk:
                raise ValueError(f"is_at_risk must be 0 or 1, found: {invalid_risk}")
