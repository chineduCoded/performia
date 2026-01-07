from pathlib import Path
import pandas as pd


class StudentDataLoader:
    def __init__(self, path: Path):
        self.path = path

    def load_latest_semester(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.path)
            if not pd.api.types.is_numeric_dtype(df["semester"]) and not pd.api.types.is_datetime64_any_dtype(df["semester"]):
                raise ValueError("semester column must be numeric or datetime for chronological sorting")

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file is empty: {self.path}")
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {self.path}: {str(e)}")

        df_latest = (
            df.sort_values(["student_id", "semester"])
              .groupby("student_id", as_index=False)
              .tail(1)
              .reset_index(drop=True)
        )
        self._validate(df_latest)
        return df_latest
    
    def _validate(self, df: pd.DataFrame):
        valid_levels = {100, 200, 300, 400, 500}
        invalid_levels = set(df["level"].unique()) - valid_levels
        if invalid_levels:
            raise ValueError(f"Invalid level values found: {invalid_levels}")

        if df["department"].isnull().any():
            raise ValueError("department column contains null values")
        if not df["department"].apply(lambda x: isinstance(x, str)).all():
            raise ValueError("department column must contain only string values")

        if not df["attendance_pct"].between(0, 100).all():
            raise ValueError("attendance_pct must be between 0 and 100")
        if not df["prev_gpa"].between(0, 5).all():
            raise ValueError("prev_gpa must be between 0 and 5")
        if not df["ca_score"].between(0, 40).all():
            raise ValueError("ca_score must be between 0 and 40")
        if not df["exam_score"].between(0, 60).all():
            raise ValueError("exam_score must be between 0 and 60")

        invalid_risk = set(df["is_at_risk"].unique()) - {0, 1}
        if invalid_risk:
            raise ValueError(f"is_at_risk must be 0 or 1, found: {invalid_risk}")