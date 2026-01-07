from pathlib import Path
import pandas as pd


class StudentDataLoader:
    def __init__(self, path: Path):
        self.path = path

    def load_latest_semester(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)

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
        assert set(df["level"].unique()).issubset(valid_levels)

        assert df["department"].notnull().all()
        assert df["department"].apply(lambda x: isinstance(x, str)).all()

        assert df["attendance_pct"].between(0, 100).all()
        assert df["prev_gpa"].between(0, 5).all()
        assert df["ca_score"].between(0, 40).all()
        assert df["exam_score"].between(0, 60).all()

        assert set(df["is_at_risk"].unique()) <= {0, 1}