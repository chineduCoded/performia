import pandas as pd


class TemporalFeatureEngineer:
    def build_sequence(self, df: pd.DataFrame):
        """
        Output shape:
        student_id --> [semester_1, semester_2]
        """
        return (
            df.sort_values(["student_id", "semester"])
                .groupby("student_id")
                .apply(lambda x: x[["prev_gpa", "attendance_pct"]].values)
        )