import uuid
import random
from typing import Union
import numpy as np
import pandas as pd
from app.utils.enums import PerformanceTrend


def generate_synthetic_data(
    num_students: int = 1000,
    semester_per_student: int = 2,
    output_file: str = "nigerian_university_students.csv",
    seed: Union[int, None] = 42,
) -> None:
    """
    Generate realistic synthetic Nigerian university student performance data
    with temporal progression.

    - Academic risk prediction
    - Performance trend detection
    - Level progression (every 2 semesters)

    num_students: Number of unique students
    semester_per_student: Number of semesters per student
    output_file: CSV file to save the generated data
    seed: Random seed for reproducibility
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    departments = [
        "Computer Science",
        "Computer Engineering",
        "Electrical Engineering",
        "Mechanical Engineering",
        "Business Administration",
        "Medicine",
        "Law",
        "Architecture",
        "Economics",
        "Psychology",
        "Sociology",
        "Biology",
        "Political Science",
        "Accounting",
        "Biochemistry",
    ]

    five_hundred_level_departments = {
        "Medicine",
        "Law",
        "Architecture",
    }

    records = []

    for _ in range(num_students):
        student_id = str(uuid.uuid4())
        department = random.choice(departments)

        if department in five_hundred_level_departments:
            start_level = random.choice([100, 200, 300, 400, 500])
            max_level = 500
        else:
            start_level = random.choice([100, 200, 300, 400])
            max_level = 400

        prev_final_score = None

        for semester in range(1, semester_per_student + 1):

            level_increment = (semester - 1) // 2
            current_level = min(start_level + level_increment * 100, max_level)

            attendance = np.clip(np.random.normal(75, 15), 20, 100)
            study_hours = np.clip(np.random.normal(12, 5), 0, 35)
            prev_gpa = np.clip(np.random.normal(3.0, 0.8), 0, 5)

            ca_score = np.clip(
                np.random.normal(15 + (attendance / 100) * 20, 5),
                0,
                40,
            )

            exam_score = np.clip(
                np.random.normal(
                    25 + (study_hours / 35) * 25 + (prev_gpa / 5) * 10,
                    10,
                ),
                0,
                60,
            )

            noise = np.random.normal(0, 3)
            final_score = np.clip(ca_score + exam_score + noise, 0, 100)

            is_at_risk = int(final_score < 45 or attendance < 50)

            if final_score >= 70:
                grade = "A"
            elif final_score >= 60:
                grade = "B"
            elif final_score >= 50:
                grade = "C"
            elif final_score >= 45:
                grade = "D"
            else:
                grade = "F"

            if prev_final_score is None:
                score_delta = 0.0
                performance_trend = PerformanceTrend.STABLE.value
            else:
                score_delta = final_score - prev_final_score
                if score_delta >= 5:
                    performance_trend = PerformanceTrend.IMPROVING.value
                elif score_delta <= -5:
                    performance_trend = PerformanceTrend.DECLINING.value
                else:
                    performance_trend = PerformanceTrend.STABLE.value

            records.append(
                {
                    "student_id": student_id,
                    "semester": semester,
                    "department": department,
                    "level": current_level,
                    "attendance_pct": round(attendance, 1),
                    "study_hours_per_week": round(study_hours, 1),
                    "prev_gpa": round(prev_gpa, 2),
                    "ca_score": round(ca_score, 1),
                    "exam_score": round(exam_score, 1),
                    "prev_final_score": round(prev_final_score, 1) if prev_final_score is not None else None,
                    "final_score": round(final_score, 1),
                    "score_delta": round(score_delta, 1),
                    "performance_trend": performance_trend,
                    "grade": grade,
                    "is_at_risk": is_at_risk,
                }
            )

            prev_final_score = final_score

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)

    print(f"Generated {len(df)} records for {num_students} students.")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_data(num_students=3000, semester_per_student=2)
