from pydantic import BaseModel

class AcademicRecord(BaseModel):
    department: str
    level: int
    attendance_pct: float
    study_hours_per_week: float
    prev_gpa: float
    ca_score: float
    exam_score: float
