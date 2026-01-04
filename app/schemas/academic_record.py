from pydantic import BaseModel, Field, field_validator, ValidationInfo
from app.utils.enums import Level

class AcademicRecord(BaseModel):
    department: str
    level: Level

    attendance_pct: float = Field(..., ge=0, le=100)
    study_hours_per_week: float = Field(..., ge=0, le=40)
    prev_gpa: float = Field(..., ge=0.0, le=5.0)

    ca_score: float = Field(..., ge=0, le=40)
    exam_score: float = Field(..., ge=0, le=60)
    
    @field_validator("exam_score", mode="after")
    @classmethod
    def exam_ca_consistency(cls, exam: float, info: ValidationInfo) -> float:
        ca = info.data.get("ca_score", 0)
        if ca + exam > 100:
            raise ValueError("CA + Exam score cannot exceed 100")
        return exam

class DepartmentMetadata(BaseModel):
     name: str
     allows_500_level: bool