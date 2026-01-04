from fastapi import APIRouter, HTTPException, status
from app.schemas.academic_record import AcademicRecord
from app.services.risk_predictor import predict_risk
from app.core.departments import DepartmentValidator

router = APIRouter()

@router.post("/predict/risk")
def predict(record: AcademicRecord):
    try:
        record = AcademicRecord.model_validate(record)
        DepartmentValidator.validate(
            record.department, 
            record.level
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    
    return predict_risk(record.model_dump())
