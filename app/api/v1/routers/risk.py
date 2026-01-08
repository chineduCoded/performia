from fastapi import APIRouter, HTTPException, status, Depends

from app.schemas.academic_record import AcademicRecord
from app.core.departments import DepartmentValidator
from app.core.dependencies import get_risk_artifact
from app.services.model_artifact import ModelArtifact

router = APIRouter()

@router.post("/predict/risk")
def predict_risk(record: AcademicRecord, artifact=Depends(get_risk_artifact)):
    
    try:
        DepartmentValidator.validate(
            record.department, 
            record.level
        )

        payload = record.model_dump()

        X = artifact.to_matrix(payload)

        prob = float(artifact.predictor.predict_proba(X)[0][1])
        risk = int(artifact.predictor.predict(X)[0])
        
        return {
            "probability": round(prob, 4),
            "risk": risk,
            "model_version": artifact.version,
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input data. Please check your request."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )
