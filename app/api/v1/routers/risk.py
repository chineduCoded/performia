import logging

from fastapi import APIRouter, HTTPException, status, Depends

from app.schemas.academic_record import AcademicRecord
from app.core.departments import DepartmentValidator
from app.core.dependencies import get_risk_artifact
from app.utils.enums import RiskClassification
from app.schemas.prediction import RiskPredictionResponse
from app.ml.inference.risk_interpreter import interpret_risk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"], prefix="/predict")

@router.post(
    "/risk",
    summary="Predict student academic risk",
    description="Predicts early student risk based on academic record.",
    response_model=RiskPredictionResponse
)
def predict_risk(
    record: AcademicRecord,
    artifact=Depends(get_risk_artifact)
) -> RiskPredictionResponse:  
    try:
        DepartmentValidator.validate(
            record.department, 
            record.level
        )

        payload = record.model_dump()

        X = artifact.prepare_features(payload)

        proba = artifact.predictor.predict_proba(X)

        if proba.ndim == 1:
            prob = float(proba[0])
        else:
            prob = float(proba[0, 1])

        risk = int(artifact.predictor.predict(X)[0])
        risk_class = RiskClassification.AT_RISK if prob >= 0.5 else RiskClassification.NOT_AT_RISK
        
        interpretation = interpret_risk(prob)

        result =  {
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100, 2),
            "predicted_risk": risk,
            "risk_class": risk_class,
            "risk_label": interpretation["risk_label"],
            "advice": interpretation["advice"],
            "severity": interpretation["severity"],
            "model_version": artifact.version,
        }

        logger.info(
            "Risk prediction",
            extra={
                "department": record.department,
                "level": record.level,
                "probability": prob,
                "risk": risk_class.value
            }
        )

        return RiskPredictionResponse(**result)
    
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid features: {ve}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed"
        )
