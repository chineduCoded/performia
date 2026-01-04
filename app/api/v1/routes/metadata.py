from fastapi import APIRouter
from app.core.departments import DEPARTMENTS

router = APIRouter()

@router.get("/metadata/departments")
def list_departments():
    return [
        {
            "name": name,
            "allows_500_level": cfg["allows_500_level"]
        }
        for name, cfg in DEPARTMENTS.items()
    ]
