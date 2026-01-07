from typing import List

from fastapi import APIRouter
from app.core.departments import DEPARTMENTS
from app.schemas.academic_record import DepartmentMetadata

router = APIRouter()

@router.get(
    "/metadata/departments",
    response_model=List[DepartmentMetadata],
    status_code=200,
    description="Retrieve metadata for all departments"
)
def list_departments() -> List[DepartmentMetadata]:
    """Returns a list of all departments with their configuration metadata."""
    return [

    DepartmentMetadata(
        name=name,
        allows_500_level=cfg.get("allows_500_level", False)
    )
        for name, cfg in DEPARTMENTS.items()
    ]
