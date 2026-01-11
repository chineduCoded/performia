from typing import Dict

DEPARTMENTS: Dict[str, dict] = {
    "Computer Science": {"allows_500_level": False},
    "Computer Engineering": {"allows_500_level": False},
    "Electrical Engineering": {"allows_500_level": False},
    "Mechanical Engineering": {"allows_500_level": False},
    "Business Administration": {"allows_500_level": False},
    "Economics": {"allows_500_level": False},
    "Psychology": {"allows_500_level": False},
    "Sociology": {"allows_500_level": False},
    "Biology": {"allows_500_level": False},
    "Political Science": {"allows_500_level": False},
    "Accounting": {"allows_500_level": False},
    "Biochemistry": {"allows_500_level": False},

    # Special cases
    "Medicine": {"allows_500_level": True},
    "Law": {"allows_500_level": True},
    "Architecture": {"allows_500_level": True},
}


class DepartmentValidator:
    @staticmethod
    def validate(department_name: str, level: int) -> None:
        if not isinstance(department_name, str):
            raise TypeError(
                f"Department name must be a string, got {type(department_name).__name__}"
            )
        
        dept = DEPARTMENTS.get(department_name)

        if dept is None:
            raise ValueError(f"Department '{department_name}' is not recognized.")
        
        if level == 500 and not dept.get("allows_500_level", False):
            raise ValueError(
                f"{department_name} does not allow 500 level"
            )