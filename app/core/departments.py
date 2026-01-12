from typing import Dict

DEPARTMENTS: Dict[str, dict] = {
    "computer science": {"allows_500_level": False},
    "computer engineering": {"allows_500_level": False},
    "electrical engineering": {"allows_500_level": False},
    "mechanical engineering": {"allows_500_level": False},
    "business administration": {"allows_500_level": False},
    "economics": {"allows_500_level": False},
    "psychology": {"allows_500_level": False},
    "sociology": {"allows_500_level": False},
    "biology": {"allows_500_level": False},
    "political science": {"allows_500_level": False},
    "accounting": {"allows_500_level": False},
    "biochemistry": {"allows_500_level": False},

    "medicine": {"allows_500_level": True},
    "law": {"allows_500_level": True},
    "architecture": {"allows_500_level": True},
}



class DepartmentValidator:
    @staticmethod
    def validate(department_name: str, level: int) -> None:
        if not isinstance(department_name, str):
            raise TypeError(
                f"Department name must be a string, got {type(department_name).__name__}"
            )

        normalized_name = department_name.strip().lower()
        dept = DEPARTMENTS.get(normalized_name)

        if dept is None:
            raise ValueError(f"Department '{department_name}' is not recognized.")

        if level == 500 and not dept.get("allows_500_level", False):
            raise ValueError(
                f"{department_name} does not allow 500 level"
            )