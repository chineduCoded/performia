from typing import Union
from pydantic import ValidationError
from fastapi.exceptions import RequestValidationError

def format_validation_error(
    exc: Union[ValidationError, RequestValidationError]
) -> list[str]:
    """
    Convert Pydantic ValidationError into a readable message list.
    """
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "")
        input_val = err.get("input", "")
        errors.append(f"{loc}: {msg} (input={input_val})")
    return errors