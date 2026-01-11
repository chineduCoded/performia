from typing import Union
from pydantic import ValidationError
from fastapi.exceptions import RequestValidationError

SENSITIVE_FIELDS = {"password", "token", "api_key", "secret", "authorization"}

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
        
        loc_parts = [str(l).lower() for l in err.get("loc", [])]
        is_sensitive = any(field in part for part in loc_parts for field in SENSITIVE_FIELDS)
        
        if is_sensitive:
            input_display = "<REDACTED>"
        else:
            input_str = str(input_val)
            input_display = input_str if len(input_str) <= 100 else input_str[:100] + "..."
        
        errors.append(f"{loc}: {msg} (input={input_display})")
    return errors