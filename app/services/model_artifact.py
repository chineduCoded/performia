from dataclasses import dataclass
from typing import Any, List

@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    threshold: float
    features: List[str]
    version: str
