from dataclasses import dataclass
from app.ml.features.protocols import FeatureEngineer
from app.ml.data.types import DataLoaderFn


from dataclasses import dataclass
from app.ml.features.protocols import FeatureEngineer
from app.ml.data.types import DataLoaderFn


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    min_precision: float
    min_recall: float
    beta: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_precision <= 1.0):
            raise ValueError("min_precision must be in [0, 1]")
        if not (0.0 <= self.min_recall <= 1.0):
            raise ValueError("min_recall must be in [0, 1]")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")


@dataclass(frozen=True, slots=True)
class RiskTask:
    name: str
    loader: DataLoaderFn
    feature_engineer: FeatureEngineer
    target: str
    policy: RiskPolicy

    @property
    def primary_metric(self) -> str:
        return f"f{self.policy.beta}"