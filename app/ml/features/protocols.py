from typing import TYPE_CHECKING, Protocol
import pandas as pd
from app.ml.features.types import FeatureGroups

if TYPE_CHECKING:
    from app.ml.data.loader import StudentDataLoader


class FeatureEngineer(Protocol):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @property
    def feature_groups(self) -> FeatureGroups:
        ...

    @property
    def feature_columns(self) -> list[str]:
        ...


class DataLoaderFn(Protocol):
    def __call__(self, loader: "StudentDataLoader") -> pd.DataFrame: 
        ...