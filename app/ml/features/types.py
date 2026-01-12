from typing import TypedDict, Tuple


class FeatureGroups(TypedDict):
    numeric: Tuple[str, ...]
    categorical: Tuple[str, ...]