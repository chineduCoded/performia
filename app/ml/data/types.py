from typing import Callable
import pandas as pd
from app.ml.data.loader import StudentDataLoader

DataLoaderFn = Callable[[StudentDataLoader], pd.DataFrame]