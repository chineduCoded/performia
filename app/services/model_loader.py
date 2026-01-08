
import threading
from joblib import load
from pathlib import Path
from app.services.model_artifact import ModelArtifact

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"

from joblib import load
from pathlib import Path
from app.services.model_artifact import ModelArtifact

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"

class ModelLoader:

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self._cache: dict[tuple[str, str], ModelArtifact] = {}
        self._lock = threading.Lock()

    def load(self, name: str, filename: str) -> ModelArtifact:
        with self._lock:
            cache_key = (name, filename)
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            path = self.artifacts_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Model artifact not found: {path}")
            
            artifact = load(path)

            if not isinstance(artifact, ModelArtifact):
                raise TypeError(
                    f"Expected ModelArtifact, got {type(artifact)}"
                )
            
            self._cache[cache_key] = artifact
            return artifact