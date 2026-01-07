
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
    _cache: dict[str, ModelArtifact] = {}
    _lock = threading.Lock()

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir

    def load(self, name: str, filename: str):
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            
            path = self.artifacts_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Model artifact not found: {path}")
            
            artifact = load(path)

            if not isinstance(artifact, ModelArtifact):
                raise TypeError(
                    f"Expected ModelArtifact, got {type(artifact)}"
                )
            
            self._cache[name] = artifact
            return artifact