
import threading
from joblib import load
from pathlib import Path
from app.services.model_artifact import ModelArtifact

BASE_DIR = Path(__file__).resolve().parents[1]  # app/
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"

from joblib import load
from pathlib import Path
from app.services.model_artifact import ModelArtifact

BASE_DIR = Path(__file__).resolve().parents[1]  # app/
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"

class ModelLoader:
    _models: dict[str, ModelArtifact] = {}
    _lock = threading.Lock()

    @classmethod
    def load_model(cls, model_name: str, filename: str) -> ModelArtifact:
        with cls._lock:
            if model_name not in cls._models:
                model_path = ARTIFACTS_DIR / filename

                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                

                artifact = load(model_path)


                if isinstance(artifact, dict):
                    required_keys = {"model", "features", "model_version"}
                    missing_keys = required_keys - artifact.keys()
                    
                    if missing_keys:
                        raise ValueError(f"Artifact missing required keys: {missing_keys}")
                    
                    model_artifact = ModelArtifact(
                        model=artifact["model"],
                        threshold=artifact.get("threshold", 0.35),
                        features=artifact["features"],
                        version=artifact["model_version"],
                    )
                elif isinstance(artifact, ModelArtifact):
                    model_artifact = artifact
                else:
                    raise TypeError("Unsupported artifact type: {type(artifact)}")
                
                cls._models[model_name] = model_artifact

        return cls._models[model_name]