from abc import ABC, abstractmethod

class BaseModelSpec(ABC):
    @abstractmethod
    def build_pipeline(self):
        pass

class BaseTemporalModelSpec(ABC):
    @abstractmethod
    def build_model(self):
        """Return a temporal model (not necessarily sklearn)"""
        pass