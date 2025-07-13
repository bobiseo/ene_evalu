# model/base.py
from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """
    Abstract base class for all models in the registry.

    Implementations must provide:
      • train(self, data)   -> None
      • predict(self, X)    -> Any (e.g., dict of arrays)
      • print_results(self, data) -> None (optional)
    """

    @abstractmethod
    def train(self, data: Any) -> None:
        """
        Train the model on the provided Data object.
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predict using the provided feature matrix or data.
        Returns predictions (e.g., dict or array).
        """
        pass

    def print_results(self, data: Any) -> None:
        """
        Optional: print evaluation metrics based on the Data object.
        By default, does nothing. Override in subclasses if needed.
        """
        pass

