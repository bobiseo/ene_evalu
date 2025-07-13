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


# from abc import ABC, abstractmethod

# import pandas as pd
# import numpy as np


# class BaseModel(ABC):
#     def __init__(self) -> None:
#         ...


#     @abstractmethod
#     def train(self) -> None:
#         """
#         Train the model using ML Models for Multi-class and mult-label classification.
#         :params: df is essential, others are model specific
#         :return: classifier
#         """
#         ...

#     @abstractmethod
#     def predict(self) -> int:
#         """

#         """
#         ...

#     @abstractmethod
#     def data_transform(self) -> None:
#         return

#     # def build(self, values) -> BaseModel:
#     def build(self, values={}):
#         values = values if isinstance(values, dict) else utils.string2any(values)
#         self.__dict__.update(self.defaults)
#         self.__dict__.update(values)
#         return self