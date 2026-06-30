from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Base class for all model wrappers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Any = None

    @abstractmethod
    def load(self) -> None:
        """Load the underlying model."""
        pass