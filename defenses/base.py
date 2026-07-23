from abc import ABC, abstractmethod
from typing import Literal


class Defense(ABC):
    """Context-only interface template for concrete defense implementations.

    ``Defense`` itself is never registered or run. A concrete defense declares
    whether it operates on model input, generation, or model output and
    implements ``apply``.
    """

    name: str
    stage: Literal["input", "generation", "output"]

    @abstractmethod
    def apply(self, text: str) -> str:
        ...
