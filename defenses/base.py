from abc import ABC, abstractmethod
from typing import Literal


class Defense(ABC):
    """A defense runs on the prompt ("input" stage) or the model's
    response ("output" stage)."""

    name: str
    stage: Literal["input", "output"]

    @abstractmethod
    def apply(self, text: str) -> str:
        ...
