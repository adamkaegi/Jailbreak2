from abc import ABC, abstractmethod


class Attack(ABC):
    """An attack transforms the user prompt before it reaches the model."""

    name: str

    @abstractmethod
    def apply(self, prompt: str) -> str:
        ...
