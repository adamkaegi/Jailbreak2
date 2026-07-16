from abc import ABC, abstractmethod


class Judge(ABC):
    """A judge labels the model output as safe or unsafe."""

    name: str

    @abstractmethod
    def apply(self, text: str) -> str:
        ...