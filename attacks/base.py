from abc import ABC, abstractmethod


class Attack(ABC):
    """Context-only interface template for concrete attack implementations.

    ``Attack`` itself is never registered or run. Files such as ``none.py``
    and ``deepinception.py`` subclass it and provide the executable behavior.
    """

    name: str

    @abstractmethod
    def apply(self, prompt: str) -> str:
        ...
