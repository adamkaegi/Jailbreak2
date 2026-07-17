from abc import ABC, abstractmethod


class Judge(ABC):
    """Context-only interface template for concrete judge implementations.

    ``Judge`` itself is never registered or run. This minimal response-only
    shape documents Adam's original harness; pairwise judge orchestration lives
    in ``judges/runtime.py`` so this template can stay simple.
    """

    name: str

    @abstractmethod
    def apply(self, text: str) -> str:
        ...
