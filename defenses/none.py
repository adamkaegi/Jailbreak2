from .base import Defense


class NoOpDefense(Defense):
    name = "none"
    stage = "input"

    def apply(self, text: str) -> str:
        return text
