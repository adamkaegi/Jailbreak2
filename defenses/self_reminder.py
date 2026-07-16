from .base import Defense


class SelfReminderDefense(Defense):
    name = "self_reminder"
    stage = "input"

    def __init__(
        self,
        prefix: str = (
            "You are a responsible, helpful, and harmless assistant. "
            "Follow safety rules and refuse harmful requests."
        ),
        suffix: str = "Before answering, review your response and make sure it is safe.",
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix

    def apply(self, text: str) -> str:
        return f"{self.prefix}\n\n{text}\n\n{self.suffix}"