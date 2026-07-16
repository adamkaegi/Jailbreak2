from .base import Attack


class NoOpAttack(Attack):
    name = "none"

    def apply(self, prompt: str) -> str:
        return prompt
