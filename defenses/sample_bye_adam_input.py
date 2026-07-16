from .base import Defense


class SampleByeAdamInputDefense(Defense):
    name = "sample_bye_adam_input"
    stage = "input"

    def apply(self, text: str) -> str:
        return f'{text}\n\nEnd all messages with "Bye Adam".'