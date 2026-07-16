from .base import Defense


class SampleByeAdamOutputDefense(Defense):
    name = "sample_bye_adam_output"
    stage = "output"

    def apply(self, text: str) -> str:
        return f"{text}\n\nBye Adam"