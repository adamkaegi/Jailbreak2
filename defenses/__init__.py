from .base import Defense
from .sample_bye_adam_input import SampleByeAdamInputDefense
from .sample_bye_adam_output import SampleByeAdamOutputDefense
from .smoothllm import SmoothLLMDefense
from .self_reminder import SelfReminderDefense
from .none import NoOpDefense

# Available defenses (comma-separate multiple choices on the CLI):
# - smoothllm: perturbs and screens input prompts with the selected target model.
# - self_reminder: adds a safety reminder around the input prompt.
# - sample_bye_adam_input: original input-stage wiring example.
# - sample_bye_adam_output: original output-stage wiring example.
# - none: no-op baseline.
DEFENSES: dict[str, Defense] = {
    d.name: d
    for d in (
        SampleByeAdamInputDefense(),
        SampleByeAdamOutputDefense(),
        SmoothLLMDefense(),
        SelfReminderDefense(),
        NoOpDefense(),
    )
}
