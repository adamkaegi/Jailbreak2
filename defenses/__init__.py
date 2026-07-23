from .base import Defense
from .sample_bye_adam_input import SampleByeAdamInputDefense
from .sample_bye_adam_output import SampleByeAdamOutputDefense
from .smoothllm import SmoothLLMDefense
from .self_reminder import SelfReminderDefense
from .none import NoOpDefense
from .perplexity import PerplexityDefense
from .llama_guard import LlamaGuardInputDefense, LlamaGuardOutputDefense

# Available defenses (comma-separate multiple choices on the CLI):
# - smoothllm: generates perturbed responses and selects from the majority class.
# - self_reminder: adds a safety reminder around the input prompt.
# - perplexity: blocks high-perplexity input prompts.
# - llama_guard_input: blocks prompts classified unsafe by Llama Guard.
# - llama_guard_output: replaces outputs classified unsafe by Llama Guard.
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
        PerplexityDefense(),
        LlamaGuardInputDefense(),
        LlamaGuardOutputDefense(),
    )
}
