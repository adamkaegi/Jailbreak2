from .base import Defense
from .sample_bye_adam_input import SampleByeAdamInputDefense
from .sample_bye_adam_output import SampleByeAdamOutputDefense
from .smoothllm import SmoothLLMDefense
from .self_reminder import SelfReminderDefense
from .none import NoOpDefense
from .perplexity import PerplexityDefense
from .llama_guard import LlamaGuardInputDefense, LlamaGuardOutputDefense

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
