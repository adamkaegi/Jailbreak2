import config

from .base import Judge
from .harmbench_mistral import HarmBenchMistralJudge
from .jailbreak_bench_llama8b import JailbreakBenchLlama8bJudge
from .sample_safe_unsafe import SampleSafeUnsafeJudge
from .strongreject import StrongRejectJudge

# Available judges:
# - sample_safe_unsafe: keyword-only wiring/debug judge.
# - strongreject: continuous 0..1 harmful-assistance evaluator.
# - harmbench_mistral_7b_val_cls: official binary HarmBench validation classifier.
# - jailbreak_bench_llama8b: local Llama 3 8B response classifier.
JUDGES: dict[str, Judge] = {
    judge.name: judge
    for judge in (
        SampleSafeUnsafeJudge(),
        StrongRejectJudge(),
        HarmBenchMistralJudge(),
        JailbreakBenchLlama8bJudge(model_name=config.JUDGE_MODEL),
    )
}
