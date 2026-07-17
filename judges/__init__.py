from .base import Judge
from .harmbench_mistral import HarmBenchMistralJudge
from .sample_safe_unsafe import SampleSafeUnsafeJudge
from .strongreject import StrongRejectJudge

# Available judges:
# - sample_safe_unsafe: keyword-only wiring/debug judge.
# - strongreject: continuous 0..1 harmful-assistance evaluator.
# - harmbench_mistral_7b_val_cls: official binary HarmBench validation classifier.
JUDGES: dict[str, Judge] = {
    judge.name: judge
    for judge in (
        SampleSafeUnsafeJudge(),
        StrongRejectJudge(),
        HarmBenchMistralJudge(),
    )
}
