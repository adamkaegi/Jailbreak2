import config

from .base import Judge
from .jailbreak_bench_llama8b import JailbreakBenchLlama8bJudge
from .sample_safe_unsafe import SampleSafeUnsafeJudge

JUDGES: dict[str, Judge] = {
    j.name: j
    for j in (
        SampleSafeUnsafeJudge(),
        JailbreakBenchLlama8bJudge(model_name=config.JUDGE_MODEL),
    )
}