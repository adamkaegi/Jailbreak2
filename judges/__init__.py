from .base import Judge
from .sample_safe_unsafe import SampleSafeUnsafeJudge

JUDGES: dict[str, Judge] = {j.name: j for j in (SampleSafeUnsafeJudge(),)}