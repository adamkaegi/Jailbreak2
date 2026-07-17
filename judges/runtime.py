"""Runtime adapters shared by response-only and pairwise judges.

The original ``judges/base.py`` remains a small context/interface template.
This module owns the richer result format and compatibility logic required by
StrongREJECT, HarmBench, CSV output, and legacy response-only judges.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JudgeResult:
    """Normalized judge output written to an experiment CSV."""

    label: str | None = None
    score: float | None = None

    @classmethod
    def from_value(cls, value: "JudgeResult | str | float") -> "JudgeResult":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(label=value)
        return cls(score=float(value))

    def display(self) -> str:
        parts: list[str] = []
        if self.label is not None:
            parts.append(f"label={self.label}")
        if self.score is not None:
            parts.append(f"score={self.score:.6f}")
        return "  ".join(parts) if parts else "no result"


def evaluate_judge(judge: Any, prompt: str, response: str) -> JudgeResult:
    """Evaluate one pair while supporting the original response-only shape."""
    pairwise_evaluate = getattr(judge, "evaluate", None)
    if callable(pairwise_evaluate):
        return JudgeResult.from_value(pairwise_evaluate(prompt, response))
    return JudgeResult.from_value(judge.apply(response))


def evaluate_judge_batch(
    judge: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
) -> list[JudgeResult]:
    """Evaluate aligned prompt/response sequences and normalize every result."""
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")

    batch_evaluate = getattr(judge, "evaluate_batch", None)
    if callable(batch_evaluate):
        raw_results = list(batch_evaluate(prompts, responses))
    else:
        raw_results = [
            evaluate_judge(judge, prompt, response)
            for prompt, response in zip(prompts, responses)
        ]

    if len(raw_results) != len(prompts):
        raise RuntimeError(
            f"Judge '{judge.name}' returned {len(raw_results)} result(s) for "
            f"{len(prompts)} prompt/response pair(s)"
        )
    return [JudgeResult.from_value(result) for result in raw_results]


def judge_requires_target_unload(judge: Any) -> bool:
    """Return whether the selected local judge needs the target's GPU memory."""
    return bool(getattr(judge, "requires_target_model_unload", False))
