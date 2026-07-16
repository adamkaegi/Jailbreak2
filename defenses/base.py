from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class DefenseResult:
    """Normalized result returned by a defense.

    Simple text-transform defenses may continue returning a string. Filters use
    this result to stop target inference and attach structured decision metadata.
    """

    text: str
    blocked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Defense(ABC):
    """A defense runs on the prompt ("input" stage) or the model's
    response ("output" stage)."""

    name: str
    stage: Literal["input", "output"]

    @abstractmethod
    def apply(self, text: str) -> str | DefenseResult:
        ...

    def apply_with_context(
        self,
        text: str,
        *,
        original_prompt: str,
        model_prompt: str | None,
    ) -> str | DefenseResult:
        """Apply the defense with optional pipeline context.

        Existing text-only defenses inherit this adapter unchanged. Context-aware
        output filters can override it to inspect both prompt and response.
        """

        return self.apply(text)
