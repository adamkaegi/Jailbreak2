"""Control signal used by input defenses to skip target inference."""

from typing import Any


class DefenseBlocked(Exception):
    """Carry a safe response out of the chain when an input is blocked."""

    def __init__(
        self,
        response: str,
        *,
        defense_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{defense_name} blocked the input")
        self.response = response
        self.defense_name = defense_name
        self.metadata = metadata or {}
