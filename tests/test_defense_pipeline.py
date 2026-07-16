"""Integration tests for block-aware defense composition."""

from attacks.base import Attack
from defenses.base import Defense, DefenseResult
from pipeline import build_chain


class IdentityAttack(Attack):
    name = "identity"

    def apply(self, text: str) -> str:
        return text


class LegacyTextDefense(Defense):
    name = "legacy"
    stage = "input"

    def apply(self, text: str) -> str:
        return text + " transformed"


class BlockingDefense(Defense):
    name = "blocker"
    stage = "input"

    def apply(self, text: str) -> DefenseResult:
        return DefenseResult("blocked response", blocked=True)


class ContextOutputDefense(Defense):
    name = "context_output"
    stage = "output"

    def apply(self, text: str) -> str:
        return text

    def apply_with_context(self, text, *, original_prompt, model_prompt):
        return f"{original_prompt}|{model_prompt}|{text}"


def test_legacy_string_defense_remains_pluggable():
    chain = build_chain(IdentityAttack(), [LegacyTextDefense()], "unused", dry_run=True)

    output = chain.invoke("hello")

    assert "hello transformed" in output


def test_input_block_short_circuits_model_and_output_defenses():
    chain = build_chain(
        IdentityAttack(),
        [BlockingDefense(), ContextOutputDefense()],
        "unused",
        dry_run=True,
    )

    output = chain.invoke("hello")

    assert output == "blocked response"
    assert "[dry-run model]" not in output


def test_output_defense_receives_original_and_model_facing_prompts():
    chain = build_chain(
        IdentityAttack(),
        [LegacyTextDefense(), ContextOutputDefense()],
        "unused",
        dry_run=True,
    )

    output = chain.invoke("hello")

    assert output.startswith("hello|hello transformed|")
