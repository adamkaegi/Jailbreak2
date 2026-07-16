"""Focused Llama Guard tests; no Ollama calls."""

import pytest

from defenses.llama_guard import (
    GuardDecision,
    LlamaGuardInputDefense,
    LlamaGuardOutputDefense,
    build_input_guard_conversation,
    build_output_guard_conversation,
    parse_guard_output,
)


class FakeClassifier:
    model_name = "llama-guard3:1b"

    def __init__(self, decision):
        self.decision = decision
        self.conversations = []

    def classify_conversation(self, conversation):
        self.conversations.append(conversation)
        return self.decision


def decision(label, categories=None, status="parsed", error=""):
    raw = label if label != "unsafe" else "unsafe\n" + ",".join(categories or [])
    return GuardDecision(label, categories or [], raw, status, error, model="fake")


@pytest.mark.parametrize(
    "raw,label,categories",
    [
        ("safe", "safe", []),
        ("SAFE", "safe", []),
        ("unsafe", "unsafe", []),
        ("unsafe\nS1", "unsafe", ["S1"]),
        ("unsafe\nS1,S2,S1", "unsafe", ["S1", "S2"]),
        ("unsafe\nS13", "unsafe", ["S13"]),
    ],
)
def test_parse_valid_outputs(raw, label, categories):
    result = parse_guard_output(raw)

    assert result.label == label
    assert result.categories == categories
    assert result.parse_status == "parsed"
    assert result.error == ""


@pytest.mark.parametrize(
    "raw",
    ["", "moderation result: safe", "unsafe\nS0", "unsafe\nS14", "safe\nS1"],
)
def test_parse_rejects_malformed_outputs(raw):
    result = parse_guard_output(raw)

    assert result.label == "unknown"
    assert result.parse_status == "malformed"
    assert result.error


def test_conversation_builders_preserve_native_roles():
    assert build_input_guard_conversation("hello") == [
        {"role": "user", "content": "hello"}
    ]
    assert build_output_guard_conversation("hello", "answer") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
    ]


def test_safe_input_is_preserved():
    classifier = FakeClassifier(decision("safe"))
    defense = LlamaGuardInputDefense(classifier=classifier)

    result = defense.apply("hello")

    assert result.text == "hello"
    assert result.blocked is False
    assert classifier.conversations[0][-1]["content"] == "hello"


def test_unsafe_input_is_blocked_with_categories():
    classifier = FakeClassifier(decision("unsafe", ["S5"]))
    defense = LlamaGuardInputDefense(classifier=classifier)

    result = defense.apply("bad prompt")

    assert result.text == defense.blocked_response
    assert result.blocked is True
    assert result.metadata["decision"]["categories"] == ["S5"]
    assert result.metadata["response_replaced"] is False


def test_safe_output_uses_model_facing_prompt_and_preserves_response():
    classifier = FakeClassifier(decision("safe"))
    defense = LlamaGuardOutputDefense(classifier=classifier)

    result = defense.apply_with_context(
        "target answer",
        original_prompt="raw prompt",
        model_prompt="attacked prompt",
    )

    assert result.text == "target answer"
    assert result.blocked is False
    assert classifier.conversations == [[
        {"role": "user", "content": "attacked prompt"},
        {"role": "assistant", "content": "target answer"},
    ]]


def test_unsafe_output_is_replaced_and_original_is_recorded():
    classifier = FakeClassifier(decision("unsafe", ["S2"]))
    defense = LlamaGuardOutputDefense(classifier=classifier)

    result = defense.apply_with_context(
        "unsafe target answer",
        original_prompt="prompt",
        model_prompt="prompt",
    )

    assert result.text == defense.blocked_response
    assert result.blocked is True
    assert result.metadata["response_replaced"] is True
    assert result.metadata["original_response"] == "unsafe target answer"


@pytest.mark.parametrize("policy,blocked", [("allow", False), ("block", True)])
def test_unknown_decision_follows_failure_policy(policy, blocked):
    classifier = FakeClassifier(decision("unknown", status="provider_error", error="down"))
    defense = LlamaGuardInputDefense(classifier=classifier, failure_policy=policy)

    assert defense.apply("hello").blocked is blocked


def test_unknown_decision_can_raise():
    classifier = FakeClassifier(decision("unknown", status="malformed", error="bad label"))
    defense = LlamaGuardInputDefense(classifier=classifier, failure_policy="raise")

    with pytest.raises(RuntimeError, match="Llama Guard input classification failed"):
        defense.apply("hello")
