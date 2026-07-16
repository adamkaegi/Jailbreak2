"""Focused tests for the perplexity input filter; no model downloads."""

import pytest

from defenses.block import DefenseBlocked
from defenses.perplexity import PerplexityDefense, PerplexityResult


class FakeScorer:
    model_name = "fake-gpt2"
    requested_device = "cpu"
    stride = 3

    def __init__(self, result=None, error=None):
        self.result = result or PerplexityResult(10.0, 5, 4, "cpu")
        self.error = error
        self.texts = []

    def score(self, text):
        self.texts.append(text)
        if self.error:
            raise self.error
        return self.result


def make_defense(score=10.0, threshold=10.0, **kwargs):
    scorer = kwargs.pop(
        "scorer", FakeScorer(PerplexityResult(score, 5, 4, "cpu"))
    )
    return PerplexityDefense(scorer=scorer, threshold=threshold, **kwargs), scorer


@pytest.mark.parametrize("score", [9.9, 10.0])
def test_score_at_or_below_threshold_is_allowed(score):
    defense, scorer = make_defense(score=score)

    result = defense.apply("hello")

    assert result == "hello"
    assert scorer.texts == ["hello"]


def test_score_above_threshold_is_blocked():
    defense, _ = make_defense(score=10.01)

    with pytest.raises(DefenseBlocked) as blocked:
        defense.apply("hello")

    assert blocked.value.response == defense.blocked_response
    assert blocked.value.defense_name == "perplexity"
    assert blocked.value.metadata["blocked"] is True
    assert blocked.value.metadata["score"] == 10.01


def test_unscorable_input_is_neutral_even_with_block_failure_policy():
    scorer = FakeScorer(PerplexityResult(None, 1, 0, "cpu"))
    defense, _ = make_defense(scorer=scorer, failure_policy="block")

    result = defense.apply("x")

    assert result == "x"


def test_allow_failure_policy_returns_original_text():
    scorer = FakeScorer(error=RuntimeError("scorer broke"))
    defense, _ = make_defense(scorer=scorer, failure_policy="allow")

    assert defense.apply("hello") == "hello"


def test_block_failure_policy_raises_block_signal_with_error_metadata():
    scorer = FakeScorer(error=RuntimeError("scorer broke"))
    defense, _ = make_defense(scorer=scorer, failure_policy="block")

    with pytest.raises(DefenseBlocked) as blocked:
        defense.apply("hello")

    assert blocked.value.metadata["error"] == "RuntimeError: scorer broke"
    assert blocked.value.metadata["failure_policy"] == "block"


def test_raise_failure_policy_wraps_error():
    scorer = FakeScorer(error=ValueError("bad tokens"))
    defense, _ = make_defense(scorer=scorer, failure_policy="raise")

    with pytest.raises(RuntimeError, match="Perplexity input defense failed: bad tokens"):
        defense.apply("hello")


def test_block_metadata_preserves_configuration_and_measurement_details():
    defense, _ = make_defense(score=12.0, threshold=11.0)

    with pytest.raises(DefenseBlocked) as blocked:
        defense.apply("hello")
    metadata = blocked.value.metadata

    assert metadata["stage"] == "input"
    assert metadata["score"] == 12.0
    assert metadata["threshold"] == 11.0
    assert metadata["token_count"] == 5
    assert metadata["predicted_token_count"] == 4
    assert metadata["model"] == "fake-gpt2"
    assert metadata["device"] == "cpu"
    assert metadata["stride"] == 3
    assert metadata["latency_seconds"] >= 0
