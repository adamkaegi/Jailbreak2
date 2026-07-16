"""Focused tests for the perplexity input filter; no model downloads."""

import pytest

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

    assert result.text == "hello"
    assert result.blocked is False
    assert scorer.texts == ["hello"]
    assert result.metadata["score"] == score


def test_score_above_threshold_is_blocked():
    defense, _ = make_defense(score=10.01)

    result = defense.apply("hello")

    assert result.text == defense.blocked_response
    assert result.blocked is True
    assert result.metadata["blocked"] is True


def test_unscorable_input_is_neutral_even_with_block_failure_policy():
    scorer = FakeScorer(PerplexityResult(None, 1, 0, "cpu"))
    defense, _ = make_defense(scorer=scorer, failure_policy="block")

    result = defense.apply("x")

    assert result.blocked is False
    assert result.metadata["status"] == "insufficient_tokens"
    assert result.metadata["score"] is None


@pytest.mark.parametrize(
    "policy,blocked",
    [("allow", False), ("block", True)],
)
def test_non_raising_failure_policies(policy, blocked):
    scorer = FakeScorer(error=RuntimeError("scorer broke"))
    defense, _ = make_defense(scorer=scorer, failure_policy=policy)

    result = defense.apply("hello")

    assert result.blocked is blocked
    assert result.metadata["error"] == "RuntimeError: scorer broke"
    assert result.metadata["failure_policy"] == policy


def test_raise_failure_policy_wraps_error():
    scorer = FakeScorer(error=ValueError("bad tokens"))
    defense, _ = make_defense(scorer=scorer, failure_policy="raise")

    with pytest.raises(RuntimeError, match="Perplexity input defense failed: bad tokens"):
        defense.apply("hello")


def test_metadata_preserves_configuration_and_measurement_details():
    defense, _ = make_defense(score=7.5, threshold=11.0)

    metadata = defense.apply("hello").metadata

    assert metadata["stage"] == "input"
    assert metadata["score"] == 7.5
    assert metadata["threshold"] == 11.0
    assert metadata["token_count"] == 5
    assert metadata["predicted_token_count"] == 4
    assert metadata["model"] == "fake-gpt2"
    assert metadata["device"] == "cpu"
    assert metadata["stride"] == 3
    assert metadata["latency_seconds"] >= 0
