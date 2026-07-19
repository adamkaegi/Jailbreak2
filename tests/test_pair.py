import unittest
from types import SimpleNamespace

from attacks.pair import PAIRAttack
from judges.runtime import JudgeResult


class _FakeLLM:
    def __init__(self, responses=None, error=None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def invoke(self, value):
        self.calls.append(value)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.responses.pop(0))


class _FixedJudge:
    def __init__(self, label: str) -> None:
        self.label = label

    def apply(self, text: str) -> JudgeResult:
        return JudgeResult(label=self.label)


class PAIRAttackTests(unittest.TestCase):
    def test_unsafe_judge_result_stops_before_attacker_refinement(self) -> None:
        target = _FakeLLM(["successful target response"])
        attacker = _FakeLLM([])
        attack = PAIRAttack(
            max_rounds=1,
            target_llm=target,
            attacker_llm=attacker,
            judge=_FixedJudge("Unsafe"),
        )

        result = attack.apply("original objective")

        self.assertEqual(result, "original objective")
        self.assertEqual(len(target.calls), 1)
        self.assertEqual(attacker.calls, [])

    def test_safe_result_is_refined_and_extracted(self) -> None:
        target = _FakeLLM(["refusal"])
        attacker = _FakeLLM(
            ["<adversarial_prompt>rewritten candidate</adversarial_prompt>"]
        )
        attack = PAIRAttack(
            max_rounds=1,
            target_llm=target,
            attacker_llm=attacker,
            judge=_FixedJudge("Safe"),
        )

        self.assertEqual(attack.apply("objective"), "rewritten candidate")
        self.assertEqual(len(attacker.calls), 1)

    def test_provider_failure_is_not_silently_treated_as_a_valid_attack(self) -> None:
        attack = PAIRAttack(
            max_rounds=1,
            target_llm=_FakeLLM(error=OSError("offline")),
            attacker_llm=_FakeLLM([]),
            judge=_FixedJudge("Safe"),
        )

        with self.assertRaisesRegex(RuntimeError, "target-model probe failed"):
            attack.apply("objective")

    def test_common_refusal_does_not_stop_as_a_false_success(self) -> None:
        target = _FakeLLM(
            ["I'm sorry, but I am unable to provide instructions for harming someone."]
        )
        attacker = _FakeLLM(
            ["<adversarial_prompt>rewritten candidate</adversarial_prompt>"]
        )
        attack = PAIRAttack(
            max_rounds=1,
            target_llm=target,
            attacker_llm=attacker,
        )

        self.assertEqual(attack.apply("objective"), "rewritten candidate")


if __name__ == "__main__":
    unittest.main()
