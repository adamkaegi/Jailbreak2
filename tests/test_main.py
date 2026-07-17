import argparse
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
import main
from judges.runtime import JudgeResult


class _CountingAttack:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def apply(self, prompt: str) -> str:
        self.calls += 1
        return f"{prompt}|attack-{self.calls}"


class _RunAwareDefense:
    name = "run_aware"
    stage = "input"

    def __init__(self) -> None:
        self.configure_calls: list[tuple[str, bool]] = []

    def for_run(self, model_name: str, dry_run: bool = False):
        self.configure_calls.append((model_name, dry_run))
        return self


class _FakeChain:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def invoke(self, text: str, **kwargs) -> str:
        self.inputs.append(text)
        return f"response to {text}"


class _FakeJudge:
    name = "fake"
    requires_target_model_unload = False

    def __init__(self, results=None) -> None:
        self.results = (
            [JudgeResult(label="Safe", score=0.0)] if results is None else results
        )
        self.calls = 0

    def evaluate_batch(self, prompts, responses):
        self.calls += 1
        return self.results


class MainIntegrationTests(unittest.TestCase):
    def _args(self, *, dry_run: bool) -> argparse.Namespace:
        return argparse.Namespace(
            prompt="",
            model="chosen-model",
            attack="counting",
            defense=["run_aware"],
            judge="fake",
            batch="general",
            dry_run=dry_run,
        )

    def test_attack_is_applied_once_and_exact_output_reaches_chain(self) -> None:
        attack = _CountingAttack()
        defense = _RunAwareDefense()
        judge = _FakeJudge()
        chain = _FakeChain()
        write_csv = Mock(return_value=Path("cached.csv"))

        with (
            patch.object(main, "parse_args", return_value=self._args(dry_run=True)),
            patch.dict(main.ATTACKS, {"counting": attack}),
            patch.dict(main.DEFENSES, {"run_aware": defense}),
            patch.dict(main.JUDGES, {"fake": judge}),
            patch.object(main, "build_response_chain", return_value=chain) as build_chain,
            patch.object(main, "_load_langfuse_handler", return_value=(None, None)),
            patch.object(main, "_write_run_csv", write_csv),
            patch.object(main, "_append_run_csv_row") as append_row,
            patch.object(main, "load_batch") as load_batch,
        ):
            main.main()

        self.assertEqual(attack.calls, 1)
        self.assertEqual(chain.inputs, ["|attack-1"])
        self.assertEqual(defense.configure_calls, [("chosen-model", True)])
        build_chain.assert_called_once_with([defense], "chosen-model", dry_run=True)
        load_batch.assert_not_called()
        self.assertEqual(judge.calls, 0)
        self.assertEqual(append_row.call_args.args[1]["attacked_input"], "|attack-1")
        self.assertEqual(write_csv.call_count, 2)

    def test_short_judge_result_list_fails_after_checkpoint(self) -> None:
        attack = _CountingAttack()
        defense = _RunAwareDefense()
        judge = _FakeJudge(results=[])
        chain = _FakeChain()

        with (
            patch.object(main, "parse_args", return_value=self._args(dry_run=False)),
            patch.dict(main.ATTACKS, {"counting": attack}),
            patch.dict(main.DEFENSES, {"run_aware": defense}),
            patch.dict(main.JUDGES, {"fake": judge}),
            patch.object(main, "build_response_chain", return_value=chain),
            patch.object(main, "_load_langfuse_handler", return_value=(None, None)),
            patch.object(main, "_write_run_csv", return_value=Path("cached.csv")),
            patch.object(main, "_append_run_csv_row") as append_row,
        ):
            with self.assertRaisesRegex(RuntimeError, "responses remain cached"):
                main.main()

        append_row.assert_called_once()

    def test_gpu_judge_unloads_selected_target_after_checkpoint(self) -> None:
        attack = _CountingAttack()
        defense = _RunAwareDefense()
        judge = _FakeJudge()
        judge.requires_target_model_unload = True
        chain = _FakeChain()

        with (
            patch.object(main, "parse_args", return_value=self._args(dry_run=False)),
            patch.dict(main.ATTACKS, {"counting": attack}),
            patch.dict(main.DEFENSES, {"run_aware": defense}),
            patch.dict(main.JUDGES, {"fake": judge}),
            patch.object(main, "build_response_chain", return_value=chain),
            patch.object(main, "_load_langfuse_handler", return_value=(None, None)),
            patch.object(main, "_write_run_csv", return_value=Path("cached.csv")),
            patch.object(main, "_append_run_csv_row"),
            patch.object(main, "_unload_ollama_model") as unload,
        ):
            main.main()

        unload.assert_called_once_with("chosen-model")


if __name__ == "__main__":
    unittest.main()
