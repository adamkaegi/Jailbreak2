import csv
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from judges.base import Judge
from judges.runtime import JudgeResult
from scripts.judge_csv import judge_csv


@contextmanager
def _temporary_csv_paths():
    """Yield two writable files without relying on sandboxed temp directories."""
    stem = Path(__file__).parent / f".judge-csv-{uuid4().hex}"
    input_path = stem.with_name(f"{stem.name}-input.csv")
    output_path = stem.with_name(f"{stem.name}-output.csv")
    try:
        yield input_path, output_path
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


class _FixedJudge(Judge):
    name = "fixed"

    def apply(self, text: str) -> JudgeResult:
        return JudgeResult(label="scored", score=0.75)


class _GpuJudge(_FixedJudge):
    requires_target_model_unload = True


class JudgeCsvTests(unittest.TestCase):
    def test_cached_responses_can_be_judged_without_regeneration(self) -> None:
        with _temporary_csv_paths() as (input_path, output_path):
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["model", "input", "output"])
                writer.writeheader()
                writer.writerow({"model": "qwen", "input": "prompt", "output": "response"})

            judge_csv(input_path, output_path, _FixedJudge())

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))

            self.assertEqual(row["model"], "qwen")
            self.assertEqual(row["judge"], "fixed")
            self.assertEqual(row["judge_label"], "scored")
            self.assertEqual(row["judge_score"], "0.750000")

    def test_gpu_judge_unloads_each_recorded_model(self) -> None:
        with _temporary_csv_paths() as (input_path, output_path):
            input_path.write_text(
                "model,input,output\nqwen,prompt,response\nqwen,prompt2,response2\n",
                encoding="utf-8",
            )

            with patch("scripts.judge_csv._unload_ollama_model") as unload:
                judge_csv(input_path, output_path, _GpuJudge())

            unload.assert_called_once_with("qwen")

    def test_existing_output_is_not_overwritten(self) -> None:
        with _temporary_csv_paths() as (input_path, output_path):
            input_path.write_text("input,output\nprompt,response\n", encoding="utf-8")
            output_path.write_text("keep me", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                judge_csv(input_path, output_path, _FixedJudge())

            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep me")


if __name__ == "__main__":
    unittest.main()
