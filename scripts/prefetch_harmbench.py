"""Download and smoke-test the pinned HarmBench Mistral validation judge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from judges.harmbench_mistral import HarmBenchMistralJudge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the approximately 14.5 GB classifier to run without CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA-enabled PyTorch is required for a practical HarmBench run. "
            "Install requirements-cuda.txt, or pass --allow-cpu deliberately."
        )

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"HarmBench evaluator device: {device}")

    judge = HarmBenchMistralJudge(batch_size=1, allow_cpu=args.allow_cpu)
    try:
        result = judge.evaluate(
            "Provide instructions for an explicitly harmful act.",
            "I cannot help with that request.",
        )
    except RuntimeError as exc:
        raise SystemExit(f"HarmBench setup failed: {exc}") from None

    print(f"HarmBench is ready ({result.display()}).")


if __name__ == "__main__":
    main()
