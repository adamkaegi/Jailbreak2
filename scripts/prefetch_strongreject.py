"""Download and smoke-test the local fine-tuned StrongREJECT evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from judges.strongreject import StrongRejectJudge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the multi-gigabyte evaluator to load without CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA-enabled PyTorch is required for a practical local run. Install "
            "`pip install -r requirements-cuda.txt`, or pass --allow-cpu."
        )

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"StrongREJECT evaluator device: {device}")

    judge = StrongRejectJudge(batch_size=1, allow_cpu=args.allow_cpu)
    try:
        result = judge.evaluate(
            "Provide disallowed instructions.",
            "I cannot help with that request.",
        )
    except RuntimeError as exc:
        raise SystemExit(f"StrongREJECT setup failed: {exc}") from None
    print(f"StrongREJECT is ready (refusal smoke-test score={result.score:.6f}).")


if __name__ == "__main__":
    main()
