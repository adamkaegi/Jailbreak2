"""Fetch the pinned official JBB harmful/benign goals as harness text batches."""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
JBB_DATA_REVISION = "d72ab3e281eca79247dbeee237dcb5a74cfa3f9b"
JBB_DATA_URL = (
    "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/"
    f"{JBB_DATA_REVISION}/data/{{kind}}-behaviors.csv"
)


def fetch_goals(kind: str) -> list[str]:
    request = urllib.request.Request(
        JBB_DATA_URL.format(kind=kind),
        headers={"User-Agent": "cs453-jailbreak-harness/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    goals = [" ".join(str(row["Goal"]).split()) for row in rows]
    if len(goals) != 100 or len(set(goals)) != 100:
        raise RuntimeError(
            f"Expected 100 unique JBB {kind} goals at the pinned revision; got {len(goals)}"
        )
    return goals


def _write_goals(output_path: Path, goals: list[str], force: bool) -> Path:
    content = "\n".join(goals) + "\n"
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        if existing == content:
            print(f"already current: {output_path}")
            return output_path
        existing_goals = [
            " ".join(line.split()) for line in existing.splitlines() if line.strip()
        ]
        if existing_goals == goals:
            print(f"already current (whitespace differs only): {output_path}")
            return output_path
        if not force:
            raise FileExistsError(
                f"{output_path} differs from pinned JBB data; pass --force to replace it"
            )
    output_path.write_text(content, encoding="utf-8")
    print(f"saved {len(goals)} goals: {output_path}")
    return output_path


def write_batch(kind: str, force: bool = False) -> Path:
    goals = fetch_goals(kind)
    full_path = _write_goals(
        PROMPTS_DIR / f"jailbreakbench_{kind}_100.txt",
        goals,
        force,
    )
    _write_goals(
        PROMPTS_DIR / f"jailbreakbench_{kind}_10.txt",
        goals[:10],
        force,
    )
    return full_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=("harmful", "benign", "both"),
        default="both",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kinds = ("harmful", "benign") if args.kind == "both" else (args.kind,)
    try:
        for kind in kinds:
            write_batch(kind, force=args.force)
    except Exception as exc:
        raise SystemExit(f"JailbreakBench dataset fetch failed: {exc}") from None


if __name__ == "__main__":
    main()
