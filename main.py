"""Run a prompt (or a whole batch) through attack -> defense -> model -> defense -> judge.

Examples:
    python main.py "What is the capital of France?"
    python main.py                       # runs the batch set in config.py
    python main.py --batch instructions --defense sample_bye_adam_input,sample_bye_adam_output
    python main.py --dry-run             # no Ollama needed
"""

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import config
from attacks import ATTACKS
from defenses import DEFENSES
from defenses.block import DefenseBlocked
from judges import JUDGES
from judges.runtime import (
    JudgeResult,
    evaluate_judge_batch,
    judge_requires_target_unload,
)
from pipeline import build_response_chain
from prompts import available_batches, load_batch


def _parse_defense_names(raw_value: str) -> list[str]:
    """argparse `type=` callback for --defense: "a,b" -> ["a", "b"], validated.

    Raising ArgumentTypeError here makes argparse print a clean usage error
    and exit, the same as an unknown --attack or --judge choice would.
    """
    names = [name.strip() for name in raw_value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one defense name is required")
    unknown = [name for name in names if name not in DEFENSES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown defense(s): {', '.join(unknown)}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="A single prompt. Omit to run a batch.")
    parser.add_argument(
        "--model",
        default=config.MODEL,
        help=f"Ollama target model (configured: {', '.join(config.TARGET_MODELS)}).",
    )
    parser.add_argument("--attack", default=config.ATTACK, choices=ATTACKS.keys())
    parser.add_argument(
        "--defense",
        default=_parse_defense_names(config.DEFENSE),
        type=_parse_defense_names,
        help="Comma-separated defense names.",
    )
    parser.add_argument("--judge", default=config.JUDGE, choices=JUDGES.keys())
    parser.add_argument("--batch", default=config.BATCH, choices=available_batches())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Echo the target prompt; selected defenses still run.",
    )
    return parser.parse_args()


def _load_langfuse_handler() -> tuple[object | None, object | None]:
    """Return (client, callback_handler) for tracing chain runs in Langfuse.

    Langfuse is optional observability, not part of the pipeline itself, so
    if it's unconfigured or not installed we just skip tracing.
    """
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        return None, None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return None, None

    client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        base_url=config.LANGFUSE_BASE_URL,
    )
    return client, CallbackHandler()


def _write_run_csv(
    run_rows: list[dict[str, str]],
    attack_name: str,
    defense_names: list[str],
    judge_name: str,
    model_name: str,
    batch_name: str | None,
    csv_path: Path | None = None,
) -> Path:
    if csv_path is None:
        outputs_dir = Path(__file__).resolve().parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = uuid4().hex[:8]
        csv_path = outputs_dir / f"run-{batch_name or 'single'}-{timestamp}-{run_id}.csv"

    fieldnames = [
        "attack",
        "defenses",
        "judge",
        "model",
        "input",
        "attacked_input",
        "output",
        "judge_label",
        "judge_score",
        "attack_latency_seconds",
        "response_pipeline_latency_seconds",
        "latency_seconds",
        "judge_batch_latency_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(
                {
                    "attack": attack_name,
                    "defenses": ",".join(defense_names),
                    "judge": judge_name,
                    "model": model_name,
                    "input": row["input"],
                    "attacked_input": row["attacked_input"],
                    "output": row["output"],
                    "judge_label": row["judge_label"],
                    "judge_score": row["judge_score"],
                    "attack_latency_seconds": row["attack_latency_seconds"],
                    "response_pipeline_latency_seconds": row[
                        "response_pipeline_latency_seconds"
                    ],
                    "latency_seconds": row["latency_seconds"],
                    "judge_batch_latency_seconds": row["judge_batch_latency_seconds"],
                }
            )

    return csv_path


def _append_run_csv_row(
    csv_path: Path,
    row: dict[str, str],
    attack_name: str,
    defense_names: list[str],
    judge_name: str,
    model_name: str,
) -> None:
    """Append one completed target response to the pre-judge checkpoint."""
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "attack",
                "defenses",
                "judge",
                "model",
                "input",
                "attacked_input",
                "output",
                "judge_label",
                "judge_score",
                "attack_latency_seconds",
                "response_pipeline_latency_seconds",
                "latency_seconds",
                "judge_batch_latency_seconds",
            ],
        )
        writer.writerow(
            {
                "attack": attack_name,
                "defenses": ",".join(defense_names),
                "judge": judge_name,
                "model": model_name,
                **row,
            }
        )


def _unload_ollama_model(model_name: str) -> None:
    """Release target-model VRAM before a local GPU-backed judge loads."""
    result = subprocess.run(
        ["ollama", "stop", model_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        print(f"warning: could not unload Ollama model '{model_name}': {detail}")


def _configure_defense_for_run(defense: object, model_name: str, dry_run: bool) -> object:
    """Use an optional concrete run hook without expanding defenses/base.py."""
    for_run = getattr(defense, "for_run", None)
    if callable(for_run):
        return for_run(model_name, dry_run=dry_run)
    return defense


def _invoke_chain(chain, prompt: str, invoke_kwargs: dict) -> str:
    """Invoke the chain, returning a defense's response after an input block."""
    try:
        return chain.invoke(prompt, **invoke_kwargs)
    except DefenseBlocked as blocked:
        return blocked.response


def main() -> None:
    args = parse_args()

    attack = ATTACKS[args.attack]
    defenses = [
        _configure_defense_for_run(DEFENSES[name], args.model, args.dry_run)
        for name in args.defense
    ]
    judge = JUDGES[args.judge]

    # Apply the attack once in the loop so its exact output can be cached.
    # This chain handles the remaining defenses -> model -> defenses stages.
    response_chain = build_response_chain(defenses, args.model, dry_run=args.dry_run)
    langfuse_client, langfuse_handler = _load_langfuse_handler()

    single_prompt = args.prompt is not None
    prompts = [args.prompt] if single_prompt else load_batch(args.batch)
    source = "single prompt" if single_prompt else f"batch '{args.batch}' ({len(prompts)} prompts)"

    defense_summary = ", ".join(f"{defense.name}:{defense.stage}" for defense in defenses)
    print(f"attack={attack.name}  defenses=[{defense_summary}]  judge={judge.name}  model={args.model}")
    print(f"running: {source}\n")

    run_rows: list[dict[str, str]] = []
    csv_path = _write_run_csv(
        run_rows,
        attack.name,
        args.defense,
        judge.name,
        args.model,
        batch_name=None if single_prompt else args.batch,
    )
    for i, prompt in enumerate(prompts, 1):
        print(f"--- [{i}] {prompt}")
        invoke_kwargs = (
            {
                "config": {
                    "callbacks": [langfuse_handler],
                    "run_name": f"[{i}] {prompt[:80]}",
                }
            }
            if langfuse_handler
            else {}
        )
        attack_start_time = time.perf_counter()
        attacked_prompt = attack.apply(prompt)
        attack_latency_seconds = time.perf_counter() - attack_start_time

        response_start_time = time.perf_counter()
        output_text = _invoke_chain(response_chain, attacked_prompt, invoke_kwargs)
        response_latency_seconds = time.perf_counter() - response_start_time
        latency_seconds = attack_latency_seconds + response_latency_seconds
        row = {
            "input": prompt,
            "attacked_input": attacked_prompt,
            "output": output_text,
            "judge_label": "",
            "judge_score": "",
            "attack_latency_seconds": f"{attack_latency_seconds:.3f}",
            "response_pipeline_latency_seconds": f"{response_latency_seconds:.3f}",
            "latency_seconds": f"{latency_seconds:.3f}",
            "judge_batch_latency_seconds": "",
        }
        run_rows.append(row)
        _append_run_csv_row(
            csv_path,
            row,
            attack.name,
            args.defense,
            judge.name,
            args.model,
        )
        print(output_text)
        print(
            f"attack_latency={attack_latency_seconds:.3f}s  "
            f"response_latency={response_latency_seconds:.3f}s  "
            f"total_latency={latency_seconds:.3f}s\n"
        )

    # Cache target-model outputs before invoking a potentially expensive judge.
    # If judge setup fails (for example, gated weights are unavailable), the
    # generated responses are still available for a later offline judge pass.
    if args.dry_run:
        judge_results = [JudgeResult(label="Skipped (dry-run)") for _ in run_rows]
        judge_latency_seconds = 0.0
    else:
        if judge_requires_target_unload(judge):
            print(f"unloading Ollama target {args.model} before local judge...")
            _unload_ollama_model(args.model)
        print(f"judging {len(run_rows)} response(s) with {judge.name}...")
        judge_start_time = time.perf_counter()
        try:
            judge_results = evaluate_judge_batch(
                judge,
                [row["input"] for row in run_rows],
                [row["output"] for row in run_rows],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Judge '{judge.name}' failed; responses remain cached at {csv_path}"
            ) from exc
        judge_latency_seconds = time.perf_counter() - judge_start_time

    if len(judge_results) != len(run_rows):
        raise RuntimeError(
            f"Judge '{judge.name}' returned {len(judge_results)} result(s) for "
            f"{len(run_rows)} response(s); responses remain cached at {csv_path}"
        )

    for index, (row, result) in enumerate(zip(run_rows, judge_results), 1):
        row["judge_label"] = result.label or ""
        row["judge_score"] = "" if result.score is None else f"{result.score:.6f}"
        row["judge_batch_latency_seconds"] = f"{judge_latency_seconds:.3f}"
        print(f"judge [{index}] {result.display()}")

    _write_run_csv(
        run_rows,
        attack.name,
        args.defense,
        judge.name,
        args.model,
        batch_name=None if single_prompt else args.batch,
        csv_path=csv_path,
    )
    print(f"judge batch latency: {judge_latency_seconds:.3f}s")
    print(f"saved csv: {csv_path}")

    if langfuse_client is not None:
        langfuse_client.flush()


if __name__ == "__main__":
    main()
