"""Run a prompt (or a whole batch) through attack -> defense -> model -> defense -> judge.

Examples:
    python main.py "What is the capital of France?"
    python main.py                       # runs the batch set in config.py
    python main.py --batch instructions --defense sample_bye_adam_input,sample_bye_adam_output
    python main.py --dry-run             # no Ollama needed
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import config
from attacks import ATTACKS
from defenses import DEFENSES
from judges import JUDGES
from pipeline import build_chain
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
    parser.add_argument("--model", default=config.MODEL)
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
        "--dry-run", action="store_true", help="Echo the prompt instead of calling Ollama."
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
) -> Path:
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = outputs_dir / f"run-{batch_name or 'single'}-{timestamp}.csv"

    fieldnames = [
        "attack",
        "defenses",
        "judge",
        "model",
        "input",
        "output",
        "judge_label",
        "latency_seconds",
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
                    "output": row["output"],
                    "judge_label": row["judge_label"],
                    "latency_seconds": row["latency_seconds"],
                }
            )

    return csv_path


def main() -> None:
    args = parse_args()

    attack = ATTACKS[args.attack]
    defenses = [DEFENSES[name] for name in args.defense]
    judge = JUDGES[args.judge]

    # build_chain() wires attack -> input defenses -> model -> output defenses
    # into a single LangChain Runnable; see pipeline.py for how that works.
    chain = build_chain(attack, defenses, args.model, dry_run=args.dry_run)
    langfuse_client, langfuse_handler = _load_langfuse_handler()

    prompts = [args.prompt] if args.prompt else load_batch(args.batch)
    source = "single prompt" if args.prompt else f"batch '{args.batch}' ({len(prompts)} prompts)"

    defense_summary = ", ".join(f"{defense.name}:{defense.stage}" for defense in defenses)
    print(f"attack={attack.name}  defenses=[{defense_summary}]  judge={judge.name}  model={args.model}")
    print(f"running: {source}\n")

    run_rows: list[dict[str, str]] = []
    for i, prompt in enumerate(prompts, 1):
        print(f"--- [{i}] {prompt}")
        start_time = time.perf_counter()
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
        output_text = chain.invoke(prompt, **invoke_kwargs)
        latency_seconds = time.perf_counter() - start_time
        judge_label = judge.apply(output_text)
        if langfuse_client is not None and langfuse_handler is not None:
            langfuse_client.create_score(
                trace_id=langfuse_handler.last_trace_id,
                name="judge_label",
                value=judge_label,
                data_type="CATEGORICAL",
            )
        run_rows.append(
            {
                "input": prompt,
                "output": output_text,
                "judge_label": judge_label,
                "latency_seconds": f"{latency_seconds:.3f}",
            }
        )
        print(output_text)
        print(f"judge={judge_label}  latency={latency_seconds:.3f}s\n")

    csv_path = _write_run_csv(
        run_rows,
        attack.name,
        args.defense,
        judge.name,
        args.model,
        batch_name=args.batch if not args.prompt else None,
    )
    print(f"saved csv: {csv_path}")

    if langfuse_client is not None:
        langfuse_client.flush()


if __name__ == "__main__":
    main()
