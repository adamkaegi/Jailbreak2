#!/usr/bin/env bash
# Runs every example command from the README's "Run" section, one after another.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py "What is the capital of France?"
python main.py
python main.py --batch instructions --defense sample_bye_adam_input,sample_bye_adam_output
python main.py --judge sample_safe_unsafe
python main.py --dry-run
