#!/usr/bin/env bash
# Small smoke matrix matching the current CLI and component names.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py --dry-run
python main.py --dry-run --defense smoothllm
python main.py "What is the capital of France?" --attack none --defense none
python main.py --batch instructions --model qwen2.5:7b-instruct --attack none --defense none
