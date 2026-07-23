#!/usr/bin/env bash
# Runs tests to check if working correctly.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py --batch prompts3 --defense smoothllm
python main.py --batch prompts3 --defense self_reminder
python main.py --batch prompts3 --defense perplexity
python main.py --batch prompts3 --defense llama_guard_input
python main.py --batch prompts3 --defense llama_guard_output
python main.py --batch prompts3 --attack gcg
python main.py --batch prompts3 --attack pair
