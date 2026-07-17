# Jailbreak — minimal attack/defense chain

A tiny LangChain harness that runs a prompt through a swappable
**attack**, a swappable **defense**, a swappable **judge**, and a local Ollama model.

## Flow

```
prompt -> attack -> [input-stage defenses] -> model -> [output-stage defenses] -> judge
```

Each defense declares a `stage` (`"input"` or `"output"`), and the chain plugs
every defense in at the right point automatically. The judge runs after the
chain, labels the model output as `Safe` or `Unsafe`, and leaves the response
text unchanged. Multiple defenses are applied in the order you list them,
grouped by stage.

## Layout

```
attacks/     one file per attack + base.py + registry in __init__.py
defenses/    one file per defense + base.py + registry in __init__.py
prompts/     one .txt per batch (one prompt per line) + loader
config.py    model / attack / defense / batch defaults
pipeline.py  builds the LCEL chain
main.py      CLI entry point
outputs/     CSV files written per run
scripts/     shell scripts that run main.py multiple times consecutively
```

## Sample components

- **Attack** `sample_hi_adam` — appends `Start all messages with "Hi Adam".`
- **Defense** `sample_bye_adam_input` (input) — appends `End all messages with "Bye Adam".` to the prompt
- **Defense** `sample_bye_adam_output` (output) — appends `Bye Adam` to the model's response
- **Judge** `sample_safe_unsafe` — labels the model output as `Safe` or `Unsafe` with a simple keyword check

## Attacks

- **Attack** `gcg` — appends a precomputed, static GCG-style universal adversarial suffix to the prompt
- **Attack** `pair` — iteratively refines the prompt in a black-box loop: sends the current prompt to the target model, judges the response with `sample_safe_unsafe`, and if still `Safe`, asks an attacker LLM (`qwen2.5:3b` by default) to mutate the prompt; repeats up to `max_rounds` times (default 3)


## Defences

- **Defense** `smoothllm` (input) — generates multiple perturbed prompt variants, scores them, and selects one from the majority class
- **Defense** `self_reminder` (input) — wraps the prompt with a responsible-assistant prefix and a safety-review suffix; both are overridable on the defense constructor
- **Defense** `perplexity` (input) — scores the attacked prompt with local GPT-2 and blocks scores above the configured threshold
- **Defense** `llama_guard_input` (input) — classifies the model-facing prompt with Llama Guard and skips target inference when unsafe
- **Defense** `llama_guard_output` (output) — classifies and replaces unsafe model output

## Judges

- **Judge** `sample` — sample

## Prompt batches

- `general` — everyday questions
- `instructions` — instruction-following prompts

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama serve # separate terminal
# pull models as needed
ollama pull llama3.2:3b # whatever model configured
ollama pull llama-guard3:1b # if running defense and need
ollama pull llama3.2:3b # whatever model configured (prompt response and judge)
```

## Local Langfuse

If you want to trace runs locally, start a Langfuse instance in a separate terminal or repo first:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Then create a Langfuse project in the UI at http://localhost:3000 and the keys to `config.py`:

```bash
LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_BASE_URL = "http://localhost:3000"
```


## Flags

The CLI accepts these flags:

- `--model` sets the Ollama model tag to use. Default: `llama3.2:3b`.
- `--attack` selects the attack implementation from `attacks/__init__.py`. Default: `sample_hi_adam`.
- `--defense` selects one or more defenses from `defenses/__init__.py` as a comma-separated list. Default: `sample_bye_adam_input,sample_bye_adam_output`.
- `--judge` selects the judge implementation from `judges/__init__.py`. Default: `sample_safe_unsafe`.
- `--batch` chooses the prompt batch file stem from `prompts/`. Default: `general`.
- `--dry-run` skips Ollama and echoes the prompt flow for wiring checks.

A `prompt` is optional. If you omit it, the selected batch runs instead.

Defaults live in `config.py`; every one is overridable with the flags above.

## Run

```bash
python main.py "What is the capital of France?"       # single prompt
python main.py                                        # batch from config.py
python main.py --batch instructions --defense sample_bye_adam_input,sample_bye_adam_output
python main.py --judge sample_safe_unsafe              # judge final model output
python main.py --dry-run                              # stub target, test wiring
```

## Adding real components later

- New attack: add a file in `attacks/`, subclass `Attack`, register it in `attacks/__init__.py`.
- New defense: add a file in `defenses/`, subclass `Defense`, set `stage`, register it in `defenses/__init__.py`.
- New judge: add a file in `judges/`, subclass `Judge`, register it in `judges/__init__.py`.
- New batch: drop a `.txt` file in `prompts/` — it's auto-discovered by its filename.

Nothing in `pipeline.py` or `main.py` changes when you do.

## Scripts

Add as needed for testing
- `scripts/run_readme_examples.sh` — runs every example from the [Run](#run) section consecutively.
- `scripts/script.sh` - runs all attacks/defenses as of Jul 16

## ToDo

- Run and see issues
