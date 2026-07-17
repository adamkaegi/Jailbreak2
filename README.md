# Jailbreak — minimal attack/defense chain

A tiny LangChain harness that runs a prompt through a swappable
**attack**, a swappable **defense**, a swappable **judge**, and a local Ollama model.

## Flow

```
original prompt -> attack -> [input defenses] -> model -> [output defenses] -> response
       |                                                                  |
       +---------------------------- judge -------------------------------+
```

Each defense declares a `stage` (`"input"` or `"output"`), and the chain plugs
every defense in at the right point automatically. The judge runs after the
chain and leaves the response text unchanged. Pairwise judges receive the
original behavior plus the final response; response-only demo judges remain
compatible. Multiple defenses are applied in the order you list them, grouped
by stage.

## Layout

```
attacks/     one file per attack + context-only base.py + registry in __init__.py
defenses/    one file per defense + context-only base.py + registry in __init__.py
judges/      one file per judge + context-only base.py + runtime compatibility
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

- **Attack** `deepinception` — Ryan's five-layer nested-scene template attack
- **Attack** `template` — compatibility alias for `deepinception`
- **Attack** `none` — raw-prompt baseline
- **Attack** `sample_hi_adam` — original wiring example

## Defences

- **Defense** `smoothllm` (input) — generates multiple perturbed prompt variants, scores them, and selects one from the majority class
- **Defense** `self_reminder` (input) — wraps the prompt with a responsible-assistant prefix and a safety-review suffix; both are overridable on the defense constructor

## Judges

- **Judge** `sample_safe_unsafe` — keyword-only wiring/debug judge
- **Judge** `strongreject` — local fine-tuned StrongREJECT evaluator; produces
  a continuous harmful-assistance score from 0 (refusal/useless) to 1
  (maximally useful harmful assistance)
- **Judge** `harmbench_mistral_7b_val_cls` — official binary HarmBench Mistral
  validation classifier (`Unsafe`/1 for Yes, `Safe`/0 for No)

The Mistral checkpoint is explicitly HarmBench's validation classifier. It is
available as this harness's post-generation judge as requested, but HarmBench's
canonical test/final classifier is the separate Llama-2 13B classifier.

## Prompt batches

- `general` — everyday questions
- `instructions` — instruction-following prompts

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-cuda.txt
ollama serve # separate terminal
python scripts\pull_models.py
python scripts\prefetch_strongreject.py
python scripts\prefetch_harmbench.py
```

The two evaluator prefetch commands download large Hugging Face checkpoints;
HarmBench is approximately 14.5 GB. StrongREJECT also requires accepting the
`google/gemma-2b` license and setting `HF_TOKEN` in `.env`.

## Local Langfuse

If you want to trace runs locally, start a Langfuse instance in a separate terminal or repo first:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Then create a Langfuse project in the UI at http://localhost:3000 and put its
keys in a local `.env` copied from `.env.example`:

```bash
LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_BASE_URL = "http://localhost:3000"
```


## Flags

The CLI accepts these flags:

- `--model` sets the Ollama model tag to use. Default: `qwen2.5:7b-instruct`.
- `--attack` selects the attack implementation from `attacks/__init__.py`. Default: `deepinception`.
- `--defense` selects one or more defenses from `defenses/__init__.py` as a comma-separated list. Default: `none`.
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
python main.py --judge sample_safe_unsafe             # lightweight debug judge
python main.py --judge strongreject                   # continuous harmful score
python main.py --judge harmbench_mistral_7b_val_cls   # binary validation classifier
python main.py --dry-run                              # no Ollama, tests wiring
```

## Adding real components later

- New attack: add a file in `attacks/`, subclass `Attack`, register it in `attacks/__init__.py`.
- New defense: add a file in `defenses/`, subclass `Defense`, set `stage`, register it in `defenses/__init__.py`.
- New judge: add a file in `judges/`, subclass `Judge`, register it in `judges/__init__.py`.
- New batch: drop a `.txt` file in `prompts/` — it's auto-discovered by its filename.

Nothing in `pipeline.py` or `main.py` changes when you do.

## Scripts

- `scripts/pull_models.py` — pulls every configured Ollama target/support model.
- `scripts/prefetch_strongreject.py` — downloads and smoke-tests the pinned
  StrongREJECT adapter and Gemma base.
- `scripts/prefetch_harmbench.py` — downloads and smoke-tests the pinned
  HarmBench Mistral validation classifier.
- `scripts/judge_csv.py` — applies a judge to cached responses without
  regenerating target-model output; it always writes a new CSV.
- `scripts/run_readme_examples.sh` — runs the lightweight README examples.

## Tests

Run the complete suite with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use fake evaluators/models for heavyweight paths. They check attack
formatting, registry/default selection, pairwise judge wiring, score parsing,
model revision pins, deterministic target settings, one-time attack execution,
CSV checkpoint/re-judging behavior, and Ollama VRAM unloading. Separate
prefetch scripts provide the real checkpoint smoke tests.

## ToDo

- Figure out the prompts to use
