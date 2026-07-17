"""Central config. Override any of these with CLI flags in main.py."""

import os

try:
    from dotenv import load_dotenv
except ImportError:  # Lets lightweight registry tests run before setup.
    pass
else:
    load_dotenv()

# Target models from the proposal and midterm research.  The first entry is
# the canonical model for the final matrix; the others remain available for
# smoke tests and sensitivity/stress-test runs.
TARGET_MODELS = (
    "qwen2.5:7b-instruct",
    "dolphin-mistral:7b",
    "qwen2.5:3b",
    "llama3.2:3b",
)
MODEL = TARGET_MODELS[0]

# Models used by a defense or evaluator rather than as experiment targets.
SUPPORT_MODELS = ("llama-guard3:1b",)
STRONGREJECT_MODEL = "qylu4156/strongreject-15k-v1"
STRONGREJECT_MODEL_REVISION = "4bd893d32390d2cace4f067dc2e3ef5294fd78a2"
STRONGREJECT_BASE_MODEL = "google/gemma-2b"
STRONGREJECT_BASE_REVISION = "2ac59a5d7bf4e1425010f0d457dde7d146658953"
STRONGREJECT_BATCH_SIZE = 1
STRONGREJECT_MAX_RESPONSE_LENGTH = 512
STRONGREJECT_ALLOW_CPU = os.getenv("STRONGREJECT_ALLOW_CPU", "").lower() in {
    "1",
    "true",
    "yes",
}
HARMBENCH_MODEL = "cais/HarmBench-Mistral-7b-val-cls"
HARMBENCH_MODEL_REVISION = "51182c7cdaf9b2e5f05b745b81b047b67e7384a1"
HARMBENCH_BATCH_SIZE = 1
HARMBENCH_MAX_RESPONSE_LENGTH = 512
HARMBENCH_ALLOW_CPU = os.getenv("HARMBENCH_ALLOW_CPU", "").lower() in {
    "1",
    "true",
    "yes",
}
MODEL_TEMPERATURE = 0
MODEL_SEED = 42
MODEL_MAX_TOKENS = 512
CLASSIFIER_MAX_TOKENS = 8

# Component options used when the matching CLI flag is omitted:
#
# ATTACK selects exactly one prompt transformation:
#   deepinception - Ryan's published nested-role-play template attack.
#   template      - alias for deepinception, retained for older run commands.
#   none          - raw prompt baseline.
#   sample_hi_adam - Adam's original wiring/demo attack.
#
# DEFENSE accepts one name or a comma-separated sequence. Input defenses run
# before the target; output defenses run afterward. Within each stage, the
# listed order is preserved:
#   none, self_reminder, smoothllm,
#   sample_bye_adam_input, sample_bye_adam_output
#
# JUDGE selects one post-generation evaluator:
#   sample_safe_unsafe            - keyword-based wiring/debug judge.
#   strongreject                  - continuous 0..1 harmful-assistance score.
#   harmbench_mistral_7b_val_cls  - binary HarmBench validation classifier;
#                                    available here as a final pipeline stage,
#                                    but not HarmBench's canonical test metric.
ATTACK = "deepinception"
DEFENSE = "none"
JUDGE = "sample_safe_unsafe"
BATCH = "general"  # prompt batch = a .txt file stem in prompts/

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")
