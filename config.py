"""Central config. Override any of these with CLI flags in main.py."""

MODEL = "llama3.2:3b"   # any pulled Ollama model tag

ATTACK = "none"      # attacks:  sample_hi_adam | none
DEFENSE = "none"  # comma-separated defenses
JUDGE = "jailbreak_bench_llama8b"   # judges:  sample_safe_unsafe | jailbreak_bench_llama8b
# If using jailbreak_bench_llama8b, you must pull the model first: `ollama pull llama3:8b`
JUDGE_MODEL = "llama3:8b"   # model tag used by jailbreak_bench_llama8b
BATCH = "jailbreakbench_harmful_100"       # prompt batch = a .txt file stem in prompts/

LANGFUSE_PUBLIC_KEY = "pk-lf-8b007852-46b2-435a-b84d-b90b592aa3c8"
LANGFUSE_SECRET_KEY = "sk-lf-9c526a34-f860-41e6-81c8-9497005d2d1b"
LANGFUSE_BASE_URL = "http://localhost:3000"
