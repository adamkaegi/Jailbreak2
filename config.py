"""Central config. Override any of these with CLI flags in main.py."""

MODEL = "llama3.2:3b"   # any pulled Ollama model tag

ATTACK = "sample_hi_adam"      # attacks:  sample_hi_adam | none
DEFENSE = "sample_bye_adam_input,sample_bye_adam_output"  # comma-separated defenses
JUDGE = "sample_safe_unsafe"   # judges:  sample_safe_unsafe
BATCH = "general"       # prompt batch = a .txt file stem in prompts/

LANGFUSE_PUBLIC_KEY = "pk-lf-8b007852-46b2-435a-b84d-b90b592aa3c8"
LANGFUSE_SECRET_KEY = "sk-lf-9c526a34-f860-41e6-81c8-9497005d2d1b"
LANGFUSE_BASE_URL = "http://localhost:3000"
