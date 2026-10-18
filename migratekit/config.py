"""Which model each agent role uses. Override any of them in .env,
for example MIGRATEKIT_MODEL_WORKER=openai/gpt-oss-20b"""
import os

DEFAULT_MODELS = {
    "planner": "llama-3.3-70b-versatile",
    "worker": "openai/gpt-oss-120b",
    "integrator": "openai/gpt-oss-20b",
    "light": "llama-3.1-8b-instant",
}


def model_for(role: str) -> str:
    return os.getenv(f"MIGRATEKIT_MODEL_{role.upper()}", DEFAULT_MODELS[role])