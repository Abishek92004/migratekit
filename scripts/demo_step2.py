"""Step 2 demo: a tool-using agent on Groq, with full tracing.

Run from the project root:
    python -m scripts.demo_step2
"""
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from migratekit.config import model_for
from migratekit.harness import AgentLoop, GroqClient, Tool, ToolRegistry
from migratekit.tracing import Tracer, read_events

SKIP = {".venv", ".git", "__pycache__", "traces", ".pytest_cache"}


def list_dir(path: str = ".") -> str:
    """List files and folders, skipping noisy ones to save tokens."""
    names = []
    for p in sorted(Path(path).iterdir()):
        if p.name in SKIP:
            continue
        names.append(p.name + ("/" if p.is_dir() else ""))
    return "\n".join(names) or "(empty)"


def read_file(path: str) -> str:
    """Read a text file, truncated to keep the prompt small."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if len(text) > 2000:
        return text[:2000] + "\n...[truncated]"
    return text


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not found. Add it to the .env file in the project root.")

    tools = ToolRegistry([
        Tool(
            "list_dir",
            "List files and folders in a directory.",
            {"type": "object", "properties": {"path": {"type": "string"}}},
            list_dir,
        ),
        Tool(
            "read_file",
            "Read a text file.",
            {"type": "object", "properties": {"path": {"type": "string"}},
             "required": ["path"]},
            read_file,
        ),
    ])

    run_id = uuid.uuid4().hex[:8]
    trace_path = Path("traces") / f"{run_id}.jsonl"

    loop = AgentLoop(
        client=GroqClient(model_for("worker")),
        tools=tools,
        tracer=Tracer(trace_path),
        run_id=run_id,
        agent_id="demo_agent",
        system=(
            "You are a helpful assistant that explores a Python project using tools. "
            "Be efficient: use at most 4 tool calls, then answer."
        ),
        max_steps=8,
        max_llm_retries=6,
        retry_base_delay_s=10,
        checkpoint_path=Path("traces") / f"{run_id}.ckpt.json",
    )

    result = loop.run(
        "Look at this project and tell me, in 3 sentences, what it is for."
    )

    print("\n=== ANSWER ===")
    print(result.final_text)
    print(
        f"\nsteps={result.steps}  stop={result.stop_reason}  "
        f"tokens_in={result.total_input_tokens}  tokens_out={result.total_output_tokens}"
    )

    print("\n=== TRACE ===")
    print(f"{'type':10} {'what':28} {'secs':>6}  ok")
    for e in read_events(trace_path):
        what = getattr(e, "tool", None) or e.model
        print(f"{e.type:10} {what:28} {e.duration_s:6.2f}  {e.success}")
    print(f"\nTrace saved to {trace_path}")


if __name__ == "__main__":
    main()