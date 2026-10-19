"""The agent loop: call model, run tools, feed results back, repeat."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from migratekit.tracing import LLMCallEvent, ToolCallEvent, Tracer

from .llm import LLMClient, LLMResponse
from .tools import ToolRegistry

FINAL_NOTE = (
    "\n\nThis is your final step. Do not call any tools. "
    "Give your best final answer now, using what you have learned."
)


@dataclass
class RunResult:
    final_text: str
    steps: int
    stop_reason: str            # "done" | "max_steps" | "token_budget"
    total_input_tokens: int
    total_output_tokens: int
    messages: list[dict[str, Any]]


def _hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]

def _flatten(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn the whole conversation into one plain-text message.

    Used on the final step so the model sees no tool-call structure to imitate.
    """
    lines: list[str] = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            lines.append(f"[{m['role']}] {content}")
            continue
        for b in content:
            if b["type"] == "text":
                lines.append(f"[assistant] {b['text']}")
            elif b["type"] == "tool_use":
                lines.append(f"[tool call] {b['name']}({json.dumps(b['input'])})")
            elif b["type"] == "tool_result":
                lines.append(f"[tool result] {b['content']}")
    transcript = "\n".join(lines)
    return [{
        "role": "user",
        "content": (
            "Here is the work done so far:\n\n" + transcript +
            "\n\nNow give your final answer in plain text. Do not call any tools."
        ),
    }]


def _is_retryable(e: Exception) -> bool:
    """Retry rate limits, server errors and network problems. Not 401/404."""
    status = getattr(e, "status_code", None)
    if status is None:
        return True
    return status == 429 or status >= 500


class AgentLoop:
    def __init__(
        self,
        client: LLMClient,
        tools: ToolRegistry,
        tracer: Tracer,
        run_id: str,
        agent_id: str,
        system: str,
        parent_agent_id: str | None = None,
        max_steps: int = 20,
        max_total_tokens: int = 200_000,
        checkpoint_path: str | Path | None = None,
        max_llm_retries: int = 3,
        retry_base_delay_s: float = 2.0,
    ):
        self.client = client
        self.tools = tools
        self.tracer = tracer
        self.run_id = run_id
        self.agent_id = agent_id
        self.system = system
        self.parent_agent_id = parent_agent_id
        self.max_steps = max_steps
        self.max_total_tokens = max_total_tokens
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.max_llm_retries = max_llm_retries
        self.retry_base_delay_s = retry_base_delay_s
        self._prev_prompt_tokens: int | None = None

    # ---- checkpoints -------------------------------------------------
    def _save(self, messages: list[dict[str, Any]]) -> None:
        if not self.checkpoint_path:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(messages), encoding="utf-8")
        os.replace(tmp, self.checkpoint_path)  # atomic: never a half-written file

    def _load(self) -> list[dict[str, Any]] | None:
        if self.checkpoint_path and self.checkpoint_path.exists():
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return None

    # ---- one model call, with retries and tracing --------------------
    def _call_llm(
        self, messages: list[dict[str, Any]], step: int, final: bool = False
    ) -> LLMResponse:
        # On the final step: no tools, plain-text history, explicit instruction.
        system = self.system + FINAL_NOTE if final else self.system
        tool_specs = [] if final else self.tools.specs()
        send = _flatten(messages) if final else messages

        last_err: Exception | None = None
        for attempt in range(self.max_llm_retries):
            start = time.time()
            try:
                resp = self.client.complete(system, send, tool_specs)
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.tracer.emit(LLMCallEvent(
                    run_id=self.run_id, agent_id=self.agent_id,
                    parent_agent_id=self.parent_agent_id, step=step,
                    start_ts=start, end_ts=time.time(), success=False, error=str(e),
                    model=self.client.model, backend=self.client.backend,
                    prompt_tokens=0, completion_tokens=0,
                ))
                print(f"[{self.agent_id}] step {step}: LLM call failed "
                      f"({type(e).__name__}: {str(e)[:150]})", flush=True)
                if not _is_retryable(e):
                    raise
                time.sleep(self.retry_base_delay_s * (2 ** attempt))
                continue

            # In an append-only loop, the whole previous prompt is the start of
            # this prompt, so those tokens are cache-reusable in principle.
            # (The flattened final step breaks that, so we record it as unshared.)
            self.tracer.emit(LLMCallEvent(
                run_id=self.run_id, agent_id=self.agent_id,
                parent_agent_id=self.parent_agent_id, step=step,
                start_ts=start, end_ts=time.time(),
                model=self.client.model, backend=self.client.backend,
                prompt_tokens=resp.input_tokens,
                completion_tokens=resp.output_tokens,
                shared_prefix_tokens=None if final else self._prev_prompt_tokens,
                ttft_s=resp.ttft_s,
                segment_hashes={
                    "system": _hash(system),
                    "tools": _hash(tool_specs),
                    "history": _hash(send),
                },
            ))
            self._prev_prompt_tokens = resp.input_tokens
            return resp
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    # ---- the loop ----------------------------------------------------
    def run(self, task: str) -> RunResult:
        messages = self._load() or [{"role": "user", "content": task}]
        # On resume, count earlier assistant turns so step numbers continue.
        step = sum(1 for m in messages if m["role"] == "assistant")
        total_in = total_out = 0
        final_text = ""

        while True:
            if step >= self.max_steps:
                return RunResult(final_text, step, "max_steps", total_in, total_out, messages)
            if total_in + total_out >= self.max_total_tokens:
                return RunResult(final_text, step, "token_budget", total_in, total_out, messages)

            resp = self._call_llm(messages, step, final=(step == self.max_steps - 1))
            total_in += resp.input_tokens
            total_out += resp.output_tokens
            if not resp.tool_calls and not resp.text.strip():
                # Empty reply (seen with some open models): ask once more,
                # with tools hidden and a plain-text request.
                resp = self._call_llm(messages, step, final=True)
                total_in += resp.input_tokens
                total_out += resp.output_tokens
            step += 1
            final_text = resp.text
            messages.append({"role": "assistant", "content": resp.content})

            if not resp.tool_calls:
                self._save(messages)
                return RunResult(final_text, step, "done", total_in, total_out, messages)

            results = []
            for call in resp.tool_calls:
                t0 = time.time()
                output, ok = self.tools.execute(call.name, call.input)
                self.tracer.emit(ToolCallEvent(
                    run_id=self.run_id, agent_id=self.agent_id,
                    parent_agent_id=self.parent_agent_id, step=step,
                    start_ts=t0, end_ts=time.time(), success=ok,
                    error=None if ok else output,
                    tool=call.name,
                    arg_bytes=len(json.dumps(call.input).encode()),
                    result_bytes=len(output.encode()),
                ))
                results.append({
                    "type": "tool_result", "tool_use_id": call.id,
                    "content": output, "is_error": not ok,
                })
            messages.append({"role": "user", "content": results})
            self._save(messages)