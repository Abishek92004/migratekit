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
    def _call_llm(self, messages: list[dict[str, Any]], step: int) -> LLMResponse:
        last_err: Exception | None = None
        for attempt in range(self.max_llm_retries):
            start = time.time()
            try:
                resp = self.client.complete(self.system, messages, self.tools.specs())
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.tracer.emit(LLMCallEvent(
                    run_id=self.run_id, agent_id=self.agent_id,
                    parent_agent_id=self.parent_agent_id, step=step,
                    start_ts=start, end_ts=time.time(), success=False, error=str(e),
                    model=self.client.model, backend=self.client.backend,
                    prompt_tokens=0, completion_tokens=0,
                ))
                time.sleep(self.retry_base_delay_s * (2 ** attempt))
                continue

            # In an append-only loop, the whole previous prompt is the start of
            # this prompt, so those tokens are cache-reusable in principle.
            self.tracer.emit(LLMCallEvent(
                run_id=self.run_id, agent_id=self.agent_id,
                parent_agent_id=self.parent_agent_id, step=step,
                start_ts=start, end_ts=time.time(),
                model=self.client.model, backend=self.client.backend,
                prompt_tokens=resp.input_tokens,
                completion_tokens=resp.output_tokens,
                shared_prefix_tokens=self._prev_prompt_tokens,
                ttft_s=resp.ttft_s,
                segment_hashes={
                    "system": _hash(self.system),
                    "tools": _hash(self.tools.specs()),
                    "history": _hash(messages),
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

            resp = self._call_llm(messages, step)
            step += 1
            total_in += resp.input_tokens
            total_out += resp.output_tokens
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