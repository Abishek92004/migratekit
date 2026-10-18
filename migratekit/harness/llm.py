"""Model client interface. The loop only talks to this, so we can swap
API models for a self-hosted vLLM server later without touching the loop."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    content: list[dict[str, Any]]   # assistant blocks, appended to history as-is
    input_tokens: int
    output_tokens: int
    stop_reason: str = ""
    ttft_s: float | None = None


class LLMClient(Protocol):
    model: str
    backend: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> LLMResponse: ...


class AnthropicClient:
    backend = "api"

    def __init__(self, model: str | None = None):
        import anthropic

        self.model = model or os.getenv("MIGRATEKIT_MODEL", "claude-sonnet-5-5")
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, system, messages, tools, max_tokens=2048) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages
        )
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)

        content: list[dict[str, Any]] = []
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
                texts.append(block.text)
            elif block.type == "tool_use":
                content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
                calls.append(ToolCall(block.id, block.name, dict(block.input)))
        return LLMResponse(
            text="\n".join(texts),
            tool_calls=calls,
            content=content,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason or "",
        )


class FakeClient:
    """Returns pre-written responses in order. Used for tests, no API needed."""

    backend = "fake"
    model = "fake-model"

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls = 0

    def complete(self, system, messages, tools, max_tokens=2048) -> LLMResponse:
        resp = self.script[self.calls]
        self.calls += 1
        return resp