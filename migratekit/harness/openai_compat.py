"""OpenAI-compatible client. Works with Groq now and vLLM/SGLang later.

Our loop stores messages in Anthropic-style blocks. This file translates
to and from the OpenAI chat format at the boundary, so the loop is unchanged.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .llm import LLMResponse, ToolCall


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "assistant":
            text = "\n".join(b["text"] for b in content if b["type"] == "text")
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                }
                for b in content
                if b["type"] == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user turn holding tool results
            for b in content:
                if b["type"] == "tool_result":
                    out.append(
                        {"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]}
                    )
    return out


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class OpenAICompatClient:
    def __init__(self, model: str, base_url: str, api_key: str, backend: str = "api"):
        from openai import OpenAI

        self.model = model
        self.backend = backend
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, system, messages, tools, max_tokens=2048) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=to_openai_messages(system, messages),
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        resp = self._client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        msg = choice.message
        content: list[dict[str, Any]] = []
        calls: list[ToolCall] = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # Malformed arguments: pass them on so the tool errors and the
                # model sees the failure and can correct itself.
                args = {"_raw_arguments": tc.function.arguments}
            content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
            calls.append(ToolCall(tc.id, tc.function.name, args))

        usage = resp.usage
        return LLMResponse(
            text=msg.content or "",
            tool_calls=calls,
            content=content,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=choice.finish_reason or "",
        )


def GroqClient(model: str = "openai/gpt-oss-120b") -> OpenAICompatClient:
    return OpenAICompatClient(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        backend="groq",
    )