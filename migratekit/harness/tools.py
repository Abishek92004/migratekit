"""Tool definitions and a registry that runs them safely."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]       # JSON Schema describing the arguments
    fn: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | tuple[Tool, ...] = ()):
        self._tools = {t.name: t for t in tools}

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Returns (output_text, ok). Never raises: errors go back to the model."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools)
            return f"Unknown tool: {name}. Available tools: {available}", False
        try:
            return str(tool.fn(**args)), True
        except Exception as e:  # noqa: BLE001 - the model should see any failure
            return f"Tool error: {type(e).__name__}: {e}", False