"""Trace event schema.

Every LLM call, tool call and compaction in a run is logged as one event.
These logs are the raw data for the research paper, so field names and
meanings should stay stable once we start collecting runs.
"""
from __future__ import annotations

import uuid
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter


def new_id() -> str:
    return uuid.uuid4().hex


class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=new_id)
    run_id: str
    agent_id: str                      # e.g. "planner", "worker_2"
    parent_agent_id: Optional[str] = None   # who spawned this agent
    step: Optional[int] = None         # step counter inside the agent's loop
    start_ts: float                    # unix time, seconds
    end_ts: float
    success: bool = True
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts


class LLMCallEvent(BaseEvent):
    type: Literal["llm_call"] = "llm_call"
    model: str
    backend: str = "api"               # "api" now, "vllm" later
    prompt_tokens: int
    completion_tokens: int
    # How many tokens at the start of this prompt are identical to this
    # agent's previous prompt. Core measurement for the KV-cache study.
    shared_prefix_tokens: Optional[int] = None
    ttft_s: Optional[float] = None     # time to first token, if known
    # Hash of each prompt part (system, tools, repo_context, history, new).
    segment_hashes: dict[str, str] = Field(default_factory=dict)
    cost_usd: Optional[float] = None


class ToolCallEvent(BaseEvent):
    type: Literal["tool_call"] = "tool_call"
    tool: str                          # e.g. "run_tests"
    arg_bytes: int = 0
    result_bytes: int = 0


class CompactionEvent(BaseEvent):
    type: Literal["compaction"] = "compaction"
    strategy: str                      # e.g. "summarize_all", "summarize_tail"
    tokens_before: int
    tokens_after: int


TraceEvent = Annotated[
    Union[LLMCallEvent, ToolCallEvent, CompactionEvent],
    Field(discriminator="type"),
]

event_adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)