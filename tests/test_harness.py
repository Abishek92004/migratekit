from migratekit.harness import (
    AgentLoop, FakeClient, LLMResponse, Tool, ToolCall, ToolRegistry,
)
from migratekit.tracing import Tracer, read_events

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    "required": ["a", "b"],
}


def resp(text="", calls=(), inp=100, out=20):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for c in calls:
        content.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.input})
    return LLMResponse(text=text, tool_calls=list(calls), content=content,
                       input_tokens=inp, output_tokens=out)


def make_loop(tmp_path, script, tools, **kw):
    return AgentLoop(
        client=FakeClient(script), tools=ToolRegistry(tools),
        tracer=Tracer(tmp_path / "trace.jsonl"),
        run_id="r1", agent_id="worker_1", system="You are a test agent.",
        retry_base_delay_s=0, **kw,
    )


def test_tool_call_then_final_answer(tmp_path):
    add = Tool("add", "Add two numbers", ADD_SCHEMA, lambda a, b: a + b)
    loop = make_loop(tmp_path, [
        resp(calls=[ToolCall("t1", "add", {"a": 2, "b": 3})], inp=100),
        resp(text="The answer is 5", inp=150),
    ], [add])
    result = loop.run("What is 2+3?")

    assert result.stop_reason == "done"
    assert result.steps == 2
    assert result.final_text == "The answer is 5"
    events = list(read_events(tmp_path / "trace.jsonl"))
    assert [e.type for e in events] == ["llm_call", "tool_call", "llm_call"]
    assert events[0].shared_prefix_tokens is None
    assert events[2].shared_prefix_tokens == 100   # previous prompt reused


def test_tool_error_goes_back_to_model(tmp_path):
    def boom(a, b):
        raise ValueError("nope")

    bad = Tool("add", "Add", ADD_SCHEMA, boom)
    loop = make_loop(tmp_path, [
        resp(calls=[ToolCall("t1", "add", {"a": 1, "b": 2})]),
        resp(text="Sorry, the tool failed"),
    ], [bad])
    result = loop.run("add")

    tool_result = result.messages[2]["content"][0]
    assert tool_result["is_error"] is True
    assert "nope" in tool_result["content"]
    assert result.stop_reason == "done"


def test_max_steps_stops_runaway_agent(tmp_path):
    add = Tool("add", "Add", ADD_SCHEMA, lambda a, b: a + b)
    endless = [resp(calls=[ToolCall(f"t{i}", "add", {"a": 1, "b": 1})]) for i in range(10)]
    result = make_loop(tmp_path, endless, [add], max_steps=3).run("loop forever")
    assert result.stop_reason == "max_steps"
    assert result.steps == 3


def test_checkpoint_and_resume(tmp_path):
    add = Tool("add", "Add", ADD_SCHEMA, lambda a, b: a + b)
    ckpt = tmp_path / "ckpt.json"

    first = make_loop(tmp_path, [
        resp(calls=[ToolCall("t1", "add", {"a": 2, "b": 3})]),
    ], [add], max_steps=1, checkpoint_path=ckpt)
    assert first.run("2+3?").stop_reason == "max_steps"
    assert ckpt.exists()

    second = make_loop(tmp_path, [resp(text="5")], [add], checkpoint_path=ckpt)
    result = second.run("2+3?")        # resumes from the checkpoint
    assert result.stop_reason == "done"
    assert result.steps == 2           # 1 step before the crash + 1 after