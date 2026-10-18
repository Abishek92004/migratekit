from migratekit.tracing import (
    CompactionEvent,
    LLMCallEvent,
    ToolCallEvent,
    Tracer,
    read_events,
)


def test_write_and_read_back(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path)

    tracer.emit(LLMCallEvent(
        run_id="run_001", agent_id="worker_1", parent_agent_id="orchestrator",
        step=1, start_ts=100.0, end_ts=103.5, model="example-model",
        prompt_tokens=5200, completion_tokens=300, shared_prefix_tokens=4800,
    ))
    tracer.emit(ToolCallEvent(
        run_id="run_001", agent_id="worker_1", step=2,
        start_ts=104.0, end_ts=146.0, tool="run_tests",
    ))
    tracer.emit(CompactionEvent(
        run_id="run_001", agent_id="worker_1", step=9,
        start_ts=200.0, end_ts=204.0, strategy="summarize_all",
        tokens_before=30000, tokens_after=4000,
    ))

    events = list(read_events(path))
    assert [e.type for e in events] == ["llm_call", "tool_call", "compaction"]
    assert events[0].duration_s == 3.5
    assert events[1].tool == "run_tests"
    assert events[2].tokens_after == 4000