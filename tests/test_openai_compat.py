from migratekit.harness.openai_compat import to_openai_messages, to_openai_tools


def test_message_translation():
    history = [
        {"role": "user", "content": "add 2 and 3"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "calling tool"},
            {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 2, "b": 3}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "5", "is_error": False},
        ]},
    ]
    out = to_openai_messages("sys", history)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "add"
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "5"}


def test_tool_spec_translation():
    spec = [{"name": "add", "description": "Add", "input_schema": {"type": "object"}}]
    out = to_openai_tools(spec)
    assert out[0]["function"]["parameters"] == {"type": "object"}