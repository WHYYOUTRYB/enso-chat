from src.web.chat_helpers import (
    SYSTEM_PROMPT,
    append_user,
    init_messages,
    parse_tool_step,
    should_summarize,
)


def test_init_messages_has_system_only():
    msgs = init_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_init_messages_custom_prompt():
    msgs = init_messages("custom sys")
    assert msgs[0]["content"] == "custom sys"


def test_append_user_returns_messages_with_user():
    msgs = init_messages()
    out = append_user(msgs, "hello")
    assert out is msgs  # mutated in place
    assert out[-1]["role"] == "user"
    assert out[-1]["content"] == "hello"


def test_parse_tool_step_shape():
    d = parse_tool_step(2, "classify_phase", {"value": 0.7}, "El Niño")
    assert d["step"] == 2
    assert d["name"] == "classify_phase"
    assert d["args"] == {"value": 0.7}
    assert d["result"] == "El Niño"
    # Long result gets a truncated preview
    long_result = "x" * 500
    d2 = parse_tool_step(1, "read_results", {}, long_result)
    assert len(d2["result_preview"]) <= 220
    assert d2["result_preview"].endswith("…")


def test_should_summarize_delegates():
    assert should_summarize([{"role": "user", "content": "hi"}]) is False
