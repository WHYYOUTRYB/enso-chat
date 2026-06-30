from src.agent.client import AssistantMessage, ToolCall
from src.agent.run_turn import TurnResult, run_turn
from src.agent.tools import ToolContext, build_tools


class _ScriptedClient:
    """Returns canned AssistantMessages in sequence (one per chat call)."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = 0

    def chat(self, messages, tools, tool_choice="auto"):
        idx = min(self.calls, len(self._turns) - 1)
        self.calls += 1
        return self._turns[idx]


def _tools(tmp_path):
    return build_tools(ToolContext(base_dir=tmp_path))


def test_run_turn_no_tool_calls_returns_text(tmp_path):
    client = _ScriptedClient([AssistantMessage(content="hello", tool_calls=[])])
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"}]
    res = run_turn(messages, _tools(tmp_path), client)
    assert isinstance(res, TurnResult)
    assert res.final_text == "hello"
    assert res.tool_calls == []
    assert res.stopped_reason == ""
    # messages got the assistant reply appended
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "hello"


def test_run_turn_executes_one_tool_then_finishes(tmp_path):
    client = _ScriptedClient([
        AssistantMessage(content="thinking",
                         tool_calls=[ToolCall(id="1", name="classify_phase",
                                              arguments={"value": 0.8})]),
        AssistantMessage(content="done", tool_calls=[]),
    ])
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "classify"}]
    res = run_turn(messages, _tools(tmp_path), client)
    assert res.final_text == "done"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "classify_phase"
    # tool result fed back into messages
    roles = [m["role"] for m in messages]
    assert "tool" in roles
    tool_msg = [m for m in messages if m["role"] == "tool"][0]
    assert "El Niño" in tool_msg["content"]


def test_run_turn_on_step_callback_receives_result(tmp_path):
    client = _ScriptedClient([
        AssistantMessage(content="c",
                         tool_calls=[ToolCall(id="1", name="classify_phase",
                                              arguments={"value": 0.8})]),
        AssistantMessage(content="done", tool_calls=[]),
    ])
    seen = []

    def on_step(step, name, args, result):
        seen.append((step, name, result))

    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "x"}]
    run_turn(messages, _tools(tmp_path), client, on_step=on_step)
    assert len(seen) == 1
    assert seen[0][0] == 1
    assert seen[0][1] == "classify_phase"
    assert "El Niño" in seen[0][2]  # result string passed to callback


def test_run_turn_max_steps_ceiling(tmp_path):
    # Always calls classify_phase -> never finishes naturally.
    client = _ScriptedClient([
        AssistantMessage(content="loop",
                         tool_calls=[ToolCall(id="x", name="classify_phase",
                                              arguments={"value": 0})]),
    ])
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "x"}]
    res = run_turn(messages, _tools(tmp_path), client, max_steps=3, loop_limit=100)
    assert res.stopped_reason == "max_steps"
    assert res.steps == 3


def test_run_turn_loop_detection(tmp_path):
    client = _ScriptedClient([
        AssistantMessage(content="loop",
                         tool_calls=[ToolCall(id="x", name="classify_phase",
                                              arguments={"value": 0})]),
    ])
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "x"}]
    res = run_turn(messages, _tools(tmp_path), client, max_steps=10, loop_limit=3)
    assert res.stopped_reason == "loop_detected"
    assert res.steps == 3


def test_run_turn_mutates_external_messages(tmp_path):
    client = _ScriptedClient([AssistantMessage(content="done", tool_calls=[])])
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"}]
    original_len = len(messages)
    run_turn(messages, _tools(tmp_path), client)
    assert len(messages) == original_len + 1  # assistant reply appended
