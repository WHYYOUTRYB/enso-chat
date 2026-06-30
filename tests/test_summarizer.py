from src.agent.client import AssistantMessage, DeepSeekError
from src.agent.summarizer import (
    TOKEN_THRESHOLD,
    estimate_tokens,
    should_summarize,
    summarize_old_messages,
)


class _SummaryClient:
    """Returns a canned summary string for the chat call."""

    def __init__(self, summary_text="SUMMARY"):
        self._text = summary_text
        self.calls = 0

    def chat(self, messages, tools, tool_choice="auto"):
        self.calls += 1
        return AssistantMessage(content=self._text, tool_calls=[])


class _FailingClient:
    def chat(self, messages, tools, tool_choice="auto"):
        raise DeepSeekError("boom", retryable=False)


def test_estimate_tokens_grows_with_messages():
    few = [{"role": "user", "content": "hi"}]
    many = [{"role": "user", "content": "x" * 1000}]
    assert estimate_tokens(few) < estimate_tokens(many)


def test_should_summarize_threshold():
    assert should_summarize([{"role": "user", "content": "hi"}]) is False
    huge = [{"role": "user", "content": "x" * (TOKEN_THRESHOLD * 2)}]
    assert should_summarize(huge) is True


def test_summarize_keeps_system_and_recent():
    messages = [{"role": "system", "content": "SYS"}]
    messages += [{"role": "user", "content": f"old{i}"} for i in range(10)]
    messages += [{"role": "assistant", "content": f"a{i}"} for i in range(10)]
    messages += [{"role": "user", "content": "recent1"},
                 {"role": "assistant", "content": "recent2"}]
    client = _SummaryClient("the summary")
    out = summarize_old_messages(messages, client, keep_recent=2)
    # System kept
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "SYS"
    # Summary injected as a second system message
    assert out[1]["role"] == "system"
    assert out[1]["content"] == "the summary"
    # Last 2 messages kept verbatim
    assert out[-1]["content"] == "recent2"
    assert out[-2]["content"] == "recent1"
    # Fewer messages than before (compressed)
    assert len(out) < len(messages)


def test_summarize_falls_back_on_failure():
    messages = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"}]
    out = summarize_old_messages(messages, _FailingClient(), keep_recent=2)
    # On failure, returns original messages unchanged
    assert out == messages


def test_summarize_skips_when_too_few_messages():
    messages = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "u1"}]
    client = _SummaryClient()
    out = summarize_old_messages(messages, client, keep_recent=6)
    # Not enough to summarize -> unchanged, no chat call
    assert out == messages
    assert client.calls == 0
