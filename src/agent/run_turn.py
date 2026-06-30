"""Turn-by-turn agentic loop for the conversational ENSO agent.

Unlike the one-shot ``run_agent`` in the original project, ``run_turn``
receives an *external* message list (the conversation history), runs one turn
(model reply + any tool calls until the model stops calling tools), then
returns control to the caller so the user can send the next message.

The heavy objects (ENSO series, results) live on the shared ``ToolContext``
that owns the tool registry; messages carry only text summaries.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.client import AssistantMessage, DeepSeekError, LLMClient
from src.agent.tools import ToolRegistry
from src.config import (
    AGENT_LOOP_LIMIT,
    AGENT_MAX_RETRIES,
    AGENT_MAX_STURNS,
    AGENT_RETRY_BASE_DELAY,
    AGENT_RETRY_MAX_DELAY,
)


@dataclass
class TurnResult:
    """Outcome of a single conversational turn."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    stopped_reason: str = ""  # "" | "max_steps" | "loop_detected"


def _freeze(arguments: dict[str, Any]) -> str:
    """Normalize a tool-call's arguments into a stable string for comparison."""
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def _chat_with_retry(
    client: LLMClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> AssistantMessage:
    """Call ``client.chat`` with exponential backoff on transient failures."""
    attempt = 0
    while True:
        try:
            return client.chat(messages, tools)
        except DeepSeekError as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
            attempt += 1
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, base_delay * 0.1)
            time.sleep(delay)


def run_turn(
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    client: LLMClient,
    *,
    on_step: Callable[[int, str, dict[str, Any], str], None] | None = None,
    max_steps: int = AGENT_MAX_STURNS,
    loop_limit: int = AGENT_LOOP_LIMIT,
    max_retries: int = AGENT_MAX_RETRIES,
    base_delay: float = AGENT_RETRY_BASE_DELAY,
    max_delay: float = AGENT_RETRY_MAX_DELAY,
) -> TurnResult:
    """Run one conversational turn until the model stops calling tools.

    Args:
        messages: The full conversation history (already includes system +
            prior turns + the user's new message). Mutated in place: assistant
            and tool messages are appended. Also returned in ``TurnResult``.
        tools: Registry of available tools.
        client: The LLM client (DeepSeek).
        on_step: Optional callback ``(step, tool_name, arguments, result)``
            fired for every tool call, so the UI can render a folding block
            showing the call and its result.
        max_steps: Hard ceiling on assistant turns within this single turn.
        loop_limit: Consecutive identical tool-call signatures that trip early
            termination (a stuck model).
        max_retries / base_delay / max_delay: retry backoff for transient
            client errors.

    Returns:
        :class:`TurnResult` with the updated messages, the final assistant
        text, the tool calls made this turn, and a stop reason.
    """
    result = TurnResult(messages=messages)
    last_sig: tuple | None = None
    repeat_count = 0

    step = 0
    while step < max_steps:
        step += 1
        assistant = _chat_with_retry(
            client,
            messages,
            tools.schemas(),
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        messages.append(assistant.to_openai_message())

        if not assistant.tool_calls:
            result.steps = step
            result.final_text = assistant.content
            return result

        # Execute every requested tool call this assistant turn.
        for call in assistant.tool_calls:
            res = tools.execute(call.name, call.arguments)
            if on_step is not None:
                on_step(step, call.name, call.arguments, res)
            result.tool_calls.append(
                {"step": step, "name": call.name, "arguments": call.arguments, "result": res}
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": res}
            )

        # Loop detection: identical set of (tool, arguments) calls in a row.
        sig = tuple((c.name, _freeze(c.arguments)) for c in assistant.tool_calls)
        if sig == last_sig:
            repeat_count += 1
        else:
            last_sig = sig
            repeat_count = 1
        if repeat_count >= loop_limit:
            result.steps = step
            result.stopped_reason = "loop_detected"
            result.final_text = (
                f"Agent stopped: detected repeated identical tool calls "
                f"({repeat_count}x). Last call: {assistant.tool_calls[0].name}."
            )
            return result

    result.steps = step
    result.stopped_reason = "max_steps"
    result.final_text = f"Agent stopped after reaching max_steps={max_steps} this turn."
    return result
