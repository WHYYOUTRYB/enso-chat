"""Compress old conversation history into a summary when it grows too long.

DeepSeek has a context window (~64k tokens). To keep a long conversation
usable, when the estimated token count exceeds ``TOKEN_THRESHOLD`` the oldest
messages (everything except the original system prompt and the most recent
``keep_recent`` messages) are sent to the model with a "summarize" instruction,
and the result is injected back as a single system message.

The estimate is deliberately crude (character count) — it only drives the
trigger decision, not billing. On any failure the original messages are
returned unchanged: never block the conversation because summarization failed.
"""

from __future__ import annotations

from typing import Any

from src.agent.client import DeepSeekError, LLMClient

TOKEN_THRESHOLD = 20000  # leave headroom under the ~64k context window

_SUMMARY_SYSTEM = (
    "你是对话摘要助手。把以下对话压缩成要点，保留：关键预测结果、数据事实、"
    "用户意图与已调用工具的结论。不要编造未出现的数值。用简短要点输出。"
)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate: ~1 char ≈ 1 token for CJK, ~0.25 for ASCII.

    Imprecise by design — only used to decide whether to summarize.
    """
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        # Crude: count chars; ASCII-heavy text is overcounted, which is safe
        # (we summarize a bit early rather than too late).
        total += len(content)
    return total


def should_summarize(messages: list[dict[str, Any]]) -> bool:
    """True if the estimated token count exceeds the threshold."""
    return estimate_tokens(messages) > TOKEN_THRESHOLD


def summarize_old_messages(
    messages: list[dict[str, Any]],
    client: LLMClient,
    *,
    keep_recent: int = 6,
) -> list[dict[str, Any]]:
    """Compress old messages into a summary; keep system + recent N.

    Returns ``[original_system, summary_system, ...keep_recent]``.
    On failure (DeepSeek error) returns the original messages unchanged.

    If there are too few messages to summarize (<= keep_recent + 1), returns
    them unchanged without calling the model.
    """
    # Need at least: system + something to summarize + keep_recent.
    if len(messages) <= 1 + keep_recent:
        return messages

    original_system = messages[0]
    old = messages[1 : len(messages) - keep_recent]
    recent = messages[len(messages) - keep_recent :]

    old_text = "\n".join(
        f"[{m.get('role', '?')}] {m.get('content', '')}" for m in old
    )
    summary_messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": old_text},
    ]

    try:
        assistant = client.chat(summary_messages, tools=[], tool_choice="none")
    except DeepSeekError:
        return messages  # never block the conversation

    summary = (assistant.content or "").strip() or "(摘要为空)"
    return [original_system, {"role": "system", "content": summary}, *recent]
