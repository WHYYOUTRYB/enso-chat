"""Pure helpers for the Streamlit chat UI (no ``streamlit`` import here).

Keeping these free of Streamlit means they can be unit-tested directly and
reused by any other frontend. The Streamlit page (``app.py``) calls these and
wraps the results in ``st.*`` components.
"""

from __future__ import annotations

from typing import Any

from src.agent.summarizer import should_summarize as _should_summarize

SYSTEM_PROMPT = """你是 ENSO 预报对话助手。用户会自由提问 ENSO / 海气环境预报相关问题。

你可以调用工具完成：加载数据、预测目标月 Niño3.4、诊断本地数据、推荐数据范围、
画图、降水与潮汐分析。每轮调用工具后，用中文向用户解释结果；不确定时如实说明
不确定性，不要编造未提供的数值。

lead（提前量）可信度：1-6 月正常，7-11 月低可信度，>=12 个月拒绝预测（超出可靠范围）。
"""

_RESULT_PREVIEW_LIMIT = 200


def init_messages(system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, Any]]:
    """Start a fresh conversation: a single system message."""
    return [{"role": "system", "content": system_prompt}]


def append_user(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Append a user message; mutate and return the same list."""
    messages.append({"role": "user", "content": text})
    return messages


def parse_tool_step(
    step: int, name: str, args: dict[str, Any], result: str
) -> dict[str, Any]:
    """Turn an on_step callback's args into a dict for rendering a fold block."""
    preview = result if len(result) <= _RESULT_PREVIEW_LIMIT else result[:_RESULT_PREVIEW_LIMIT] + "…"
    return {"step": step, "name": name, "args": args, "result": result, "result_preview": preview}


def should_summarize(messages: list[dict[str, Any]]) -> bool:
    """True if the conversation history exceeds the token threshold."""
    return _should_summarize(messages)


def hint_no_key() -> str:
    """Text shown when no DeepSeek API key is available."""
    return "对话需要 DeepSeek API key。请在侧栏填入，或设置 DEEPSEEK_API_KEY 环境变量。"
