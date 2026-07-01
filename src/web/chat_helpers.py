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
画图、降水与潮汐分析、可靠性评估。每轮调用工具后，用中文向用户解释结果；
不确定时如实说明不确定性，不要编造未提供的数值。

## 工具选择规则（重要，避免调错工具）

- 用户要"对比多个方法/精度对比/综合预测" → 必须调 `compare_methods`，不要自己分别调几个预测工具。
- 用户问"实时/realtime 预测准不准/可不可靠" → 必须调 `report_realtime_skill`，不要调 `report_hindcast_skill`（后者是 SODA 训练域的，不能套到实时预测）。
- 用户问"SODA/训练域/方法上限" → 调 `report_hindcast_skill`。
- 不确定用哪个 hindcast 工具时：涉及"实时/实时数据/cross-domain/realtime"字样 → `report_realtime_skill`；否则 → `report_hindcast_skill`。
- 做了 `forecast_cnn_lstm(mode=realtime)` 预测后，若用户关心可靠性，主动调 `report_realtime_skill` 给出跨域 ACC。
- 做了 `forecast_enhanced` 预测后，结果里已带 ACC 和可信度标注，直接用，无需再调 hindcast 工具。

## lead 可信度

lead（提前量）可信度：1-6 月正常，7-11 月低可信度，>=12 个月拒绝预测（超出可靠范围）。
增强轨按 per-lead ACC 数据驱动分档（ACC<0.3 拒绝、<0.5 低可信），以工具返回的 ACC 为准。

## 回复格式

- 先给结论（预测值/相位/可信度一句话），再给依据（数据来源、ACC、对比基准）。
- 不要复述工具返回的原始字符串，要提炼成用户能懂的话。
- 涉及数值时，只用工具结果里实际出现的数字，禁止编造或外推。
- 回复保持简洁，避免长篇大论；用户追问细节时再展开。
- 跨域/实时预测必须保留"非实时/跨域精度低于 SODA hindcast"的标注，不要删。
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
