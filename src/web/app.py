"""Streamlit chat UI for the conversational ENSO agent.

Run with::

    pip install streamlit
    streamlit run src/web/app.py

The page drives the turn-by-turn loop (``run_turn``) with a DeepSeek client.
Outputs go to a per-session temp directory. No OfflineClient — conversation
requires an LLM; without a key the chat input is disabled.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend before any src import

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.agent.client import DeepSeekClient, DeepSeekError
from src.agent.run_turn import run_turn
from src.agent.summarizer import summarize_old_messages
from src.agent.tools import ToolContext, build_tools
from src.web.chat_helpers import (
    SYSTEM_PROMPT,
    append_user,
    hint_no_key,
    init_messages,
    parse_tool_step,
    should_summarize,
)


def _session_base_dir() -> Path:
    if "base_dir" not in st.session_state:
        path = Path(tempfile.mkdtemp(prefix="enso_chat_"))
        atexit.register(shutil.rmtree, path, ignore_errors=True)
        st.session_state["base_dir"] = path
    return st.session_state["base_dir"]


def _get_messages() -> list[dict]:
    if "messages" not in st.session_state:
        st.session_state["messages"] = init_messages(SYSTEM_PROMPT)
    return st.session_state["messages"]


def _resolve_client(api_key: str, model_choice: str):
    """Return an LLM client or None (no key -> disabled chat).

    ``model_choice`` selects the backend ("DeepSeek" or "GLM"); each resolves
    its key with priority: sidebar input > Streamlit secrets > env var. The
    secrets path is for cloud deployment where there is no shell env var.
    """
    key = (api_key or "").strip()
    env_name = "DEEPSEEK_API_KEY" if model_choice == "DeepSeek" else "GLM_API_KEY"
    secret_name = "DEEPSEEK_API_KEY" if model_choice == "DeepSeek" else "GLM_API_KEY"
    if not key:
        try:
            key = st.secrets.get(secret_name, "")
        except Exception:  # noqa: BLE001 — st.secrets raises if no secrets file
            key = ""
    if not key:
        key = os.environ.get(env_name, "")
    if not key:
        return None
    try:
        if model_choice == "GLM":
            from src.agent.glm_client import GLMClient

            return GLMClient(api_key=key)
        return DeepSeekClient(api_key=key)
    except DeepSeekError:
        return None


def _handle_uploaded_csv(uploaded) -> str | None:
    """Save an uploaded ENSO CSV to the session temp dir; return its path.

    Returns the saved path (stored on the context so the agent can load it via
    load_user_enso), or None if nothing was uploaded this run.
    """
    if uploaded is None:
        return None
    base_dir = _session_base_dir()
    user_dir = base_dir / "data" / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / uploaded.name
    path.write_bytes(uploaded.getvalue())
    st.session_state["user_csv_path"] = str(path)
    return str(path)


def _new_figures(ctx) -> list[Path]:
    """Return figures added since the last render (and mark them shown)."""
    shown = st.session_state.setdefault("shown_figures", set())
    fresh = [p for p in ctx.figure_paths if str(p) not in shown]
    shown.update(str(p) for p in fresh)
    return fresh


def main() -> None:
    st.set_page_config(page_title="ENSO 对话 Agent", page_icon="🌊", layout="wide")
    st.title("🌊 ENSO 对话式 Agent")

    with st.sidebar:
        st.header("配置")
        model_choice = st.radio("LLM 后端", ["DeepSeek", "GLM"], index=0,
                                help="DeepSeek 默认；GLM(智谱)国内直连更稳，可切换对比")
        api_key = st.text_input(f"{model_choice} API Key", type="password",
                                placeholder="留空读环境变量")
        uploaded = st.file_uploader("上传 ENSO CSV（date+nino34 列）", type=["csv"],
                                    help="上传后会得到路径，让 agent 用 load_user_enso 加载")
        if uploaded is not None:
            path = _handle_uploaded_csv(uploaded)
            st.success(f"已上传: {Path(path).name}")
            st.caption(f"路径: {path}")
            st.caption("在对话里说「用我上传的数据」即可。")
        if st.button("🗑️ 清空对话"):
            st.session_state["messages"] = init_messages(SYSTEM_PROMPT)
            st.rerun()

    # Lazy-init shared tool context (persists across turns).
    if "ctx" not in st.session_state:
        st.session_state["ctx"] = ToolContext(base_dir=_session_base_dir())
        st.session_state["tools"] = build_tools(st.session_state["ctx"])
    tools = st.session_state["tools"]
    ctx = st.session_state["ctx"]
    client = _resolve_client(api_key, model_choice)

    messages = _get_messages()

    # Render conversation history (skip the system prompt).
    for msg in messages:
        if msg["role"] == "system":
            if msg["content"].startswith("历史摘要"):
                with st.chat_message("assistant"):
                    st.caption("📝 " + msg["content"])
            continue
        if msg["role"] == "tool":
            continue  # tool results are shown inside fold blocks, not as bubbles
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"] or "")

    # Chat input.
    user_input = st.chat_input("问点 ENSO 相关的…", disabled=(client is None))
    if client is None:
        st.info(hint_no_key())
    if not user_input:
        return

    append_user(messages, user_input)

    if should_summarize(messages):
        messages = summarize_old_messages(messages, client)
        st.session_state["messages"] = messages

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        steps_box = []
        with st.spinner("思考中…"):
            try:
                result = run_turn(
                    messages, tools, client,
                    on_step=lambda s, n, a, r: steps_box.append(parse_tool_step(s, n, a, r)),
                )
            except DeepSeekError as exc:
                st.error(f"DeepSeek 调用失败：{exc.message}")
                return
        for step_dict in steps_box:
            _render_tool_step(step_dict)
        # Show any figures produced this turn inline.
        for fig in _new_figures(ctx):
            st.image(str(fig), caption=fig.name, use_container_width=True)
        st.markdown(result.final_text or "(无回复)")
        if result.stopped_reason:
            st.caption(f"stopped: {result.stopped_reason}")


def _render_tool_step(step_dict: dict) -> None:
    with st.expander(f"🔧 step {step_dict['step']}: {step_dict['name']}", expanded=False):
        st.code(str(step_dict["args"]), language="python")
        st.text(step_dict["result_preview"])


if __name__ == "__main__":
    main()
