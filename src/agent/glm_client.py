"""GLM (Zhipu BigModel) LLM client — OpenAI-compatible, function-calling capable.

Mirrors :class:`DeepSeekClient` via the same :class:`LLMClient` Protocol. GLM's
OpenAI-compatible endpoint (``open.bigmodel.cn/api/paas/v4``) accepts the same
``tools`` / ``tool_calls`` / ``tool_call_id`` schema, so the agent loop, tool
registry, and tests need zero changes — swapping the client is the only diff.

Authentication: Bearer token from ``GLM_API_KEY`` env (or constructor arg).
The web layer (sidebar > st.secrets > env) resolves the key before constructing
this client, mirroring the DeepSeek path.
"""

from __future__ import annotations

import os

from src.agent.client import AssistantMessage, DeepSeekError, LLMClient
from src.config import AGENT_REQUEST_TIMEOUT, GLM_API_KEY_ENV, GLM_API_URL, GLM_CHAT_PATH, GLM_MODEL


def _resolve_glm_config(api_key: str | None, base_url: str | None, model: str | None) -> tuple[str, str, str]:
    key = api_key or os.environ.get(GLM_API_KEY_ENV)
    if not key:
        raise DeepSeekError(
            f"No GLM API key found. Set the {GLM_API_KEY_ENV} environment variable "
            f"(get one at https://open.bigmodel.cn)."
        )
    url = base_url or os.environ.get("GLM_BASE_URL") or GLM_API_URL
    mdl = model or os.environ.get("GLM_MODEL") or GLM_MODEL
    return key, url, mdl


class GLMClient:
    """OpenAI-compatible chat client targeting the Zhipu GLM API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = AGENT_REQUEST_TIMEOUT,
    ):
        self.api_key, self.base_url, self.model = _resolve_glm_config(api_key, base_url, model)
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + GLM_CHAT_PATH

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> AssistantMessage:
        # Reuse DeepSeekClient's request/response machinery (urllib + retry
        # classification) by constructing one with GLM's resolved config.
        from src.agent.client import DeepSeekClient

        delegate = DeepSeekClient.__new__(DeepSeekClient)
        delegate.api_key = self.api_key
        delegate.base_url = self.base_url
        delegate.model = self.model
        delegate.timeout = self.timeout
        return delegate.chat(messages, tools, tool_choice)


# LLMClient is a Protocol — GLMClient satisfies it structurally without inheriting.
_ = LLMClient  # noqa: F841 (keep the import meaningful for type checkers)
