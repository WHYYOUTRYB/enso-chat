"""Pluggable LLM clients for the agent layer.

:class:`DeepSeekClient` — OpenAI-compatible chat-completions over ``urllib``
(no extra dependency). DeepSeek's ``deepseek-chat`` model supports function
calling; ``deepseek-reasoner`` does NOT, so callers must use ``deepseek-chat``.

Returns :class:`AssistantMessage` objects whose ``tool_calls`` carry
already-parsed argument dicts, so the loop is client-agnostic.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.config import (
    AGENT_REQUEST_TIMEOUT,
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_API_URL,
    DEEPSEEK_BASE_URL_ENV,
    DEEPSEEK_CHAT_PATH,
    DEEPSEEK_MODEL,
    DEEPSEEK_MODEL_ENV,
)


@dataclass
class ToolCall:
    """A single tool invocation requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantMessage:
    """The assistant's response: free text plus zero or more tool calls."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_openai_message(self) -> dict[str, Any]:
        """Serialize to the OpenAI/DeepSeek assistant-message wire format."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in self.tool_calls
            ]
        return message


class LLMClient(Protocol):
    """Interface every client implements."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> AssistantMessage:
        ...


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek API call fails or returns an error payload.

    ``retryable`` flags transient failures (429/5xx/network) that the agent
    loop can safely retry; non-retryable errors (auth, bad request, parse
    failures) propagate immediately.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


def _resolve_deepseek_config(
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> tuple[str, str, str]:
    key = api_key or os.environ.get(DEEPSEEK_API_KEY_ENV)
    if not key:
        raise DeepSeekError(
            f"No DeepSeek API key found. Set the {DEEPSEEK_API_KEY_ENV} environment variable "
            f"(get one at https://platform.deepseek.com), or use the offline client."
        )
    url = base_url or os.environ.get(DEEPSEEK_BASE_URL_ENV) or DEEPSEEK_API_URL
    mdl = model or os.environ.get(DEEPSEEK_MODEL_ENV) or DEEPSEEK_MODEL
    if mdl == "deepseek-reasoner":
        raise DeepSeekError(
            "deepseek-reasoner does not support function calling. "
            "Use deepseek-chat (the default) for the agentic tool loop."
        )
    return key, url, mdl


class DeepSeekClient:
    """OpenAI-compatible chat client targeting the DeepSeek API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = AGENT_REQUEST_TIMEOUT,
    ):
        self.api_key, self.base_url, self.model = _resolve_deepseek_config(api_key, base_url, model)
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + DEEPSEEK_CHAT_PATH

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> AssistantMessage:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tool_choice": tool_choice,
        }
        if tools:
            payload["tools"] = tools

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            # 429 (rate limit) and 5xx (server) are transient; 4xx auth/client
            # errors (401/403/400/404) will not resolve by retrying.
            retryable = exc.code == 429 or exc.code >= 500
            raise DeepSeekError(
                f"DeepSeek API returned HTTP {exc.code}: {detail}",
                retryable=retryable,
            ) from exc
        except urllib.error.URLError as exc:
            # DNS failure, connection refused, socket timeout — all transient.
            raise DeepSeekError(
                f"DeepSeek API request failed: {exc.reason}",
                retryable=True,
            ) from exc
        except OSError as exc:
            # Network-layer fallback: treat as transient.
            raise DeepSeekError(
                f"DeepSeek API request failed: {exc}",
                retryable=True,
            ) from exc

        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: bytes) -> AssistantMessage:
        data = json.loads(raw.decode("utf-8", errors="replace"))
        if data.get("error"):
            raise DeepSeekError(f"DeepSeek API error: {json.dumps(data['error'], ensure_ascii=False)}")
        choices = data.get("choices") or []
        if not choices:
            raise DeepSeekError(f"DeepSeek API returned no choices: {raw.decode('utf-8', 'replace')}")
        message = choices[0].get("message", {})
        content = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function", {})
            args_raw = function.get("arguments", "{}")
            try:
                arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": args_raw}
            tool_calls.append(
                ToolCall(
                    id=raw_call.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
                )
            )
        return AssistantMessage(content=content, tool_calls=tool_calls)
