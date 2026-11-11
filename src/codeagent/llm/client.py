"""OpenAI 兼容的 LLM 客户端（Ollama / DeepSeek / opencode-go relay 均可）。

从旧 RAG-M 的 rag/llm.py 复制精简：去掉 observability/tracing 依赖，用 no-op
装饰器保留调用点；保留 chat / chat_tools / stream_chat 与 extra_headers 的
$UUID 每次请求替换（opencode-go relay 的会话头要求）。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar

import httpx

# extra_headers 值里的占位符：每次请求替换为新 uuid4（opencode-go 要求 x-opencode-session）。
UUID_PLACEHOLDER = "$UUID"

F = TypeVar("F", bound=Callable[..., Any])


def traceable(**_kwargs: Any) -> Callable[[F], F]:
    """no-op 装饰器：旧实现接 LangSmith，code-agent 不引入 observability 依赖。"""

    def decorator(func: F) -> F:
        return func

    return decorator


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        temperature: float = 0.0,
        timeout: float = 120.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.extra_headers = {str(key): str(value) for key, value in (extra_headers or {}).items()}

    def _headers(self) -> dict[str, str]:
        """每次请求重新构造头：extra_headers 中的 $UUID 换成新的 uuid4（会话路由用）。"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for key, value in self.extra_headers.items():
            if UUID_PLACEHOLDER in value:
                value = value.replace(UUID_PLACEHOLDER, str(uuid.uuid4()))
            headers[key] = value
        return headers

    def _client(self) -> httpx.Client:
        local = "127.0.0.1" in self.base_url or "localhost" in self.base_url
        return httpx.Client(timeout=self.timeout, trust_env=not local)

    def _payload(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        stream: bool,
        json_mode: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @traceable(run_type="llm", name="llm_chat")
    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 1024, json_mode: bool = False) -> dict[str, Any]:
        with self._client() as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, max_tokens, False, json_mode),
            )
            if resp.status_code >= 400:
                raise LLMError(f"LLM {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
        usage = data.get("usage") or {}
        return {
            "text": data["choices"][0]["message"]["content"] or "",
            "tokens_in": int(usage.get("prompt_tokens") or 0),
            "tokens_out": int(usage.get("completion_tokens") or 0),
        }

    @traceable(run_type="llm", name="llm_chat_tools")
    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """带工具调用的 chat：返回 text / tool_calls / usage。"""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": "auto",
        }
        with self._client() as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
            )
            if resp.status_code >= 400:
                raise LLMError(f"LLM {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return {
            "text": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "tokens_in": int(usage.get("prompt_tokens") or 0),
            "tokens_out": int(usage.get("completion_tokens") or 0),
        }

    def stream_chat(self, messages: list[dict[str, Any]], max_tokens: int = 1024) -> Iterator[str]:
        with self._client() as client, client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(messages, max_tokens, True, False),
        ) as resp:
            if resp.status_code >= 400:
                raise LLMError(f"LLM {resp.status_code}: {resp.read()[:300]}")
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
