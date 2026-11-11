"""LLM 客户端与路由测试（无网络）：$UUID 头、响应解析、降级/熔断/首 token 超时。"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from codeagent.config import Settings
from codeagent.llm.client import UUID_PLACEHOLDER, LLMClient, LLMError
from codeagent.llm.router import FallbackLLM, build_llm


class FakeClient:
    def __init__(self, name: str, *, fail: bool = False, stream: list[str] | None = None) -> None:
        self.name = name
        self.fail = fail
        self.stream = list(stream or [name])
        self.calls = 0

    def chat(self, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False) -> dict[str, Any]:
        self.calls += 1
        if self.fail:
            raise LLMError(f"{self.name} down")
        return {"text": self.name, "tokens_in": 1, "tokens_out": 2}

    def chat_tools(self, messages: list[dict], tools: list[dict], max_tokens: int = 1024) -> dict[str, Any]:
        self.calls += 1
        if self.fail:
            raise LLMError(f"{self.name} down")
        return {"text": self.name, "tool_calls": [], "tokens_in": 1, "tokens_out": 2}

    def stream_chat(self, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        self.calls += 1
        yield from self.stream


def _mock(client: LLMClient, handler) -> None:
    client._client = lambda: httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[method-assign]


def test_uuid_header_rotates() -> None:
    client = LLMClient("http://127.0.0.1:1/v1", "m", extra_headers={"x-opencode-session": UUID_PLACEHOLDER})
    first = client._headers()
    second = client._headers()
    assert first["Authorization"] == "Bearer ollama"
    assert first["x-opencode-session"] != second["x-opencode-session"]
    assert UUID_PLACEHOLDER not in first["x-opencode-session"]
    assert len(first["x-opencode-session"]) == 36


def test_chat_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "m"
        assert body["messages"][0]["content"] == "q"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
        )

    client = LLMClient("http://llm.local/v1", "m")
    _mock(client, handler)
    assert client.chat([{"role": "user", "content": "q"}]) == {
        "text": "hi",
        "tokens_in": 3,
        "tokens_out": 4,
    }


def test_chat_tools_parses_tool_calls() -> None:
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_code", "arguments": '{"query": "x"}'},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"] and body["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": None, "tool_calls": calls}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 6},
            },
        )

    client = LLMClient("http://llm.local/v1", "m")
    _mock(client, handler)
    result = client.chat_tools([{"role": "user", "content": "q"}], [{"type": "function"}])
    assert result["tool_calls"] == calls
    assert result["text"] == ""
    assert result["tokens_in"] == 5


def test_http_error_raises_llm_error() -> None:
    client = LLMClient("http://llm.local/v1", "m")
    _mock(client, lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="LLM 500"):
        client.chat([{"role": "user", "content": "q"}])


def test_stream_chat_yields_tokens() -> None:
    payload = 'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: [DONE]\n'

    client = LLMClient("http://llm.local/v1", "m")
    _mock(client, lambda request: httpx.Response(200, text=payload))
    assert list(client.stream_chat([{"role": "user", "content": "q"}])) == ["a"]


def test_fallback_switches_provider() -> None:
    bad, good = FakeClient("bad", fail=True), FakeClient("good")
    router = FallbackLLM([bad, good], chat_timeout=5, first_token_timeout=5)
    assert router.provider_names == ["bad", "good"]
    assert router.chat([{"role": "user", "content": "q"}])["text"] == "good"
    assert router.last_provider == "good"
    assert router.provider_status()["bad"]["failures"] == 1
    assert router.provider_status()["good"]["failures"] == 0
    assert bad.calls == 1


def test_circuit_breaker_opens_and_skips() -> None:
    bad1, bad2 = FakeClient("bad1", fail=True), FakeClient("bad2", fail=True)
    router = FallbackLLM([bad1, bad2], fail_threshold=1, cooldown=60)
    with pytest.raises(LLMError, match="all LLM providers failed"):
        router.chat([{"role": "user", "content": "q"}])
    assert router.provider_status() == {
        "bad1": {"state": "open", "failures": 1},
        "bad2": {"state": "open", "failures": 1},
    }
    with pytest.raises(LLMError):
        router.chat([{"role": "user", "content": "q"}])
    assert bad1.calls == 1  # 熔断后再调用被直接拒绝


def test_stream_first_token_timeout_falls_back() -> None:
    class SlowStream(FakeClient):
        def stream_chat(self, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
            self.calls += 1
            time.sleep(0.2)
            yield "slow"

    slow, good = SlowStream("slow"), FakeClient("good", stream=["ok1", "ok2"])
    router = FallbackLLM([slow, good], chat_timeout=1, first_token_timeout=0.05)
    assert list(router.stream_chat([{"role": "user", "content": "q"}])) == ["ok1", "ok2"]
    assert router.last_provider == "good"


def test_no_providers_raises() -> None:
    router = FallbackLLM([{"name": "broken", "model": "m"}], chat_timeout=1)
    assert router.providers == []
    assert router.skipped and router.skipped[0]["reason"] == "base_url or model missing"
    with pytest.raises(LLMError):
        router.chat([])


def test_build_llm_from_settings() -> None:
    payload = [{"name": "local", "base_url": "http://127.0.0.1:1/v1", "model": "m"}]
    router = build_llm(Settings(llm_providers=json.dumps(payload), _env_file=None))
    assert router.provider_names == ["local"]
    assert router.clients[0].base_url == "http://127.0.0.1:1/v1"
