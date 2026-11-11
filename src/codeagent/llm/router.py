"""多 provider LLM 降级：顺序回退、首 token 超时与熔断。

从旧 RAG-M 的 rag/llm_router.py 复制精简（去掉 observability/tracing 依赖）：

- `LLMClient` 保持对外行为不变；本模块只做编排，内部复用 `LLMClient`；
- `chat` / `chat_tools` 是同步接口：整请求超时后切换到下一个 provider；
- `stream_chat` 只在收到第一个 token 之前允许切换；首 token 之后异常直接抛出；
- 熔断器为失败阈值 + 冷却 + 单探针半开；provider 状态供 `/api/health` 使用。

未实现（本轮范围外）：负载均衡、按请求选择 provider、key 轮换。
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from ..config import Settings, get_settings
from .client import LLMClient, LLMError

logger = logging.getLogger(__name__)

DEFAULT_FAIL_THRESHOLD = 3
DEFAULT_COOLDOWN = 60.0
DEFAULT_CHAT_TIMEOUT = 60.0
DEFAULT_FIRST_TOKEN_TIMEOUT = 15.0


def _to_float(value: Any, default: float) -> float:
    """配置里的数字可能是 JSON number / 字符串；非法值退回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class CircuitBreaker:
    """失败计数 + 冷却后半开单探针的熔断器。

    - `closed`：正常放行，失败累计到阈值后进入 `open`；
    - `open`：冷却期内全部拒绝，不再打后端；
    - `half_open`：冷却结束后的单探针窗口，成功则 `closed`，失败则重新打开。
    """

    def __init__(
        self,
        fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
        cooldown: float = DEFAULT_COOLDOWN,
        *,
        name: str = "",
        clock: Callable[[], float] | None = None,
    ) -> None:
        if fail_threshold < 1:
            raise ValueError("fail_threshold must be >= 1")
        if cooldown < 0:
            raise ValueError("cooldown must be >= 0")
        self.fail_threshold = int(fail_threshold)
        self.cooldown = float(cooldown)
        self.name = name
        self.failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._clock = clock
        self._lock = threading.Lock()

    def _now(self) -> float:
        # 不绑定 time.monotonic，方便测试 monkeypatch 模块里的 time。
        return self._clock() if self._clock is not None else time.monotonic()

    def _state_locked(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._probe_in_flight:
            return "half_open"
        if self._now() - self._opened_at >= self.cooldown:
            return "half_open"
        return "open"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def allow_request(self) -> bool:
        """closed 直接放行；open 拒绝；冷却结束只放行一个半开探针。"""
        with self._lock:
            if self._opened_at is None:
                return True
            if self._now() - self._opened_at < self.cooldown or self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self._probe_in_flight = False
            if self.failures >= self.fail_threshold:
                # 半开探针失败 / 再次失败都重新计时冷却。
                self._opened_at = self._now()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"state": self._state_locked(), "failures": self.failures}


@dataclass(slots=True)
class _Provider:
    """一个已就绪的 provider（client + 独立熔断器 + 首 token 超时）。"""

    name: str
    client: Any
    breaker: CircuitBreaker
    first_token_timeout: float


class FallbackLLM:
    """按配置顺序调用多个 OpenAI 兼容 client，失败时自动降级。

    `providers` 既可以是 `Settings`（读取 `resolved_llm_providers`），也可以直接传
    provider 配置 dict 列表或现成的 client 列表（测试注入用）。
    """

    def __init__(
        self,
        providers: Sequence[Mapping[str, Any] | Any] | Mapping[str, Any] | Settings,
        *,
        chat_timeout: float | None = None,
        first_token_timeout: float | None = None,
        temperature: float | None = None,
        fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
        cooldown: float = DEFAULT_COOLDOWN,
        clock: Callable[[], float] | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        specs, timeout_defaults = self._normalize_input(
            providers,
            chat_timeout=chat_timeout,
            first_token_timeout=first_token_timeout,
            temperature=temperature,
        )
        self.chat_timeout = timeout_defaults["chat_timeout"]
        self.first_token_timeout = timeout_defaults["first_token_timeout"]
        self.temperature = timeout_defaults["temperature"]
        self.fail_threshold = int(fail_threshold)
        self.cooldown = float(cooldown)

        factory = client_factory or LLMClient
        self.providers: list[_Provider] = []
        self.skipped: list[dict[str, str]] = []
        self.last_provider: str | None = None

        for index, spec in enumerate(specs, start=1):
            provider = self._build_provider(
                spec,
                index=index,
                factory=factory,
                fail_threshold=self.fail_threshold,
                cooldown=self.cooldown,
                clock=clock,
            )
            if provider is not None:
                self.providers.append(provider)

        if not self.providers:
            logger.error("no usable LLM providers; chat/stream_chat will raise LLMError")

    # ---------- construction ----------
    @staticmethod
    def _normalize_input(
        providers: Sequence[Mapping[str, Any] | Any] | Mapping[str, Any] | Settings,
        *,
        chat_timeout: float | None,
        first_token_timeout: float | None,
        temperature: float | None,
    ) -> tuple[list[Any], dict[str, float]]:
        if hasattr(providers, "resolved_llm_providers"):
            settings = providers
            raw_specs = list(settings.resolved_llm_providers)
            if chat_timeout is None:
                chat_timeout = getattr(settings, "llm_chat_timeout", None)
            if first_token_timeout is None:
                first_token_timeout = getattr(settings, "llm_first_token_timeout", None)
            if temperature is None:
                temperature = getattr(settings, "llm_temperature", None)
        elif isinstance(providers, Mapping):
            raw_specs = [providers]
        else:
            raw_specs = list(providers)

        defaults = {
            "chat_timeout": _to_float(chat_timeout, DEFAULT_CHAT_TIMEOUT),
            "first_token_timeout": _to_float(first_token_timeout, DEFAULT_FIRST_TOKEN_TIMEOUT),
            "temperature": _to_float(temperature, 0.0),
        }
        return raw_specs, defaults

    def _unique_name(self, name: str) -> str:
        used = {provider.name for provider in self.providers}
        if name not in used:
            return name
        suffix = 2
        while f"{name}#{suffix}" in used:
            suffix += 1
        return f"{name}#{suffix}"

    def _resolve_api_key(self, data: Mapping[str, Any], name: str) -> str | None:
        env_name = str(data.get("api_key_env") or "").strip()
        if env_name:
            api_key = os.environ.get(env_name, "").strip()
            if not api_key:
                reason = f"missing api key env {env_name}"
                logger.warning("LLM provider %s skipped: %s", name, reason)
                self.skipped.append({"name": name, "reason": reason})
                return None
            return api_key
        # 本地 OpenAI 兼容服务通常不校验 key；沿用项目既有默认值。
        return str(data.get("api_key") or "ollama")

    def _normalize_extra_headers(self, raw: Any, name: str) -> dict[str, str] | None:
        """校验 provider 的 extra_headers（字符串→字符串）；非法配置跳过该 provider。

        值里的 `$UUID` 由 LLMClient 在每次请求时替换为新的 uuid4（供 opencode-go 的
        x-opencode-session 等会话头使用）。
        """
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            reason = "extra_headers must be an object"
            logger.warning("LLM provider %s skipped: %s", name, reason)
            self.skipped.append({"name": name, "reason": reason})
            return None
        headers: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str) or not key.strip():
                reason = "extra_headers must be string:string"
                logger.warning("LLM provider %s skipped: %s", name, reason)
                self.skipped.append({"name": name, "reason": reason})
                return None
            headers[key] = value
        return headers

    def _build_provider(
        self,
        spec: Mapping[str, Any] | Any,
        *,
        index: int,
        factory: Callable[..., Any],
        fail_threshold: int,
        cooldown: float,
        clock: Callable[[], float] | None,
    ) -> _Provider | None:
        if isinstance(spec, Mapping):
            data = dict(spec)
            name = str(data.get("name") or data.get("model") or f"provider-{index}")
            first_token_timeout = _to_float(data.get("first_token_timeout"), self.first_token_timeout)
            client = data.get("client")
            if client is None:
                base_url = str(data.get("base_url") or "").strip()
                model = str(data.get("model") or "").strip()
                if not base_url or not model:
                    reason = "base_url or model missing"
                    logger.warning("LLM provider %s skipped: %s", name, reason)
                    self.skipped.append({"name": name, "reason": reason})
                    return None
                api_key = self._resolve_api_key(data, name)
                if api_key is None:
                    return None
                extra_headers = self._normalize_extra_headers(data.get("extra_headers"), name)
                if extra_headers is None:
                    return None
                temperature = _to_float(data.get("temperature"), self.temperature)
                # 底层 httpx 超时不能短于首 token 预算；整请求超时由 _call_with_timeout 兜底。
                http_timeout = _to_float(data.get("timeout"), max(self.chat_timeout, first_token_timeout))
                client = factory(
                    base_url,
                    model,
                    api_key,
                    temperature,
                    timeout=http_timeout,
                    extra_headers=extra_headers,
                )
        else:
            client = spec
            name = str(getattr(client, "name", None) or getattr(client, "model", None) or f"provider-{index}")
            first_token_timeout = self.first_token_timeout

        name = self._unique_name(name)
        breaker = CircuitBreaker(fail_threshold, cooldown, name=name, clock=clock)
        return _Provider(name=name, client=client, breaker=breaker, first_token_timeout=first_token_timeout)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> FallbackLLM:
        """按 Settings 构建；等价于 `FallbackLLM(settings)`。"""
        return cls(settings, **kwargs)

    # ---------- public introspection ----------
    @property
    def clients(self) -> list[Any]:
        return [provider.client for provider in self.providers]

    @property
    def breakers(self) -> dict[str, CircuitBreaker]:
        return {provider.name: provider.breaker for provider in self.providers}

    @property
    def provider_names(self) -> list[str]:
        return [provider.name for provider in self.providers]

    def provider_status(self) -> dict[str, dict[str, Any]]:
        """供 health 使用：{name: {"state": closed|open|half_open, "failures": int}}。"""
        return {provider.name: provider.breaker.snapshot() for provider in self.providers}

    # ---------- timeout / failure helpers ----------
    @staticmethod
    def _call_with_timeout(
        fn: Callable[[], Any],
        timeout: float,
        *,
        provider: str,
        operation: str,
    ) -> Any:
        """用 daemon 线程执行同步调用；超时抛 TimeoutError，调用方继续降级。"""
        if timeout is None or timeout <= 0:
            return fn()

        result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                result.put((True, fn()))
            except BaseException as exc:  # noqa: BLE001 - 原样回传给调用线程
                result.put((False, exc))

        thread = threading.Thread(target=worker, name=f"llm-{provider}-{operation}", daemon=True)
        thread.start()
        try:
            ok, value = result.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"LLM provider {provider} {operation} timed out after {timeout:g}s"
            ) from None
        if ok:
            return value
        raise value

    def _record_failure(self, provider: _Provider, exc: BaseException) -> None:
        provider.breaker.record_failure()
        logger.warning(
            "LLM provider %s failed (%s): %s",
            provider.name,
            type(exc).__name__,
            str(exc)[:300],
        )

    def _raise_all_failed(
        self, operation: str, details: list[str], last_error: BaseException | None
    ) -> NoReturn:
        message = f"all LLM providers failed during {operation}: "
        message += "; ".join(details) if details else "no providers configured"
        error = LLMError(message)
        if last_error is not None:
            raise error from last_error
        raise error

    def _call(self, operation: str, fn: Callable[[Any], Any], timeout: float) -> Any:
        """顺序尝试 provider：熔断跳过、超时/异常切下一个；全部失败抛 LLMError。"""
        details: list[str] = []
        last_error: BaseException | None = None
        for provider in self.providers:
            if not provider.breaker.allow_request():
                details.append(f"{provider.name}: circuit open")
                continue
            try:
                result = self._call_with_timeout(
                    lambda p=provider: fn(p),
                    timeout,
                    provider=provider.name,
                    operation=operation,
                )
            except Exception as exc:  # noqa: BLE001 - 超时 / LLMError / 网络错误统一降级
                last_error = exc
                self._record_failure(provider, exc)
                details.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                continue
            provider.breaker.record_success()
            self.last_provider = provider.name
            return result
        self._raise_all_failed(operation, details, last_error)

    # ---------- LLMClient-compatible API ----------
    def chat(self, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False) -> dict:
        return self._call(
            "chat",
            lambda provider: provider.client.chat(messages, max_tokens, json_mode=json_mode),
            self.chat_timeout,
        )

    def chat_tools(self, messages: list[dict], tools: list[dict], max_tokens: int = 1024) -> dict:
        return self._call(
            "chat_tools",
            lambda provider: provider.client.chat_tools(messages, tools, max_tokens),
            self.chat_timeout,
        )

    def stream_chat(self, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        """流式生成：首 token 前可降级；首 token 后异常直接抛出，不再切换。"""
        details: list[str] = []
        last_error: BaseException | None = None
        for provider in self.providers:
            if not provider.breaker.allow_request():
                details.append(f"{provider.name}: circuit open")
                continue
            try:
                # iter() 兼容生成器 / 列表等测试替身；LLMClient 返回生成器。
                iterator = iter(provider.client.stream_chat(messages, max_tokens))
            except Exception as exc:  # noqa: BLE001 - provider 初始化/连接错误均降级
                last_error = exc
                self._record_failure(provider, exc)
                details.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                continue

            try:
                first = self._call_with_timeout(
                    lambda it=iterator: next(it),
                    provider.first_token_timeout,
                    provider=provider.name,
                    operation="stream first token",
                )
            except StopIteration:
                provider.breaker.record_success()
                self.last_provider = provider.name
                return
            except Exception as exc:  # noqa: BLE001 - 首 token 前的任何异常都可降级
                last_error = exc
                self._record_failure(provider, exc)
                details.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                continue

            # 首 token 已到达：此后不再切换，流中途异常直接抛给调用方。
            provider.breaker.record_success()
            self.last_provider = provider.name
            yield first
            try:
                yield from iterator
            except Exception as exc:
                self._record_failure(provider, exc)
                raise
            return

        self._raise_all_failed("stream_chat", details, last_error)


def build_llm(settings: Settings | None = None) -> FallbackLLM:
    """按 `settings.resolved_llm_providers` 构建带降级的 LLM 实例。"""
    return FallbackLLM.from_settings(settings if settings is not None else get_settings())
