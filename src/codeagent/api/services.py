"""应用依赖装配：AppServices + build_services（依赖失败时置 None，端点返回 503）。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agent.loop import AgentLimits, CodeAgent
from ..config import Settings
from ..index.indexer import CodeIndexer
from ..index.store import CodeStore
from ..llm import build_llm
from ..retrieval.service import CodeService
from ..sessions import build_session_repository

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    """进程级依赖；None 表示该依赖初始化失败（对应端点 503）。"""

    settings: Settings
    store: Any | None = None
    indexer: Any | None = None
    service: Any | None = None
    sessions: Any | None = None
    llm: Any | None = None
    # 每次 ask 用当前仓库的 AgentTools 构建一个 CodeAgent（预算来自 settings）
    agent_factory: Callable[[Any], Any] | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def availability(self) -> dict[str, bool]:
        return {
            "store": self.store is not None,
            "indexer": self.indexer is not None,
            "service": self.service is not None,
            "sessions": self.sessions is not None,
            "llm": self.llm is not None,
            "agent_factory": self.agent_factory is not None,
        }


def build_agent_factory(settings: Settings, llm: Any) -> Callable[[Any], CodeAgent]:
    """把 Agent 装配收拢到 lifespan：调用方只负责传入绑定好仓库的 AgentTools。"""
    limits = AgentLimits.from_settings(settings)

    def factory(tools: Any) -> CodeAgent:
        return CodeAgent(tools, llm, limits)

    return factory


def build_services(settings: Settings) -> AppServices:
    """按配置构建依赖；单个依赖失败只记录，不影响其它端点。"""
    services = AppServices(settings=settings)
    try:
        services.store = CodeStore(settings.code_store_path)
    except Exception as exc:  # noqa: BLE001 - 启动期依赖失败降级为 503
        services.errors["store"] = f"{type(exc).__name__}: {exc}"
        logger.exception("code store unavailable: %s", settings.code_store_path)
    if services.store is not None:
        services.service = CodeService(services.store)
        try:
            services.indexer = CodeIndexer(settings.code_store_path, settings.code_allowed_roots)
        except Exception as exc:  # noqa: BLE001
            services.errors["indexer"] = f"{type(exc).__name__}: {exc}"
            logger.exception("code indexer unavailable: %s", settings.code_store_path)
    try:
        services.sessions = build_session_repository(settings)
    except Exception as exc:  # noqa: BLE001
        services.errors["sessions"] = f"{type(exc).__name__}: {exc}"
        logger.exception("session repository unavailable")
    try:
        services.llm = build_llm(settings)
        services.agent_factory = build_agent_factory(settings, services.llm)
    except Exception as exc:  # noqa: BLE001
        services.errors["llm"] = f"{type(exc).__name__}: {exc}"
        logger.exception("llm unavailable")
    return services


__all__ = ["AppServices", "build_agent_factory", "build_services"]
