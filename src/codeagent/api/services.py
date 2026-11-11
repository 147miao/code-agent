"""应用依赖装配：AppServices + build_services（依赖失败时置 None，端点返回 503）。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agent.loop import AgentLimits, CodeAgent
from ..codegraph import GraphService
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
    # Neo4j 代码图服务：构造时只保存配置与 store，首次使用才建连（lazy）。
    graph: Any | None = None
    # 每次 ask 用当前仓库的 AgentTools 构建一个 CodeAgent（预算来自 settings）
    agent_factory: Callable[[Any], Any] | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def availability(self) -> dict[str, bool]:
        deps = {
            "store": self.store is not None,
            "indexer": self.indexer is not None,
            "service": self.service is not None,
            "sessions": self.sessions is not None,
            "llm": self.llm is not None,
            "agent_factory": self.agent_factory is not None,
        }
        # 只在已配置时暴露 graph，避免未配置 Neo4j 时 health 变 degraded。
        if self.graph is not None and getattr(self.graph, "configured", False):
            deps["graph"] = True
        return deps


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
            # 不连接 Neo4j：只装配配置/store/缓存路径，连接在首次 build/status 时发生。
            services.graph = GraphService(settings, services.store)
        except Exception as exc:  # noqa: BLE001
            services.errors["graph"] = f"{type(exc).__name__}: {exc}"
            logger.exception("graph service unavailable")
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
    if services.graph is not None:
        # LLM 可能初始化失败；缺失时 /api/graph/build?with_llm=true 返回 503。
        services.graph.llm = services.llm
    return services


__all__ = ["AppServices", "build_agent_factory", "build_services"]
