"""代码图服务：惰性 Neo4j、构建编排、impact 回退与状态。

GraphService 在 API lifespan 构造时只保存配置与 store，不建立连接；首次 build/status
才连接。impact 在 Neo4j 不可用时自动回退 SQLite，不影响检索/答案端点。
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from ..config import get_settings
from .builder import build_graph_data, load_graph_inputs, merge_llm_edges, write_graph
from .client import Neo4jClient, Neo4jConfig
from .impact import DEFAULT_DEPTH, fallback_impact, graph_impact, graph_status
from .llm_edges import LlmEdgeCache, LlmEdgeConfig, extract_llm_edges

logger = logging.getLogger(__name__)


class GraphService:
    """按 settings + CodeStore 组装的图服务；延迟连接、构建串行化。"""

    def __init__(
        self,
        settings: Any,
        store: Any,
        *,
        llm: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.config = Neo4jConfig.from_settings(settings)
        self.client = client or Neo4jClient(self.config)
        self._build_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(getattr(self.client, "configured", False))

    def describe(self) -> dict[str, Any]:
        return {
            **self.config.describe(),
            "llm": self.llm is not None,
        }

    def build(
        self,
        repo_id: str,
        with_llm: bool = False,
        *,
        llm_paths: list[str] | None = None,
        llm_config: LlmEdgeConfig | None = None,
    ) -> dict[str, Any]:
        """重建 repo 图；with_llm=True 时追加 CALLS/TESTS 候选边。"""
        from .client import GraphUnavailableError

        if not self.configured:
            raise GraphUnavailableError("Neo4j 未配置（NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD）")
        with self._build_lock:
            repo, data = load_graph_inputs(self.store, repo_id)
            graph = build_graph_data(repo, data)
            llm_result: dict[str, Any] | None = None
            if with_llm:
                if self.llm is None:
                    raise GraphUnavailableError("LLM 未配置，无法生成候选边")
                config = (llm_config or LlmEdgeConfig.from_settings(self.settings)).with_paths(llm_paths)
                llm_result = extract_llm_edges(
                    self.store,
                    repo_id,
                    self.llm,
                    config=config,
                    data=data,
                    repo=repo,
                )
                merge_llm_edges(graph, llm_result)
            report = write_graph(self.client, graph)
        report["config"] = self.config.describe()
        report["llm"] = (llm_result or {}).get("stats")
        report["llm_skipped"] = (llm_result or {}).get("skipped") or []
        return report

    def delete(self, repo_id: str) -> dict[str, Any]:
        """删除该 repo 的全部图节点/边；用于 API 删除/reindex 同步。"""
        from .client import GraphUnavailableError
        from .schema import delete_repo_graph

        if not self.configured:
            raise GraphUnavailableError("Neo4j 未配置（NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD）")
        with self._build_lock:
            return delete_repo_graph(self.client, repo_id)

    def impact(
        self,
        repo_id: str,
        *,
        path: str | None = None,
        qualname: str | None = None,
        depth: int = DEFAULT_DEPTH,
    ) -> dict[str, Any]:
        """优先图查询；未配置/连接失败回退 SQLite（fallback="sqlite"）。"""
        from .client import GraphUnavailableError, GraphValidationError

        if not path and not qualname:
            raise GraphValidationError("path 与 qualname 至少提供一个")
        if self.store.get_repo(repo_id) is None:
            raise GraphValidationError(f"repo not found: {repo_id}")
        if not self.configured:
            return fallback_impact(self.store, repo_id, path=path, qualname=qualname, depth=depth)
        try:
            return graph_impact(self.client, repo_id, path=path, qualname=qualname, depth=depth)
        except GraphUnavailableError as exc:
            logger.warning("graph impact unavailable, fallback to sqlite: %s", exc)
            return fallback_impact(
                self.store,
                repo_id,
                path=path,
                qualname=qualname,
                depth=depth,
                error=str(exc),
            )

    def status(self, repo_id: str | None = None) -> dict[str, Any]:
        """图统计；Neo4j 未配置/连接失败抛 GraphUnavailableError（API 映射 503）。"""
        from .client import GraphUnavailableError

        if not self.configured:
            raise GraphUnavailableError("Neo4j 未配置（NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD）")
        result = graph_status(self.client, repo_id)
        result["config"] = self.config.describe()
        result["llm"] = self.llm is not None
        result["cache"] = LlmEdgeCache(self.store.db_path).stats(repo_id)
        return result


def build_graph(
    service: GraphService,
    repo_id: str,
    *,
    with_llm: bool = False,
    llm_paths: list[str] | None = None,
    llm_config: LlmEdgeConfig | None = None,
) -> dict[str, Any]:
    """模块级门面：GraphService.build 的薄包装。"""
    return service.build(repo_id, with_llm, llm_paths=llm_paths, llm_config=llm_config)


def impact(
    service: GraphService,
    repo_id: str,
    *,
    path: str | None = None,
    qualname: str | None = None,
    depth: int = DEFAULT_DEPTH,
) -> dict[str, Any]:
    """模块级门面：GraphService.impact。"""
    return service.impact(repo_id, path=path, qualname=qualname, depth=depth)


def status(service: GraphService, repo_id: str | None = None) -> dict[str, Any]:
    """模块级门面：GraphService.status。"""
    return service.status(repo_id)


def build_graph_service(settings: Any | None = None, store: Any = None, *, llm: Any | None = None) -> GraphService:
    """按 Settings 构建服务（供脚本/测试直接使用；API 复用 AppServices.graph）。"""
    return GraphService(settings or get_settings(), store, llm=llm)


__all__ = [
    "GraphService",
    "build_graph",
    "build_graph_service",
    "impact",
    "status",
]
