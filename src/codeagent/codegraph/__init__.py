"""R2 代码图：AST 边 + LLM 候选边 + impact 图查询（Neo4j，回退 SQLite）。

对外入口：
- build_graph(service, repo_id, with_llm=False)：重建 repo 图，返回节点/边统计；
- impact(service, repo_id, path=.../qualname=..., depth=1)：反向图查询；
- status(service, repo_id=None)：图节点/边与 LLM 缓存统计。

GraphService 惰性连接 Neo4j：配置缺失时 build/status 抛 GraphUnavailableError（API 503），
impact 自动回退 CodeStore.reverse_imports 并标记 fallback="sqlite"。
"""
from __future__ import annotations

from .builder import build_graph_data, write_graph
from .client import (
    GraphError,
    GraphUnavailableError,
    GraphValidationError,
    Neo4jClient,
    Neo4jConfig,
)
from .impact import fallback_impact, graph_impact
from .llm_edges import LlmEdgeCache, LlmEdgeConfig, extract_llm_edges
from .schema import ensure_schema, graph_counts
from .service import GraphService, build_graph, build_graph_service, impact, status

__all__ = [
    "GraphError",
    "GraphService",
    "GraphUnavailableError",
    "GraphValidationError",
    "LlmEdgeCache",
    "LlmEdgeConfig",
    "Neo4jClient",
    "Neo4jConfig",
    "build_graph",
    "build_graph_data",
    "build_graph_service",
    "ensure_schema",
    "extract_llm_edges",
    "fallback_impact",
    "graph_counts",
    "graph_impact",
    "impact",
    "status",
    "write_graph",
]
