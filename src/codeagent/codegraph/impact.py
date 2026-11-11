"""impact 图查询与 SQLite 回退。

优先在 Neo4j 上按反向 CALLS/TESTS/REFERENCES/IMPORTS 遍历（depth 1-2），返回命中节点
与每条边的证据；Neo4j 未配置/连接失败/查询失败时回退到 CodeStore.reverse_imports，
结果标记 fallback="sqlite"，保持旧 API 的 imports/heuristic/note 字段。
"""
from __future__ import annotations

import logging
from typing import Any

from .builder import normalize_path
from .client import GraphValidationError
from .schema import graph_counts

logger = logging.getLogger(__name__)

IMPACT_NOTE = "文件级反向 import（启发式，非精确调用链）"
GRAPH_IMPACT_NOTE = "Neo4j 反向图查询（CALLS/TESTS/REFERENCES/IMPORTS，depth 1-2）"
DEFAULT_DEPTH = 1
MAX_DEPTH = 2
MAX_IMPACT_LIMIT = 200
IMPACT_RELATIONS = ("CALLS", "TESTS", "REFERENCES", "IMPORTS")


def resolve_query_path(store: Any, repo_id: str, path: str | None, qualname: str | None) -> str:
    """path 优先；只给 qualname 时从 SQLite 反查定义文件（用于回退与 IMPORTS 统计）。"""
    if path:
        return normalize_path(path)
    if not qualname:
        return ""
    symbol = store.get_symbol(repo_id, str(qualname))
    if not symbol:
        # 兼容调用方传入 path::qualname 的图键。
        plain = str(qualname).split("::", 1)
        if len(plain) == 2:
            return normalize_path(plain[0])
        return ""
    return normalize_path(str(symbol.get("path") or ""))


def fallback_impact(
    store: Any,
    repo_id: str,
    *,
    path: str | None = None,
    qualname: str | None = None,
    depth: int = DEFAULT_DEPTH,
    error: str = "",
) -> dict[str, Any]:
    """SQLite 回退：只做文件级反向 import，不假装有图结果。"""
    resolved_path = resolve_query_path(store, repo_id, path, qualname)
    imports = store.reverse_imports(repo_id, resolved_path) if resolved_path else []
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "path": resolved_path,
        "qualname": qualname or "",
        "depth": int(depth),
        "imports": imports,
        "nodes": [],
        "edges": [],
        "seeds": 0,
        "fallback": "sqlite",
        "source": "sqlite",
        "heuristic": True,
        "note": IMPACT_NOTE,
    }
    if error:
        result["graph_error"] = error
    return result


def _node_summary(node: Any) -> dict[str, Any]:
    props = dict(node)
    labels = list(getattr(node, "labels", []) or [])
    return {
        "label": labels[0] if labels else "",
        "key": props.get("qualname") or "",
        "qualname": props.get("symbol_qualname") or props.get("qualname") or "",
        "kind": props.get("kind") or "",
        "path": props.get("path") or "",
        "name": props.get("name") or "",
        "start_line": int(props.get("start_line") or 0),
        "end_line": int(props.get("end_line") or 0),
    }


def _relationship_summary(rel: Any) -> dict[str, Any]:
    props = dict(rel)
    start, end = rel.start_node, rel.end_node
    return {
        "type": rel.type,
        "edge_key": props.get("edge_key") or "",
        "source": props.get("source") or "",
        "confidence": float(props.get("confidence") or 0.0),
        "lineno": int(props.get("lineno") or props.get("evidence_line") or 0),
        "module": props.get("module") or "",
        "from": str(dict(start).get("qualname") or ""),
        "to": str(dict(end).get("qualname") or ""),
    }


def graph_impact(
    client: Any,
    repo_id: str,
    *,
    path: str | None = None,
    qualname: str | None = None,
    depth: int = DEFAULT_DEPTH,
    limit: int = MAX_IMPACT_LIMIT,
) -> dict[str, Any]:
    """Neo4j 反向遍历；查询失败抛 GraphUnavailableError，由调用方决定是否回退。"""
    depth = int(depth)
    if depth < 1 or depth > MAX_DEPTH:
        raise GraphValidationError(f"depth 必须在 1..{MAX_DEPTH} 之间")
    if not path and not qualname:
        raise GraphValidationError("path 与 qualname 至少提供一个")
    resolved_path = normalize_path(path) if path else None
    normalized_qualname = str(qualname).strip() if qualname else None
    limit = max(1, min(int(limit), MAX_IMPACT_LIMIT))
    # depth 已做 1..2 校验，可直接拼进变长路径（Cypher 不支持参数化跳数）。
    query = (
        "MATCH (t) "
        "WHERE t.repo_id = $repo_id AND ( "
        "  ($path IS NOT NULL AND t.path = $path AND (t:File OR t:Symbol)) "
        "  OR ($qualname IS NOT NULL AND t:Symbol AND "
        "      (t.qualname = $qualname OR t.symbol_qualname = $qualname)) "
        ") "
        f"MATCH p = (src)-[:CALLS|TESTS|REFERENCES|IMPORTS*1..{depth}]->(t) "
        "RETURN p LIMIT $limit"
    )
    records = client.records(
        query,
        {
            "repo_id": repo_id,
            "path": resolved_path,
            "qualname": normalized_qualname,
            "limit": limit,
        },
    )
    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    imports: set[str] = set()
    for record in records:
        path_value = record["p"]
        for node in path_value.nodes:
            summary = _node_summary(node)
            nodes[(summary["label"], str(summary["key"]))] = summary
        for rel in path_value.relationships:
            summary = _relationship_summary(rel)
            edges[(summary["type"], summary["edge_key"])] = summary
            if (
                summary["type"] == "IMPORTS"
                and resolved_path
                and summary["to"] == resolved_path
                and summary["from"]
            ):
                imports.add(summary["from"])
    return {
        "repo_id": repo_id,
        "path": resolved_path or "",
        "qualname": normalized_qualname or "",
        "depth": depth,
        "imports": sorted(imports),
        "nodes": sorted(nodes.values(), key=lambda item: (item["label"], item["key"])),
        "edges": sorted(edges.values(), key=lambda item: (item["type"], item["edge_key"])),
        "seeds": len(nodes),
        "fallback": None,
        "source": "graph",
        "heuristic": False,
        "note": GRAPH_IMPACT_NOTE,
    }


def graph_status(client: Any, repo_id: str | None = None) -> dict[str, Any]:
    """图连接可用时的统计（节点/边按类型）。"""
    counts = graph_counts(client, repo_id)
    return {
        "enabled": True,
        "connected": True,
        "repo_id": repo_id or "",
        **counts,
    }


__all__ = [
    "DEFAULT_DEPTH",
    "GRAPH_IMPACT_NOTE",
    "IMPACT_NOTE",
    "MAX_DEPTH",
    "fallback_impact",
    "graph_impact",
    "graph_status",
    "resolve_query_path",
]
