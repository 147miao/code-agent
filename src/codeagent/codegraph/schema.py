"""Neo4j 图 schema：约束、索引与仓库级清理。

节点统一带 repo_id + kind + qualname 三元组，约束保证同一 repo 内同一标签的键唯一。
Symbol 的 qualname 采用 `path::qualname`（路径限定键），因为不同文件允许同名函数
（例如多个 `main`），而计划冻结的 (repo_id, kind, qualname) 必须是全局唯一的；
原始 Python qualname 存在 `symbol_qualname` 属性，impact 查询两者都接受。
File 用 path、Module 用模块名、Name 用引用名作为 qualname。
"""
from __future__ import annotations

from typing import Any

NODE_LABELS: tuple[str, ...] = ("File", "Module", "Symbol", "Name")
EDGE_TYPES: tuple[str, ...] = ("CONTAINS", "IMPORTS", "REFERENCES", "CALLS", "TESTS")

CONSTRAINTS: tuple[str, ...] = tuple(
    f"CREATE CONSTRAINT codegraph_{label.lower()}_key IF NOT EXISTS "
    f"FOR (n:{label}) REQUIRE (n.repo_id, n.kind, n.qualname) IS UNIQUE"
    for label in NODE_LABELS
)

INDEXES: tuple[str, ...] = (
    "CREATE INDEX codegraph_symbol_name IF NOT EXISTS FOR (n:Symbol) ON (n.name)",
    "CREATE INDEX codegraph_symbol_path IF NOT EXISTS FOR (n:Symbol) ON (n.repo_id, n.path)",
    "CREATE INDEX codegraph_symbol_lookup IF NOT EXISTS FOR (n:Symbol) ON (n.repo_id, n.symbol_qualname)",
    "CREATE INDEX codegraph_file_path IF NOT EXISTS FOR (n:File) ON (n.repo_id, n.path)",
    "CREATE INDEX codegraph_module_name IF NOT EXISTS FOR (n:Module) ON (n.repo_id, n.name)",
    "CREATE INDEX codegraph_name_name IF NOT EXISTS FOR (n:Name) ON (n.repo_id, n.name)",
    # 关系索引让 CALLS/TESTS 即使暂时没有边也存在于 schema，避免 impact 查询告警。
    "CREATE INDEX codegraph_calls_repo IF NOT EXISTS FOR ()-[r:CALLS]-() ON (r.repo_id)",
    "CREATE INDEX codegraph_tests_repo IF NOT EXISTS FOR ()-[r:TESTS]-() ON (r.repo_id)",
)

# 逆序删除（先删边随 DETACH DELETE 一起清理）；按 repo_id 前缀命中唯一约束索引。
DELETE_QUERIES: tuple[tuple[str, str], ...] = tuple(
    (label, f"MATCH (n:{label} {{repo_id: $repo_id}}) DETACH DELETE n") for label in NODE_LABELS
)


def ensure_schema(client: Any) -> dict[str, int]:
    """幂等创建约束/索引；返回执行条数（供 build 结果展示）。"""
    for statement in CONSTRAINTS:
        client.execute(statement)
    for statement in INDEXES:
        client.execute(statement)
    return {"constraints": len(CONSTRAINTS), "indexes": len(INDEXES)}


def delete_repo_graph(client: Any, repo_id: str) -> dict[str, int]:
    """删除该 repo 的全部图节点与边（DETACH DELETE），返回删除前各标签计数。"""
    before = graph_counts(client, repo_id)
    for _label, query in DELETE_QUERIES:
        client.execute(query, {"repo_id": repo_id})
    return before


def graph_counts(client: Any, repo_id: str | None = None) -> dict[str, Any]:
    """统计节点/边；repo_id 为空时统计全库。缺失类型补 0，保证响应字段稳定。"""
    node_rows = client.execute(
        "MATCH (n) WHERE ($repo_id IS NULL OR n.repo_id = $repo_id) "
        "AND (n:File OR n:Module OR n:Symbol OR n:Name) "
        "RETURN labels(n)[0] AS label, count(*) AS total",
        {"repo_id": repo_id},
    )
    edge_rows = client.execute(
        "MATCH ()-[r]->() WHERE ($repo_id IS NULL OR r.repo_id = $repo_id) "
        "RETURN type(r) AS type, count(*) AS total",
        {"repo_id": repo_id},
    )
    nodes = {label: 0 for label in NODE_LABELS}
    edges = {edge_type: 0 for edge_type in EDGE_TYPES}
    for row in node_rows:
        label = str(row.get("label") or "")
        if label in nodes:
            nodes[label] = int(row.get("total") or 0)
    for row in edge_rows:
        edge_type = str(row.get("type") or "")
        if edge_type in edges:
            edges[edge_type] = int(row.get("total") or 0)
    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": sum(nodes.values()),
        "total_edges": sum(edges.values()),
    }


__all__ = [
    "CONSTRAINTS",
    "DELETE_QUERIES",
    "EDGE_TYPES",
    "INDEXES",
    "NODE_LABELS",
    "delete_repo_graph",
    "ensure_schema",
    "graph_counts",
]
