"""从 CodeStore 的 AST 索引生成 Neo4j 图数据（纯函数）并批量写入。

边生成规则：
- CONTAINS：File -> Symbol（symbols 表每一行，含 <module>）；
- IMPORTS：File -> File（import 能解析到本仓库文件）或 File -> Module（外部/无法解析）；
- REFERENCES：Symbol -> Symbol（名字能解析）或 Symbol -> Name（解析不到），按最小包含符号归属；
- CALLS/TESTS：由 llm_edges 生成，merge_llm_edges 合并进同一份图数据。

Symbol 节点键为 `path::qualname`：不同文件允许同名函数，而 (repo_id,kind,qualname)
需要全局唯一；原始 qualname 保留在 `symbol_qualname`，impact 同时支持两种写法。
所有写库语句使用 UNWIND 批处理；重建前按 repo_id 删除旧节点/边。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from time import perf_counter
from typing import Any

from .schema import EDGE_TYPES, NODE_LABELS, delete_repo_graph, ensure_schema, graph_counts

MODULE_QUALNAME = "<module>"
DEFAULT_BATCH_SIZE = 500
DOCSTRING_MAX_CHARS = 500
UNKNOWN_MODULE = "<unknown>"
CALLS_EDGES = ("CALLS", "TESTS")
_WINDOWS_SEP = re.compile(r"\\")

GraphData = dict[str, Any]


def normalize_path(path: str) -> str:
    return _WINDOWS_SEP.sub("/", str(path or "")).lstrip("./")


def symbol_key(path: str, qualname: str) -> str:
    """Symbol 节点的全局唯一键：路径 + 原始 qualname。"""
    return f"{normalize_path(path)}::{qualname}"


def symbol_name(qualname: str) -> str:
    if qualname == MODULE_QUALNAME:
        return MODULE_QUALNAME
    return str(qualname).rsplit(".", 1)[-1]


def module_candidates(path: str) -> list[str]:
    """由仓库相对路径推断所有可能的模块名后缀（与 CodeStore.reverse_imports 一致）。"""
    normalized = normalize_path(path)
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[index:]) for index in range(len(parts))]


def build_module_index(files: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """模块名 -> 文件路径；同名模块保留路径字典序最小者（相对导入不受影响）。"""
    index: dict[str, str] = {}
    for item in sorted(files, key=lambda row: str(row.get("path") or "")):
        path = normalize_path(str(item.get("path") or ""))
        if not path:
            continue
        for candidate in module_candidates(path):
            index.setdefault(candidate, path)
    return index


def resolve_import(
    current_path: str,
    module: str,
    level: int,
    module_index: Mapping[str, str],
) -> str | None:
    """把 import 解析为仓库内文件；解析不到返回 None。

    相对导入 level>=1：从当前模块所在包向上回退 (level-1) 层，再拼模块名；
    绝对导入直接查 module_index（含任意层级后缀）。
    """
    path = normalize_path(current_path)
    module = str(module or "").strip()
    try:
        level = int(level or 0)
    except (TypeError, ValueError):
        level = 0
    if level > 0:
        parts = [part for part in path.split("/") if part]
        # level=1 表示当前模块所在包；level=2 再向上一层。
        package = parts[:-1]
        drop = max(0, level - 1)
        if drop:
            package = package[:-drop] if drop <= len(package) else []
        joined = ".".join([*package, *([part for part in module.split(".") if part] if module else [])])
        return module_index.get(joined) if joined else None
    return module_index.get(module) if module else None


def display_module(module: str, level: int) -> str:
    prefix = "." * max(0, int(level or 0))
    return f"{prefix}{module}" if (prefix or module) else UNKNOWN_MODULE


def _clean_props(props: Mapping[str, Any]) -> dict[str, Any]:
    """Neo4j 不接受嵌套 null 赋值；去掉 None 值即可（SET += 也不会清空旧属性）。"""
    return {str(key): value for key, value in props.items() if value is not None}


def _node_row(repo_id: str, label: str, kind: str, qualname: str, props: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "repo_id": repo_id,
        "kind": kind,
        "qualname": qualname,
        "props": _clean_props(props),
    }


def _edge_row(
    repo_id: str,
    edge_type: str,
    from_label: str,
    from_kind: str,
    from_qualname: str,
    to_label: str,
    to_kind: str,
    to_qualname: str,
    edge_key: str,
    props: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": edge_type,
        "repo_id": repo_id,
        "from_label": from_label,
        "from_kind": from_kind,
        "from_qualname": from_qualname,
        "to_label": to_label,
        "to_kind": to_kind,
        "to_qualname": to_qualname,
        "edge_key": edge_key,
        "props": _clean_props(props),
    }


def _empty_graph(repo_id: str) -> GraphData:
    return {
        "repo_id": repo_id,
        "nodes": {label: [] for label in NODE_LABELS},
        "edges": {edge_type: [] for edge_type in EDGE_TYPES},
        "stats": {
            "files": 0,
            "symbols": 0,
            "imports": 0,
            "references": 0,
            "resolved_imports": 0,
            "unresolved_imports": 0,
            "resolved_references": 0,
            "unresolved_references": 0,
        },
    }


def build_graph_data(
    repo: Mapping[str, Any],
    data: Mapping[str, Sequence[Mapping[str, Any]]],
) -> GraphData:
    """把 CodeStore.export_repo 的结果转成图数据；不连接 Neo4j，可独立单测。"""
    repo_id = str(repo.get("repo_id") or repo.get("id") or "")
    graph = _empty_graph(repo_id)
    files = [item for item in data.get("files") or [] if item.get("path")]
    symbols = [item for item in data.get("symbols") or [] if item.get("path") and item.get("qualname")]
    imports = [item for item in data.get("imports") or [] if item.get("src_path")]
    references = [item for item in data.get("references") or [] if item.get("src_path") and item.get("name")]
    chunks = list(data.get("chunks") or [])

    module_index = build_module_index(files)
    files_by_path = {normalize_path(str(item["path"])): item for item in files}
    chunks_by_path: dict[str, list[Mapping[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_path.setdefault(normalize_path(str(chunk.get("path") or "")), []).append(chunk)

    # ---- File nodes ----
    symbols_by_path: dict[str, list[dict[str, Any]]] = {}
    seen_symbol_keys: set[str] = set()
    for item in symbols:
        path = normalize_path(str(item.get("path") or ""))
        qualname = str(item.get("qualname") or "")
        key = symbol_key(path, qualname)
        if key in seen_symbol_keys:
            continue
        seen_symbol_keys.add(key)
        row = {
            "path": path,
            "qualname": qualname,
            "kind": str(item.get("kind") or ""),
            "start_line": int(item.get("start_line") or 1),
            "end_line": int(item.get("end_line") or 1),
            "code_hash": str(item.get("code_hash") or ""),
            "docstring": item.get("docstring"),
        }
        symbols_by_path.setdefault(path, []).append(row)
    for symbols_here in symbols_by_path.values():
        symbols_here.sort(key=lambda row: (row["start_line"], row["qualname"]))

    for item in files:
        path = normalize_path(str(item["path"]))
        file_symbols = symbols_by_path.get(path, [])
        file_chunks = chunks_by_path.get(path, [])
        loc = max([int(chunk.get("end_line") or 0) for chunk in file_chunks] or [0])
        if file_symbols:
            loc = max(loc, max(int(symbol["end_line"] or 0) for symbol in file_symbols))
        graph["nodes"]["File"].append(
            _node_row(
                repo_id,
                "File",
                "file",
                path,
                {
                    "path": path,
                    "name": path.rsplit("/", 1)[-1],
                    "file_hash": str(item.get("file_hash") or ""),
                    "size": int(item.get("size") or 0),
                    "mtime": float(item.get("mtime") or 0.0),
                    "loc": loc,
                    "chunk_count": len(file_chunks),
                    "symbol_count": len(file_symbols),
                    "repo_name": str(repo.get("name") or ""),
                },
            )
        )
        for symbol in file_symbols:
            key = symbol_key(path, symbol["qualname"])
            graph["nodes"]["Symbol"].append(
                _node_row(
                    repo_id,
                    "Symbol",
                    symbol["kind"],
                    key,
                    {
                        "path": path,
                        "symbol_qualname": symbol["qualname"],
                        "name": symbol_name(symbol["qualname"]),
                        "start_line": symbol["start_line"],
                        "end_line": symbol["end_line"],
                        "docstring": (symbol["docstring"] or "")[:DOCSTRING_MAX_CHARS] or None,
                        "code_hash": symbol["code_hash"],
                    },
                )
            )
            graph["edges"]["CONTAINS"].append(
                _edge_row(
                    repo_id,
                    "CONTAINS",
                    "File",
                    "file",
                    path,
                    "Symbol",
                    symbol["kind"],
                    key,
                    f"CONTAINS:{path}->{key}",
                    {
                        "path": path,
                        "symbol_kind": symbol["kind"],
                        "start_line": symbol["start_line"],
                        "end_line": symbol["end_line"],
                    },
                )
            )

    # ---- IMPORTS ----
    edge_keys: dict[str, set[str]] = {edge_type: set() for edge_type in EDGE_TYPES}
    for list_name, edges in graph["edges"].items():
        edge_keys[list_name] = {str(edge["edge_key"]) for edge in edges}

    imported_targets: dict[str, set[str]] = {}
    for item in imports:
        src_path = normalize_path(str(item.get("src_path") or ""))
        if src_path not in files_by_path:
            continue
        module = str(item.get("module") or "")
        level = int(item.get("level") or 0)
        lineno = int(item.get("lineno") or 0)
        target = resolve_import(src_path, module, level, module_index)
        display = display_module(module, level)
        if target is not None and target != src_path:
            imported_targets.setdefault(src_path, set()).add(target)
            to_label, to_kind, to_qualname = "File", "file", target
            edge_key = f"IMPORTS:{src_path}->{target}@{lineno}"
        else:
            to_label, to_kind, to_qualname = "Module", "module", display
            edge_key = f"IMPORTS:{src_path}->{display}@{lineno}"
        if edge_key in edge_keys["IMPORTS"]:
            continue
        edge_keys["IMPORTS"].add(edge_key)
        graph["edges"]["IMPORTS"].append(
            _edge_row(
                repo_id,
                "IMPORTS",
                "File",
                "file",
                src_path,
                to_label,
                to_kind,
                to_qualname,
                edge_key,
                {
                    "module": module,
                    "level": level,
                    "lineno": lineno,
                    "target": target or display,
                    "resolved": bool(target and target != src_path),
                    "source": "ast",
                    "confidence": 1.0,
                },
            )
        )
        if to_label == "Module":
            graph["nodes"]["Module"].append(
                _node_row(repo_id, "Module", "module", display, {"name": display, "module": display})
            )

    # ---- REFERENCES ----
    symbols_by_name: dict[str, list[dict[str, Any]]] = {}
    for items in symbols_by_path.values():
        for symbol in items:
            if symbol["kind"] == "module":
                continue
            symbols_by_name.setdefault(symbol_name(symbol["qualname"]), []).append(symbol)
    for items in symbols_by_name.values():
        items.sort(key=lambda row: (row["path"], row["start_line"], row["qualname"]))

    def resolve_reference(name: str, path: str, lineno: int) -> dict[str, Any] | None:
        local = [
            row
            for row in symbols_by_path.get(path, [])
            if row["kind"] != "module" and symbol_name(row["qualname"]) == name
        ]
        if local:
            return _best_symbol(local, lineno)
        imported: list[dict[str, Any]] = []
        for target_path in sorted(imported_targets.get(path, ())):
            imported.extend(
                row
                for row in symbols_by_path.get(target_path, [])
                if row["kind"] != "module" and symbol_name(row["qualname"]) == name
            )
        if imported:
            return _best_symbol(imported, lineno)
        global_matches = symbols_by_name.get(name) or []
        if len(global_matches) == 1:
            return global_matches[0]
        return None

    for item in references:
        src_path = normalize_path(str(item.get("src_path") or ""))
        if src_path not in files_by_path:
            continue
        name = str(item.get("name") or "")
        lineno = int(item.get("lineno") or 0)
        source = _enclosing_symbol(symbols_by_path.get(src_path, []), lineno)
        if source is None:
            continue
        target = resolve_reference(name, src_path, lineno)
        source_key = symbol_key(src_path, source["qualname"])
        if target is not None:
            to_label, to_kind = "Symbol", target["kind"]
            to_qualname = symbol_key(target["path"], target["qualname"])
            resolved = True
            graph["stats"]["resolved_references"] += 1
            edge_key = f"REFERENCES:{source_key}->{name}@{lineno}->{to_qualname}"
        else:
            to_label, to_kind, to_qualname = "Name", "name", name
            resolved = False
            graph["stats"]["unresolved_references"] += 1
            edge_key = f"REFERENCES:{source_key}->{name}@{lineno}"
            graph["nodes"]["Name"].append(_node_row(repo_id, "Name", "name", name, {"name": name}))
        if edge_key in edge_keys["REFERENCES"]:
            continue
        edge_keys["REFERENCES"].add(edge_key)
        graph["edges"]["REFERENCES"].append(
            _edge_row(
                repo_id,
                "REFERENCES",
                "Symbol",
                source["kind"],
                source_key,
                to_label,
                to_kind,
                to_qualname,
                edge_key,
                {
                    "name": name,
                    "lineno": lineno,
                    "resolved": resolved,
                    "source": "ast",
                    "confidence": 1.0,
                },
            )
        )

    graph["stats"].update(
        {
            "files": len(files),
            "symbols": len(graph["nodes"]["Symbol"]),
            "imports": len(graph["edges"]["IMPORTS"]),
            "references": len(graph["edges"]["REFERENCES"]),
            "resolved_imports": sum(1 for edge in graph["edges"]["IMPORTS"] if edge["props"].get("resolved")),
            "unresolved_imports": sum(1 for edge in graph["edges"]["IMPORTS"] if not edge["props"].get("resolved")),
        }
    )
    graph["nodes"]["Module"] = _dedupe_nodes(graph["nodes"]["Module"])
    graph["nodes"]["Name"] = _dedupe_nodes(graph["nodes"]["Name"])
    return graph


def _best_symbol(items: Sequence[Mapping[str, Any]], lineno: int) -> Mapping[str, Any]:
    """同名多定义时优先选择行号之前最近的；否则取最靠前的定义。"""
    preceding = [row for row in items if int(row.get("start_line") or 0) <= lineno]
    if preceding:
        return max(preceding, key=lambda row: (int(row.get("start_line") or 0), str(row.get("path"))))
    return min(items, key=lambda row: (int(row.get("start_line") or 0), str(row.get("path"))))


def _enclosing_symbol(symbols: Sequence[Mapping[str, Any]], lineno: int) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in symbols
        if int(row.get("start_line") or 1) <= lineno <= int(row.get("end_line") or 1)
    ]
    if not candidates:
        return None
    # 最小跨度 = 最内层；同跨度按 kind 优先级（module 最外层，最后兜底）。
    return min(
        candidates,
        key=lambda row: (
            int(row.get("end_line") or 0) - int(row.get("start_line") or 0),
            int(row.get("start_line") or 0),
        ),
    )


def _dedupe_nodes(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["repo_id"]), str(row["kind"]), str(row["qualname"]))
        seen.setdefault(key, dict(row))
    return list(seen.values())


def load_graph_inputs(store: Any, repo_id: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """读取 repo 元数据与五张派生表；repo 不存在抛 GraphValidationError。"""
    from .client import GraphValidationError

    repo = store.get_repo(repo_id)
    if repo is None:
        raise GraphValidationError(f"repo not found: {repo_id}")
    return repo, store.export_repo(repo_id)


def merge_llm_edges(graph: GraphData, llm_result: Mapping[str, Any]) -> GraphData:
    """把 llm_edges 生成的 CALLS/TESTS 行并入图数据（节点已由 AST 阶段创建）。"""
    for edge in llm_result.get("edges") or []:
        edge_type = str(edge.get("type") or "").upper()
        if edge_type not in CALLS_EDGES:
            continue
        graph["edges"][edge_type].append(dict(edge))
    graph["stats"]["llm"] = dict(llm_result.get("stats") or {})
    return graph


def _batches(rows: Sequence[Mapping[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    chunk_size = max(1, int(size))
    return [list(rows[index : index + chunk_size]) for index in range(0, len(rows), chunk_size)]


def write_graph(
    client: Any,
    graph: GraphData,
    *,
    delete_existing: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """建 schema → 清理 → UNWIND 写节点/边 → 统计；返回构建报告。"""
    started = perf_counter()
    repo_id = str(graph.get("repo_id") or "")
    ensure_schema(client)
    deleted = delete_repo_graph(client, repo_id) if delete_existing else {"nodes": {}, "edges": {}}

    for label in NODE_LABELS:
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{label} {{repo_id: row.repo_id, kind: row.kind, qualname: row.qualname}}) "
            "SET n += row.props"
        )
        for batch in _batches(graph["nodes"].get(label) or [], batch_size):
            client.execute(query, {"rows": batch})

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for edge_type in EDGE_TYPES:
        for row in graph["edges"].get(edge_type) or []:
            key = (edge_type, str(row["from_label"]), str(row["to_label"]))
            groups.setdefault(key, []).append(row)
    for (edge_type, from_label, to_label), rows in groups.items():
        query = (
            "UNWIND $rows AS row "
            f"MATCH (a:{from_label} {{repo_id: row.repo_id, kind: row.from_kind, qualname: row.from_qualname}}) "
            f"MATCH (b:{to_label} {{repo_id: row.repo_id, kind: row.to_kind, qualname: row.to_qualname}}) "
            f"MERGE (a)-[r:{edge_type} {{repo_id: row.repo_id, edge_key: row.edge_key}}]->(b) "
            "SET r += row.props"
        )
        for batch in _batches(rows, batch_size):
            client.execute(query, {"rows": batch})

    counts = graph_counts(client, repo_id)
    stats = dict(graph.get("stats") or {})
    stats.update(
        {
            "nodes": counts["nodes"],
            "edges": counts["edges"],
            "total_nodes": counts["total_nodes"],
            "total_edges": counts["total_edges"],
            "deleted": deleted.get("nodes") or {},
            "written": {
                "nodes": {label: len(graph["nodes"].get(label) or []) for label in NODE_LABELS},
                "edges": {edge_type: len(graph["edges"].get(edge_type) or []) for edge_type in EDGE_TYPES},
            },
            "ms": round((perf_counter() - started) * 1000, 1),
        }
    )
    return {"repo_id": repo_id, **stats}


__all__ = [
    "CALLS_EDGES",
    "DEFAULT_BATCH_SIZE",
    "GraphData",
    "MODULE_QUALNAME",
    "build_graph_data",
    "build_module_index",
    "display_module",
    "load_graph_inputs",
    "merge_llm_edges",
    "module_candidates",
    "normalize_path",
    "resolve_import",
    "symbol_key",
    "symbol_name",
    "write_graph",
]
