"""codegraph.builder 单元测试：AST 节点/边生成（不依赖 Neo4j）。"""

from __future__ import annotations

from conftest import build_index, make_repo, write_text

from codeagent.codegraph.builder import (
    build_graph_data,
    build_module_index,
    module_candidates,
    resolve_import,
    symbol_key,
)


def _graph(indexed) -> dict:
    repo = indexed.store.get_repo(indexed.repo_id)
    assert repo is not None
    return build_graph_data(repo, indexed.store.export_repo(indexed.repo_id))


def test_build_graph_contains_files_and_symbols(indexed) -> None:
    graph = _graph(indexed)
    file_keys = {row["qualname"] for row in graph["nodes"]["File"]}
    assert {"pkg/__init__.py", "pkg/a.py", "pkg/b.py"} <= file_keys
    symbol_keys = {row["qualname"] for row in graph["nodes"]["Symbol"]}
    assert symbol_key("pkg/b.py", "ToolExecutor") in symbol_keys
    assert symbol_key("pkg/a.py", "make") in symbol_keys
    # 每个 symbol 恰好一条 CONTAINS；模块符号也在内。
    assert len(graph["edges"]["CONTAINS"]) == len(graph["nodes"]["Symbol"])


def test_imports_resolved_to_file_and_module_nodes(tmp_path) -> None:
    root = make_repo(tmp_path / "repo")
    write_text(root / "pkg" / "c.py", "import os\nfrom json import dumps\n\n\ndef f():\n    return dumps({})\n")
    indexed = build_index(root, tmp_path)
    graph = _graph(indexed)
    imports = graph["edges"]["IMPORTS"]
    resolved = [edge for edge in imports if edge["from_qualname"] == "pkg/a.py"]
    assert resolved and resolved[0]["to_qualname"] == "pkg/b.py"
    assert resolved[0]["props"]["resolved"] is True
    module_keys = {row["qualname"] for row in graph["nodes"]["Module"]}
    assert {"os", "json"} <= module_keys
    external = [edge for edge in imports if edge["from_qualname"] == "pkg/c.py"]
    assert {edge["to_qualname"] for edge in external} >= {"os", "json"}


def test_references_resolve_symbol_and_fall_back_to_name(tmp_path) -> None:
    root = make_repo(tmp_path / "repo")
    write_text(root / "pkg" / "c.py", "def use():\n    return missing_thing()\n")
    indexed = build_index(root, tmp_path)
    graph = _graph(indexed)
    references = graph["edges"]["REFERENCES"]
    resolved = [
        edge
        for edge in references
        if edge["from_qualname"] == symbol_key("pkg/a.py", "make")
        and edge["props"]["name"] == "ToolExecutor"
    ]
    assert resolved
    assert resolved[0]["to_label"] == "Symbol"
    assert resolved[0]["to_qualname"] == symbol_key("pkg/b.py", "ToolExecutor")
    name_edges = [edge for edge in references if edge["to_label"] == "Name"]
    assert any(edge["to_qualname"] == "missing_thing" for edge in name_edges)
    name_nodes = {row["qualname"] for row in graph["nodes"]["Name"]}
    assert "missing_thing" in name_nodes
    assert graph["stats"]["unresolved_references"] >= 1


def test_duplicate_qualnames_get_path_scoped_keys(tmp_path) -> None:
    root = make_repo(tmp_path / "repo")
    write_text(root / "pkg" / "d1.py", "def _dup():\n    return 1\n")
    write_text(root / "pkg" / "d2.py", "def _dup():\n    return 2\n")
    indexed = build_index(root, tmp_path)
    graph = _graph(indexed)
    keys = {row["qualname"] for row in graph["nodes"]["Symbol"]}
    assert symbol_key("pkg/d1.py", "_dup") in keys
    assert symbol_key("pkg/d2.py", "_dup") in keys
    assert len([key for key in keys if key.endswith("::_dup")]) == 2


def test_module_candidates_and_relative_resolution() -> None:
    assert module_candidates("src/pkg/mod.py")[:2] == ["src.pkg.mod", "pkg.mod"]
    assert module_candidates("pkg/__init__.py") == ["pkg"]
    files = [{"path": "pkg/b.py"}, {"path": "pkg/sub/c.py"}]
    index = build_module_index(files)
    assert resolve_import("pkg/a.py", "b", 1, index) == "pkg/b.py"
    assert resolve_import("pkg/sub/__init__.py", "c", 1, index) == "pkg/sub/c.py"
    assert resolve_import("pkg/a.py", "os", 0, index) is None


def test_graph_is_repo_scoped(indexed, tmp_path) -> None:
    root = make_repo(tmp_path / "other")
    other = build_index(root, tmp_path)
    graph = _graph(other)
    assert all(row["repo_id"] == other.repo_id for rows in graph["nodes"].values() for row in rows)
    assert all(row["repo_id"] == other.repo_id for rows in graph["edges"].values() for row in rows)
